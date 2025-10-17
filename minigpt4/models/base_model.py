import os
local_rank = int(os.environ.get("LOCAL_RANK","0"))
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
try:
    # peft <= 0.4 used this name
    from peft import prepare_model_for_int8_training
except ImportError:
    # peft >= 0.5 renamed it
    from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training

from minigpt4.common.dist_utils import download_cached_file
from minigpt4.common.utils import get_abs_path, is_url
from minigpt4.models.eva_vit import create_eva_vit_g
from minigpt4.models.modeling_llama import LlamaForCausalLM



class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[-1].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze
    ):
        logging.info('Loading VIT')
        logging.info(f'Image size: {img_size}, Precision: {precision}')

        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        if not freeze:
            precision = "fp32"  # fp16 is not for training
            logging.info(f'Training mode - setting precision to {precision}')

        try:
            visual_encoder = create_eva_vit_g(
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_checkpoint=use_grad_checkpoint,
                precision=precision
            )
            logging.info(f'Successfully created EVA-ViT with img_size={img_size}')
        except Exception as e:
            logging.error(f'Error creating EVA-ViT: {str(e)}')
            raise

        ln_vision = LayerNorm(visual_encoder.num_features)

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("Vision encoder frozen")
        else:
            logging.info("Vision encoder trainable")

        logging.info('VIT Loading Complete')
        return visual_encoder, ln_vision

    def init_llm(
        self,
        llama_model_path,
        low_resource: bool = False,
        low_res_device: int = 0,
        lora_r: int = 0,
        lora_target_modules=("q_proj", "v_proj"),
        **lora_kargs,
    ):
        logging.info("Loading LLAMA")
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        llama_tokenizer.pad_token = "$$"

        quant_config = None
        if low_resource:
            # prefer the modern quantization path
            try:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                logging.warning(
                    "Requested 8-bit but BitsAndBytesConfig unavailable; falling back to fp16 (%s)", e
                )
                low_resource = False

        if low_resource:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                quantization_config=quant_config,
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        # Attach LoRA if requested
        if lora_r and lora_r > 0:
            # On k-bit models this is helpful; safe to try.
            try:
                llama_model = prepare_model_for_int8_training(llama_model)
            except Exception:
                pass

            loraconfig = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_kargs.get("lora_alpha", 16)),
                lora_dropout=float(lora_kargs.get("lora_dropout", 0.1)),
                bias="none",
                target_modules=list(lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            try:
                llama_model.print_trainable_parameters()
            except Exception:
                pass
        else:
            # No LoRA → freeze all base LLaMA params here.
            for _, p in llama_model.named_parameters():
                p.requires_grad = False

        logging.info("Loading LLAMA Done")
        return llama_model, llama_tokenizer



    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 safely."""
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        # do LN in fp32 (on the same device), using fp32 weight/bias too
        out = torch.nn.functional.layer_norm(
            x.to(dtype=torch.float32),
            self.normalized_shape,
            self.weight.to(dtype=torch.float32) if self.weight is not None else None,
            self.bias.to(dtype=torch.float32) if self.bias is not None else None,
            self.eps,
        )
        return out.to(dtype=orig_dtype)




