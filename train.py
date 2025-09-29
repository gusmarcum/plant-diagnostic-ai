import traceback
import sys
import logging
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# Trigger @registry decorators
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ---- force HF to keep the whole LLM on the local GPU for this rank ----
import transformers
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_target = f"cuda:{LOCAL_RANK}"

def _pin_device_map(kwargs):
    dm = kwargs.get("device_map", None)
    if dm is None or (isinstance(dm, str) and str(dm).lower() == "auto"):
        kwargs["device_map"] = {"": _target}
    return kwargs

if hasattr(transformers, "AutoModelForCausalLM"):
    _orig_auto = transformers.AutoModelForCausalLM.from_pretrained
    def _auto_pinned(*args, **kwargs):
        return _orig_auto(*args, **_pin_device_map(kwargs))
    transformers.AutoModelForCausalLM.from_pretrained = _auto_pinned

try:
    from transformers import LlamaForCausalLM
    _orig_llama = LlamaForCausalLM.from_pretrained
    def _llama_pinned(*args, **kwargs):
        return _orig_llama(*args, **_pin_device_map(kwargs))
    LlamaForCausalLM.from_pretrained = _llama_pinned
except Exception:
    pass
# -----------------------------------------------------------------------

if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, key-value pairs in xxx=yyy format",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))


def ensure_library_root():
    """Make sure registry has a library_root path."""
    lib = None
    try:
        lib = registry.get_path("library_root")
        if lib:
            return lib  # already registered, nothing to do
    except Exception:
        lib = None

    if not lib:
        lib = os.getenv("MINIGPT4_LIB_ROOT") or os.path.dirname(os.path.abspath(__file__))

    # Register only if not already present; ignore if some other code beat us to it.
    try:
        registry.register_path("library_root", lib)
    except KeyError:
        pass
    return lib



    # Otherwise try common hooks
    for nm in ("validation", "validate", "evaluate", "eval"):
        fn = getattr(runner, nm, None)
        if callable(fn):
            logging.info("[train.py] Trying runner.%s()", nm)
            try:
                fn(); logging.info("[train.py] runner.%s() finished.", nm); return
            except TypeError:
                try:
                    fn("val"); logging.info("[train.py] runner.%s('val') finished.", nm); return
                except Exception as e:
                    logging.warning("[train.py] runner.%s('val') failed: %s", nm, e)
                    logging.debug(traceback.format_exc())
            except Exception as e:
                logging.warning("[train.py] runner.%s() failed: %s", nm, e)
                logging.debug(traceback.format_exc())

    # Last-ditch: iterate a few val samples to prove the loader works
    try:
        ds = getattr(runner, "datasets", None)
        val = ds.get("val") if isinstance(ds, dict) else None
        if val is None:
            logging.warning("[train.py] No 'val' dataset visible; cannot run fallback iteration.")
            return
        n = 0
        for _ in val:
            n += 1
            if n >= 5:
                break
        logging.info("[train.py] Fallback iterated %d samples from 'val'.", n)
    except Exception as e:
        logging.warning("[train.py] Fallback iteration failed: %s", e)


def main():
    # set before init_distributed_mode to keep same job_id across ranks
    job_id = now()

    args = parse_args()
    cfg = Config(args)
    ensure_library_root()

    init_distributed_mode(cfg.run_cfg)
    print(f"[rank {get_rank()}] LOCAL_RANK={os.getenv('LOCAL_RANK')} -> cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name()}")
    setup_seeds(cfg)

    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="minigptv", name=cfg.run_cfg.job_name)
        wandb.watch(model)

    runner = get_runner_class(cfg)(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    #Train (RunnerBase handles val-eval each epoch if valid_splits exist)
    runner.train()


if __name__ == "__main__":
    main()
