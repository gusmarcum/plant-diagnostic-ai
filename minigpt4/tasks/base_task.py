"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
import wandb

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"
        self.cfg = ""
        self.current_step = 0  # Initialize the step counter


    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples, optimizer, accumulation_steps=4, max_grad_norm=1.0):
        """
        Performs a single training step with gradient accumulation and optimization.
        Returns the loss tensor for backpropagation.

        Args:
            model: The model to train
            samples: Training samples
            optimizer: The optimizer instance (passed from Runner)
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        try:
            use_mixed_precision = self.cfg.run_cfg.get('amp', True)

            # Initialize scaler if needed
            if use_mixed_precision and not hasattr(self, 'scaler'):
                self.scaler = torch.amp.GradScaler("cuda")
                logging.info("Initialized GradScaler for mixed precision training")

            # Forward pass with type validation
            with torch.amp.autocast("cuda", enabled=use_mixed_precision):
                output = model(samples)

                if not isinstance(output, dict) or "loss" not in output:
                    raise ValueError(f"Model output must be dict with 'loss' key. Got: {type(output)}")

                loss = output["loss"]

                if not isinstance(loss, torch.Tensor):
                    raise ValueError(f"Loss must be a tensor. Got type: {type(loss)}")

                if not loss.requires_grad:
                    raise ValueError("Loss tensor requires_grad=False, cannot backpropagate")

                logging.debug(f"Initial loss tensor: {loss}, requires_grad={loss.requires_grad}")

                # Scale loss for accumulation while preserving tensor properties
                loss = loss / accumulation_steps

                # Store loss value for logging before any detachment
                loss_value = loss.detach().item()

                # Log metrics if needed
                if (self.current_step + 1) % accumulation_steps == 0 and self.cfg.run_cfg.wandb_log:
                    wandb.log({
                        "step": self.current_step,
                        "loss": loss_value * accumulation_steps,  # Unscale for logging
                        "lr": optimizer.param_groups[0]["lr"],  # Use passed optimizer
                    })

            self.current_step += 1
            return loss

        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.
        """
        try:
            use_amp = scaler is not None
            accumulation_steps = self.cfg.run_cfg.get('accumulation_steps', 4)
            max_grad_norm = self.cfg.run_cfg.get('max_grad_norm', 1.0)

            if not hasattr(data_loader, "__next__"):
                data_loader = iter(data_loader)

            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
            metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

            logging.info(
                f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch."
                f" Using accumulation_steps={accumulation_steps}, max_grad_norm={max_grad_norm}"
            )

            header = "Train: data epoch: [{}]".format(epoch)
            inner_epoch = start_iters // iters_per_epoch if start_iters is not None else epoch
            if start_iters is not None:
                header = header + "; inner epoch [{}]".format(inner_epoch)

            for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
                try:
                    if i >= iters_per_epoch:
                        break

                    # Prepare samples
                    samples = next(data_loader)
                    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                    samples.update({
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    })

                    # Update learning rate
                    lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

                    # Forward pass and loss computation
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss = self.train_step(
                            model=model,
                            samples=samples,
                            optimizer=optimizer,
                            accumulation_steps=accumulation_steps,
                            max_grad_norm=max_grad_norm
                        )

                        # Validate loss is tensor
                        if not isinstance(loss, torch.Tensor):
                            raise ValueError(f"Loss must be a Tensor, got {type(loss)}. Value: {loss}")
                        logging.debug(f"Loss after train_step: {loss}, Type: {type(loss)}")

                    # Backward pass
                    if use_amp:
                        if not isinstance(loss, torch.Tensor):
                            raise ValueError(f"Cannot scale non-tensor loss: {type(loss)}")
                        scaled_loss = scaler.scale(loss)
                        logging.debug(f"Scaled loss: {scaled_loss}, Type: {type(scaled_loss)}")
                        scaled_loss.backward()
                    else:
                        loss.backward()

                    # Optimization step
                    if (i + 1) % accum_grad_iters == 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                            if max_grad_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=max_grad_norm
                                )
                                logging.debug(f"Gradient norm after clipping: {grad_norm}")
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            if max_grad_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=max_grad_norm
                                )
                            optimizer.step()

                        optimizer.zero_grad()

                        # Logging
                        if self.cfg.run_cfg.wandb_log:
                            log_dict = {
                                "epoch": inner_epoch,
                                "lr": optimizer.param_groups[0]["lr"],
                                "loss": loss.detach().item(),
                            }
                            if max_grad_norm > 0:
                                log_dict["grad_norm"] = grad_norm
                            wandb.log(log_dict)

                        # Console logging every 50 steps
                        if (i + 1) % 50 == 0:
                            lrs = [pg["lr"] for pg in optimizer.param_groups]
                            logging.info(f"step={(i+1)} grad_norm={grad_norm:.4f} lr={','.join(f'{x:.6e}' for x in lrs)}")

                    # Update metrics
                    metric_logger.update(loss=loss.detach().item())
                    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                except Exception as e:
                    logging.error(f"Error in iteration {i}: {str(e)}")
                    logging.error("Stack trace:", exc_info=True)
                    raise

            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))

            return {
                k: "{:.3f}".format(meter.global_avg)
                for k, meter in metric_logger.meters.items()
            }

        except Exception as e:
            logging.error(f"Error in training loop: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
