import torch
# from torch._six import inf
from math import inf
import math
import logging
from termcolor import colored
import sys
import os
import time


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, use_grad_scaler=True, max_grad_norm_for_update=1e4):
        self._use_grad_scaler = bool(use_grad_scaler and torch.cuda.is_available())
        self._max_grad_norm_for_update = max_grad_norm_for_update
        self._scaler = torch.amp.GradScaler('cuda') if self._use_grad_scaler else None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        if self._use_grad_scaler:
            # Keep FP16 behavior identical to the original implementation.
            self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = ampscaler_get_grad_norm(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
            return norm

        # BF16 / no-GradScaler path: keep clipping and skip bad updates.
        if not torch.isfinite(loss.detach()):
            optimizer.zero_grad(set_to_none=True)
            return torch.tensor(0.0, device=loss.device)

        if isinstance(parameters, torch.Tensor):
            param_list = [parameters]
        elif parameters is None:
            param_list = []
        else:
            param_list = list(parameters)

        loss.backward(create_graph=create_graph, retain_graph=retain_graph)

        if update_grad:
            if clip_grad is not None:
                assert param_list
                norm = torch.nn.utils.clip_grad_norm_(param_list, clip_grad)
            else:
                norm = ampscaler_get_grad_norm(param_list)

            if norm is None:
                norm = torch.tensor(0.0, device=loss.device)

            norm_value = float(norm.detach().item()) if isinstance(norm, torch.Tensor) else float(norm)
            is_nonfinite = not math.isfinite(norm_value)
            is_too_large = (
                self._max_grad_norm_for_update is not None
                and norm_value > float(self._max_grad_norm_for_update)
            )

            if is_nonfinite or is_too_large:
                optimizer.zero_grad(set_to_none=True)
                return torch.tensor(0.0, device=loss.device)

            optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        if self._use_grad_scaler:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._use_grad_scaler and state_dict:
            self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # remove any existing handlers to avoid duplicate log output
    # (some imported libraries may add handlers to the root logger)
    if logger.hasHandlers():
        logger.handlers.clear()

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger