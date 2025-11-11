import copy
import torch.optim as optim
# from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist


def is_main_process():
    return dist.get_rank() == 0


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model

    if config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY,
                                betas=(0.9, 0.98), eps=1e-8, )
    if config.TRAIN.OPTIMIZER == 'sgd':
        print('Using SGD optimizer')
        optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR,
                              weight_decay=config.TRAIN.WEIGHT_DECAY,
                              momentum=0.99)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY,
                                betas=(0.9, 0.98), eps=1e-8, )

    return optimizer

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    t_initial = int(n_iter_per_epoch * config.TRAIN.COS_DECAY_EPOCHS)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=t_initial,
        lr_min=config.TRAIN.LR * config.TRAIN.COS_DECAY_LR_MULTIPLIER,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        interval=int(n_iter_per_epoch * config.TRAIN.STEP_DECAY_INTERVAL_EPOCHS),
        decay_rate=config.TRAIN.STEP_DECAY_MULTIPLIER,
    )

    return lr_scheduler


import math
from typing import List
from torch.optim import Optimizer
from timm.scheduler.scheduler import Scheduler
import logging
_logger = logging.getLogger(__name__)

class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            interval=None,  
            decay_rate=None,  
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        self.interval = interval  
        self.decay_rate = decay_rate  
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                if self.interval and self.decay_rate:
                    extra_cycles = (t - self.cycle_limit * self.t_initial) // self.interval
                    decay_factor = self.decay_rate ** extra_cycles
                    lrs = [self.lr_min * decay_factor for lr in lr_max_values]
                else:
                    lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))
