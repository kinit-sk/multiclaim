import torch

from training.backend import Config
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from training.tricks import SAM


def get_opimizer_scheduler(model, cfg: Config):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = _get_optimizer(optimizer_grouped_parameters, cfg)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=cfg.train.num_steps,
        num_cycles=1.0    
    )
    
    if cfg.optimizer.sam:
        return SAM(optimizer, rho=cfg.optimizer.sam_rho)
    else:
        return optimizer, scheduler


def _get_optimizer(parameters, cfg: Config):
    if cfg.optimizer.name == 'adamw':
        return torch.optim.AdamW(
            parameters, 
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise Exception(f'Wrong optimizer name {cfg.optimizer.name}')