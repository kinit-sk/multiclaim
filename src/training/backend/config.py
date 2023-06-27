from dataclasses import dataclass, asdict, field
import datetime
import os
from typing import Any, Dict, Optional
import yaml


@dataclass
class Model:
    name: str = 'sentence-transformers/all-mpnet-base-v2'
    pooling: str = 'default'  # 'mean', 'default'


@dataclass
class Train:
    device: str = 'cuda' 
    dtype: str = 'float32'  # float32, float16, bfloat16
    batch_size: int = 36 
    num_epochs: int = 20
    metrics_window_size: int = 128
    warmup_steps: int = 400
    evaluation_steps: int = -1  # If -1, after each epoch
    label_smoothing: float = 0.  # [0, 1]
    ema: bool = False  # Exponential moving average
    freezing: bool = False  # Gradually makes early modules untrainable
    output_path: Optional[str] = os.path.join('/', 'labs', 'cache', 'runs')  # If None, no model checkpointing every evaluation_steps
    save_vectors: bool = False  # Whether to save vectors for evaluation
    save_models: bool = False # Whether to save models
    p_max_seq_length: int = 512
    fc_max_seq_length: int = 128


@dataclass
class Optimizer:
    name: str = 'adamw'  # adamw, shampoo_fb, shampoo_google
    lr: float = 5e-5
    weight_decay: float = 0.005 
    clip_value: float = 1.0  # if None, no gradient clipping
    sam: bool = False
    sam_step: float = 1.  # Will be ignored if sam is False, propability of sam step
    sam_rho: float = 0.05  # if sam_step=10 then rho=0.5,

    
@dataclass
class WandB:
    project: str = 'test'
    name: str = ''
    logging: bool = True

    
@dataclass
class Config:
    model: Model = Model()
    train: Train = Train()
    optimizer: Optimizer = Optimizer()
    wandb: WandB = WandB()

    seed: int = 3407

    def __init__(self, path: Optional[str] = None):
        self.timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.wandb.name = self.timestamp
        if path is not None: 
            with open(path, 'r') as f: self.init_class(yaml.load(f, Loader=yaml.SafeLoader))
        
    def to_dict(self):
        return asdict(self)
    
    def init_class(self, d):
        for name in dir(self):
            if name.startswith('_') or name.endswith('_') or name not in d:
                continue
            attr = getattr(self, name)
            if isinstance(d[name], dict):
                for k, v in d[name].items():
                    setattr(attr, k, v)
            else: 
                setattr(self, name, d[name])


def init_ctx_from_wandb_config(cfg: Config, wandb_config: Dict) -> None:
    cfg_dict = {}
    for param_name, param in wandb_config.items():
        if '.' not in param_name:
            continue
        inner_cfg = cfg_dict
        split_name = param_name.split('.')
        for s in split_name[:-1]:
            if s not in inner_cfg:
                inner_cfg[s] = {}
            inner_cfg = inner_cfg[s]
        inner_cfg[split_name[-1]] = param

    cfg.init_class(cfg_dict)