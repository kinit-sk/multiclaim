import argparse
import copy
import glob
import logging
import os
import random
import yaml
from functools import partial
from statistics import mean, median, StatisticsError
from typing import Dict, List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from training.backend import Config, init_ctx_from_wandb_config
from training.loss import MNRloss
from training.model import get_model_tokenizer
from training.optimizer import get_opimizer_scheduler
from training.tricks import SAM, compute_ema, freeze_layers
from datasets.dataset import Dataset as BaseDataset
from datasets.dataset_factory import dataset_factory
from evaluation import (PytorchVectorizer, embedding_results,
                            process_result_generator)

BATCH = Tuple[Dict[str, torch.Tensor]]


def safe_mean(x, round_to=4):
    try:
        return round(mean(x), round_to)
    except StatisticsError:
        return None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch, tokenizer, cfg: Config):
    p_text, fc_text = zip(*batch)
    return (
        tokenizer(list(p_text), padding=True, truncation=True, max_length=cfg.train.p_max_seq_length, return_tensors='pt'),
        tokenizer(list(fc_text), padding=True, truncation=True, max_length=cfg.train.fc_max_seq_length, return_tensors='pt')
    )


def train_step(step, model, batch: BATCH, loss_fn, optimizer, scheduler, scaler, ctx_autocast, cfg: Config):
    model.train()
    posts_encoded, fact_checks_encoded = batch
    posts_encoded, fact_checks_encoded = posts_encoded.to(cfg.train.device), fact_checks_encoded.to(cfg.train.device)
    
    def fw_bw():
        optimizer.zero_grad(set_to_none=True)
        with ctx_autocast:
            loss = loss_fn(model(**fact_checks_encoded), model(**posts_encoded))
        scaler.scale(loss).backward()
        if cfg.optimizer.clip_value is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_value)
        return loss  
      
    if isinstance(optimizer, SAM):
        step_type = ('first', 'second') if (step % cfg.optimizer.sam_step) == 0 else ('skip',)
        for s in step_type:
            loss = fw_bw()
            scaler.step(optimizer, step_type=s)
            scaler.update()
    else:
        loss = fw_bw()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    return loss.item()


def evaluate_datasets(datasets, model, tokenizer, run_path, save_vectors):
    model.eval()
    results = {}
        
    vectorizer_path = run_path if save_vectors else None
    vct = PytorchVectorizer(dir_path=vectorizer_path, model=model, tokenizer=tokenizer, batch_size=256, port_embeddings_to_cpu=True)
    
    for dataset_name, dataset in datasets.items():
        result_gen = embedding_results(dataset, vct, vct, post_split_size=512, device='cuda')
        metrics = process_result_generator(result_gen, csv_path=os.path.join(run_path, dataset_name + '.csv'))
        for value_name, values in metrics.items():
            v_mean, v_ci_lower, v_ci_upper = values
            results[f'{dataset_name}_{value_name}_mean'] = v_mean
            results[f'{dataset_name}_{value_name}_lower'] = v_ci_lower 
            results[f'{dataset_name}_{value_name}_upper'] = v_ci_upper  
    return results


def train(cfg, train_dataset, dev_datasets, test_datasets):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Supervised')

    set_seed(cfg.seed)
    run = wandb.init(
        config=cfg.to_dict(), project=cfg.wandb.project, name=cfg.wandb.name, 
        mode='online' if cfg.wandb.logging else 'disabled'
    )
    init_ctx_from_wandb_config(cfg, run.config)

    if cfg.train.output_path is not None:
        run_path = os.path.join(cfg.train.output_path, cfg.timestamp)
        os.makedirs(run_path)
        # save config
        with open(os.path.join(run_path, 'config.yaml') ,'w') as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    logger.info('Get model')
    model, tokenizer = get_model_tokenizer(cfg.model.name, pooling=cfg.model.pooling)
    model = model.to(cfg.train.device)

    # Only if EMA is used
    if cfg.train.ema:
        ema_model = copy.deepcopy(model)

    logger.info('Prepare dataloader')
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, cfg=cfg)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=partial_collate_fn, 
        pin_memory=True,
        drop_last=True
    )
    cfg.train.num_steps = len(train_dataloader) * cfg.train.num_epochs
    # If evaluation is done after each epoch
    if cfg.train.evaluation_steps == -1:
        cfg.train.evaluation_steps = len(train_dataloader)

    logger.info('Prepare optimizer and loss')
    optimizer, scheduler = get_opimizer_scheduler(model, cfg)
    loss_fn = MNRloss(cfg.train.label_smoothing)
    
    logger.info('Train')
    step = 0
    training_loss = []

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.train.dtype]
    ctx_autocast = torch.autocast(device_type=cfg.train.device.split(':')[0], dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.dtype == 'float16'))

    for epoch in tqdm(range(cfg.train.num_epochs), desc='Training'):
        for batch in tqdm(train_dataloader, desc='Batch'):
            step += 1

            train_loss_value = train_step(step, model, batch, loss_fn, optimizer, scheduler, scaler, ctx_autocast, cfg)
            training_loss.append(train_loss_value)
            
            # Loss reporting
            if (step % cfg.train.metrics_window_size) == 0: 
                agg_loss = median(training_loss)
                tqdm.write(
                    f'step {step}/{cfg.train.num_steps}: train/loss-med{cfg.train.metrics_window_size} {agg_loss:.4f}'
                )
                run.log({
                    f'train/loss-med{cfg.train.metrics_window_size}': train_loss_value, 
                    'lr': optimizer.param_groups[0]['lr']
                    }, step=step
                )
                training_loss = []

            # Only if EMA is used
            if cfg.train.ema:
                compute_ema(model, ema_model, smoothing=0.99)
            
            # Evaluation reporting
            if (step % cfg.train.evaluation_steps == 0) or (cfg.train.num_steps == step):
                step_path = os.path.join(run_path, str(step))
                os.makedirs(step_path)

                eval_model = ema_model if cfg.train.ema else model
                    
                results = evaluate_datasets({**dev_datasets, **test_datasets}, eval_model, tokenizer, step_path, save_vectors=cfg.train.save_vectors)
                run.log(results, step=step, commit=True)
                for dataset_name in [*dev_datasets.keys(), *test_datasets.keys()]:
                    artifact = wandb.Artifact(name=f'{dataset_name}_{step}', type='dataset')
                    artifact.add_file(local_path=os.path.join(step_path, f'{dataset_name}.csv'))
                    run.log_artifact(artifact)

                mean_dev_ps = safe_mean([results[f'{dataset_name}_pair_success_at_10_mean'] for dataset_name in dev_datasets.keys()])
                mean_test_ps = safe_mean([results[f'{dataset_name}_pair_success_at_10_mean'] for dataset_name in test_datasets.keys()])
                tqdm.write(
                    f'step {step}/{cfg.train.num_steps}: dev/mean-ps@10 {mean_dev_ps} test/mean-ps@10 {mean_test_ps}'
                )

                if cfg.train.save_models:
                    torch.save(eval_model.state_dict(), os.path.join(step_path, 'model.pt'))

        # Applying layer freezing at the end of the epoch
        if cfg.train.freezing:
            _, feeze_level = freeze_layers(
                model=model, optimizers=optimizer, 
                current_duration=epoch / cfg.train.num_epochs, 
                freeze_start=0.0, freeze_level=1.0
            )
            run.log({'step': step, 'feeze_level': feeze_level})

    run.finish()


if __name__ == '__main__':
    train()
