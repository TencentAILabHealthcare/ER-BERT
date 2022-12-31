#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import torch
import numpy as np

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback)

import data.test_10X_dataset as module_data
from model.bert_seq2seq import get_EncoderDecoder_model
from model.metric import Seq2Seq_metrics
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    model_input = config['model']['TransformerVariant'].split('-')[0].lower()
    assert config['data_loader']['args']['encoder_input'] == model_input, "The input of dataloader is different from model input!"

    config['data_loader']['args']['logger'] = logger
    dataset = config.init_obj('data_loader', module_data)
    if config['data_loader']['args']['generation_discriminator_split']:
        logger.info('Split the data for generation and discriminator separately...')
        _, neg_discriminator_df, data_for_generation_df, data_for_discriminator_df = dataset.get_data()
        train_dataset, valid_dataset = dataset.get_seq2seq_train_dataset(df=data_for_generation_df)
    else:
        logger.info('Using all the original data...')
        pos_df, neg_df, _, _ = dataset.get_data()
        train_dataset, valid_dataset = dataset.get_seq2seq_train_dataset(df=pos_df)
    
    epitope_tokenizer = dataset.get_epitope_tokenizer()
    receptor_tokenizer = dataset.get_receptor_tokenizer()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config._save_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['trainer']['epochs'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        learning_rate=config['trainer']['lr'],
        warmup_ratio=config['trainer']['warmup'],
        evaluation_strategy="epoch",
        eval_accumulation_steps=config['trainer']['eval_accumulation_steps'] if 'eval_accumulation_steps' in config['trainer'] else None,
        per_device_eval_batch_size=config['trainer']['batch_size'],
        logging_strategy="steps",
        logging_steps=config['trainer']['logging_steps'],
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=1,
        load_best_model_at_end=True,
        no_cuda=False,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=True,
        metric_for_best_model='acc',
        logging_dir=config._log_dir,
        predict_with_generate=True)

    model = get_EncoderDecoder_model(
        logger=logger,
        TransformerVariant=config['model']['TransformerVariant'],
        EpitopeBert_dir=config['model']['EpitopeBert_dir'],
        ReceptorBert_dir=config['model']['ReceptorBert_dir'],
        epitope_tokenizer=epitope_tokenizer,
        receptor_tokenizer=receptor_tokenizer,
        epitope_max_len=config['data_loader']['args']['epitope_max_len'],
        receptor_max_len=config['data_loader']['args']['receptor_max_len'],
        resume=config['model']['resume'] if "resume" in config['model'] else None
    )
    logger.info(model)

    trainable_params = model.parameters()
    params = sum([np.prod(p.size()) for p in trainable_params if p.requires_grad])
    logger.info(f'Trainable parameters {params}.')

    logger.info('Setting parameters for beam search decoding')
    model.config.early_stopping = config['model']['beam_search']['early_stopping']
    model.config.num_beams = config['model']['beam_search']['num_beams']
    model.config.no_repeat_ngram_size = config['model']['beam_search']['no_repeat_ngram_size']

    seq2seq_metrics = Seq2Seq_metrics(
        logger=logger,
        model_variant=config['model']['TransformerVariant'],
        epitope_tokenizer=epitope_tokenizer,
        receptor_tokenizer=receptor_tokenizer,
        blosum_dir=config['metrics']['blosum_dir'],
        blosum=config['metrics']['blosum']
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Defaults to None, see above
        compute_metrics=seq2seq_metrics.compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(config._save_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
