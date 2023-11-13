#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join

import data.bert_finetuning_er_dataset as module_data
# import data.bert_finetuning_er_alphabeta_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.bert_binding as module_arch
from trainer.bert_finetuning_er_trainer import BERTERTrainer as Trainer
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

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.get_test_dataloader()
    logger.info('Number of pairs in train: {}, valid: {}, and test: {}.'.format(
        data_loader.sampler.__len__(), 
        valid_data_loader.sampler.__len__(), 
        test_data_loader.sampler.__len__()
    ))
    epitope_tokenizer = data_loader.get_epitope_tokenizer()
    receptor_tokenizer = data_loader.get_receptor_tokenizer()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    logger.info(model)
    
    trainable_params = model.parameters()
    optimizer = config.init_obj('optimizer', transformers, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # save two berts
    epitope_bert_save_dir = join(config.save_dir, 'EpitopeBert')
    receptor_bert_save_dir = join(config.save_dir, 'ReceptorBert')
    logger.info(f'Saving two berts to {epitope_bert_save_dir} and {receptor_bert_save_dir}')
    os.makedirs(epitope_bert_save_dir)
    model.EpitopeBert.save_pretrained(epitope_bert_save_dir)
    os.makedirs(receptor_bert_save_dir)
    model.ReceptorBert.save_pretrained(receptor_bert_save_dir)

    test_output = trainer.test(epitope_tokenizer=epitope_tokenizer,
                               receptor_tokenizer=receptor_tokenizer)
    logger.info(test_output)

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
