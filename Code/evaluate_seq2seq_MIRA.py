#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import argparse
import collections
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

from transformers import (
    EncoderDecoderModel)

import data.test_MIRA_dataset as module_data
import model.bert_binding as module_arch
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

    model_input = config['TransformerVariant'].split('-')[0].lower()

    config['data_loader']['args']['logger'] = logger
    dataset_test = config.init_obj('data_loader', module_data)
    if config['data_loader']['args']['generation_discriminator_split']:
        logger.info('Split the data for generation and discriminator separately...')
        _, neg_discriminator_df, data_for_generation_df, data_for_discriminator_df = dataset_test.get_data()
        use_dataset = data_for_discriminator_df
    else:
        logger.info('Using all the original data...')
        pos_df, neg_df, _, _ = dataset_test.get_data()
        use_dataset = pos_df

    epitope_tokenizer = dataset_test.get_epitope_tokenizer()
    receptor_tokenizer = dataset_test.get_receptor_tokenizer()
    epitope_split_fn = dataset_test.get_epitope_split_fn()
    receptor_split_fn = dataset_test.get_receptor_split_fn()

    # load model
    logger.info(f"Loading pre-trained model from {config.resume}")
    model = EncoderDecoderModel.from_pretrained(config.resume).to("cuda")

    log_example = []

    def seq_generate(input_seq, max_length, input_split_fn, input_tokenizer, target_tokenizer, beams, gene_num=1000):
        input_tokenized = input_tokenizer(" ".join(input_split_fn(input_seq)),
                                          padding="max_length",
                                          max_length=max_length,
                                          truncation=True,
                                          return_tensors="pt")
        input_ids = input_tokenized.input_ids.to("cuda")
        attention_mask = input_tokenized.attention_mask.to("cuda")

        outputs = model.generate(input_ids, 
                                 attention_mask=attention_mask,
                                 num_beams=beams,
                                 num_return_sequences=gene_num)
        output_str = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_str_nospace = [s.replace(" ", "") for s in output_str]
        output_str_nospace = [s for s in output_str_nospace if s != '']

        if len(log_example) == 0:
            logger.info(f'output {outputs[0]} -> output string {output_str_nospace[0]}')
            log_example.append(1)

        return output_str_nospace

    if model_input == 'epitope':
        logger.info('The input is epitope, generate receptor sequences.')
        logger.info('To generate {} sequences.'.format(
                len(set(use_dataset['epitope'])) * 1000
            ))
        result_dict = {'epitope': [], 'generated_beta': []}
        for epitope in tqdm(list(set(use_dataset['epitope']))):
            predict_seq = seq_generate(input_seq=epitope, 
                                       max_length=config['data_loader']['args']['epitope_max_len'],
                                       input_split_fn=epitope_split_fn,
                                       input_tokenizer=epitope_tokenizer,
                                       target_tokenizer=receptor_tokenizer,
                                       beams=1000,
                                       gene_num=1000)

            result_dict['epitope'] += [epitope] * len(predict_seq)
            result_dict['generated_beta'] += predict_seq
    
    else:
        logger.info('The input is receptor, generate epitope sequences.')
        result_dict = {'beta': [], 'generated_epitope': []}
        
        epitope_list = list(set(use_dataset['epitope']))
        used_epitope_list = []
        selected_beta_list = []
        random.seed(0)
        for epitope in epitope_list:
            beta_list = list(set(use_dataset[use_dataset['epitope']==epitope]['beta']))
            if len(beta_list) > 100:
                selected_beta = random.choices(beta_list, k=100)
            else:
                selected_beta = beta_list
            used_epitope_list += [epitope] * len(selected_beta)
            selected_beta_list += selected_beta
        selected_beta_df = pd.DataFrame({'epitope': used_epitope_list,
                                         'beta': selected_beta_list})
        selected_beta_df.to_csv(join(config._log_dir, 'selected_beta.csv'), index=False)
        logger.info('For each epitope, random sample 100 betas for generation.')
        logger.info('In total, {} epitopes needs to be generated'.format(len(selected_beta_list)*5))

        for beta in tqdm(list(selected_beta_list)):
            predict_seq = seq_generate(input_seq=beta, 
                                       max_length=config['data_loader']['args']['receptor_max_len'],
                                       input_split_fn=receptor_split_fn,
                                       input_tokenizer=receptor_tokenizer,
                                       target_tokenizer=epitope_tokenizer,
                                       beams=5,
                                       gene_num=5)

            result_dict['beta'] += [beta] * len(predict_seq)
            result_dict['generated_epitope'] += predict_seq

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(join(config._log_dir, 'result.csv'), index=False)

    logger.info('\nUsing the trained discriminator to check the performance of the generated sequences.')
    config['data_loader_discriminator']['args']['logger'] = logger
    dataset_discriminator = config.init_obj('data_loader_discriminator', module_data)
    if config['data_loader']['args']['generation_discriminator_split']:
        logger.info('Split the data for generation and discriminator separately...')
        _, neg_discriminator_df, data_for_generation_df, data_for_discriminator_df = dataset_discriminator.get_data()
        true_data_loader, test_data_loader = dataset_discriminator.get_seq2seq_dataloader(
            pos_df=data_for_discriminator_df,
            neg_df=neg_discriminator_df,
            model_input=model_input,
            result_df=result_df
        )
    else:
        logger.info('Using all the original data...')
        pos_df, neg_df, _, _ = dataset_discriminator.get_data()
        true_data_loader, test_data_loader = dataset_discriminator.get_seq2seq_dataloader(
            pos_df=pos_df,
            neg_df=neg_df,
            model_input=model_input,
            result_df=result_df
        )

    dis_epitope_tokenizer = dataset_discriminator.get_epitope_tokenizer()
    dis_receptor_tokenizer = dataset_discriminator.get_receptor_tokenizer()
    
    del model
    external_discriminator = config.init_obj('discriminator', module_arch)
    logger.info('Loading checkpint from {}'.format(
        config['discriminator_resume']))
    checkpoint = torch.load(config['discriminator_resume'], map_location="cuda:0")
    state_dict = checkpoint['state_dict']
    external_discriminator.load_state_dict(state_dict)
    external_discriminator.to("cuda")

    def test(data_loader):
        result_dict = {'epitope': [], 'receptor': [], 'label_pred': [], 'label_true': []}
        with torch.no_grad():
            for batch_idx, (epitope_tokenized, receptor_tokenized, target) in enumerate(data_loader):
                epitope_tokenized = {k: v.to("cuda") for k, v in epitope_tokenized.items()}
                receptor_tokenized = {k: v.to("cuda") for k, v in receptor_tokenized.items()}
                output = external_discriminator(epitope_tokenized, receptor_tokenized)

                epitope = dis_epitope_tokenizer.batch_decode(epitope_tokenized['input_ids'],
                                                         skip_special_tokens=True)
                epitope = [s.replace(" ", "") for s in epitope]
                receptor = dis_receptor_tokenizer.batch_decode(receptor_tokenized['input_ids'],
                                                           skip_special_tokens=True)
                receptor = [s.replace(" ", "") for s in receptor]
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()

                result_dict['epitope'].append(epitope)
                result_dict['receptor'].append(receptor)
                result_dict['label_pred'].append(y_pred)
                result_dict['label_true'].append(target.cpu().detach().numpy())
        
        result_dict['epitope'] = [v for l in result_dict['epitope'] for v in l]
        result_dict['receptor'] = [v for l in result_dict['receptor'] for v in l]
        result_dict['label_pred'] = list(np.concatenate(result_dict['label_pred']).flatten())
        result_dict['label_true'] = list(np.concatenate(result_dict['label_true']).flatten())
        return pd.DataFrame(result_dict)

    logger.info('Discriminator on the true dataset.')
    true_discriminator_df = test(data_loader=true_data_loader)
    true_discriminator_df.to_csv(join(config._log_dir, 'true_discriminator.csv'), index=False)
    
    logger.info('Discriminator on the generated dataset.')
    test_discriminator_df = test(data_loader=test_data_loader)
    test_discriminator_df.to_csv(join(config._log_dir, 'generated_discriminator.csv'), index=False)


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