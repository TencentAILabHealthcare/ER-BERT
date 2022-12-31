# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from os.path import join, exists
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
from data.bert_finetuning_er_dataset import ERDataset
from data.bert_finetuning_er_seq2seq_dataset import Seq2SeqDataset
from bert_data_prepare.utility import is_valid_aaseq


class EpitopeReceptorDataset(object):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       data_dir, 
                       seq_dir,
                       shuffle=True,
                       neg_ratio=1.0,
                       validation_split=0.1, 
                       test_split=0.1,
                       generation_discriminator_split=False,
                       discriminator_ratio=None,

                       epitope_vocab_dir=None,
                       receptor_vocab_dir=None,
                       epitope_tokenizer_dir=None,
                       receptor_tokenizer_dir=None,
                       epitope_tokenizer_name='UAA',
                       receptor_tokenizer_name='UAA',
                       epitope_token_length_list='2,3',
                       receptor_token_length_list='2,3',
                       epitope_max_len=None,
                       receptor_max_len=None,
                       
                       encoder_input='epitope',
                       epitope_seq_name='epitope', 
                       receptor_seq_name='beta'):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.seq_dir = seq_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.neg_ratio = neg_ratio
        self.validation_split = validation_split
        self.test_split = test_split
        self.generation_discriminator_split = generation_discriminator_split
        self.discriminator_ratio = discriminator_ratio

        self.encoder_input = encoder_input
        self.epitope_seq_name = epitope_seq_name
        self.receptor_seq_name = receptor_seq_name
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        '''Get tokenizer'''
        self.logger.info(f'Creating {epitope_seq_name} tokenizer...')
        self.EpitopeTokenizer = get_tokenizer(tokenizer_name=epitope_tokenizer_name,
                                              logger=self.logger,
                                              add_hyphen=False,
                                              vocab_dir=epitope_vocab_dir,
                                              token_length_list=epitope_token_length_list)
        self.epitope_tokenizer = self.EpitopeTokenizer.get_bert_tokenizer(
            max_len=self.epitope_max_len, 
            tokenizer_dir=epitope_tokenizer_dir)

        self.logger.info(f'Creating {receptor_seq_name} tokenizer...')
        self.ReceptorTokenizer = get_tokenizer(tokenizer_name=receptor_tokenizer_name,
                                               logger=self.logger,
                                               add_hyphen=False,
                                               vocab_dir=receptor_vocab_dir,
                                               token_length_list=receptor_token_length_list)
        self.receptor_tokenizer = self.ReceptorTokenizer.get_bert_tokenizer(
            max_len=self.receptor_max_len,
            tokenizer_dir=receptor_tokenizer_dir)

    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer

    def get_receptor_tokenizer(self):
        return self.receptor_tokenizer

    def get_epitope_split_fn(self):
        return self.EpitopeTokenizer.split

    def get_receptor_split_fn(self):
        return self.ReceptorTokenizer.split

    def get_data(self):
        pos_df, neg_df, data_for_generation_df, data_for_discriminator_df = self._load_data()
        return pos_df, neg_df, data_for_generation_df, data_for_discriminator_df

    def get_seq2seq_dataloader(self, pos_df, neg_df, model_input, result_df):
        self.logger.info('{} unique epitopes in {} positive pairs'.format(
            len(set(pos_df['epitope'])), len(pos_df)))
        self.logger.info("Get negative pairs.")
        true_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
        true_data_loader = self.__create_dataloader(df=true_df)

        result_df['label'] = [-1] * len(result_df)
        if model_input == 'epitope':
            test_data_loader = self.__create_dataloader(
                df=result_df,
                epitope_seq_name='epitope',
                receptor_seq_name='generated_beta'
            )
        else:
            test_data_loader = self.__create_dataloader(
                df=result_df,
                epitope_seq_name='generated_epitope',
                receptor_seq_name='beta'
            )

        return true_data_loader, test_data_loader

    def get_seq2seq_train_dataset(self, df):
        train_df, valid_df = train_test_split(df, test_size=0.01, random_state=self.seed)
        train_dataset = Seq2SeqDataset(epitope_seqs=list(train_df['epitope']),
                                        receptor_seqs=list(train_df['beta']),
                                        epitope_split_fun=self.EpitopeTokenizer.split,
                                        receptor_split_fun=self.ReceptorTokenizer.split,
                                        epitope_tokenizer=self.epitope_tokenizer,
                                        receptor_tokenizer=self.receptor_tokenizer,
                                        encoder_input=self.encoder_input,
                                        epitope_max_len=self.epitope_max_len,
                                        receptor_max_len=self.receptor_max_len,
                                        logger=self.logger)
        valid_dataset = Seq2SeqDataset(epitope_seqs=list(valid_df['epitope']),
                                        receptor_seqs=list(valid_df['beta']),
                                        epitope_split_fun=self.EpitopeTokenizer.split,
                                        receptor_split_fun=self.ReceptorTokenizer.split,
                                        epitope_tokenizer=self.epitope_tokenizer,
                                        receptor_tokenizer=self.receptor_tokenizer,
                                        encoder_input=self.encoder_input,
                                        epitope_max_len=self.epitope_max_len,
                                        receptor_max_len=self.receptor_max_len,
                                        logger=self.logger)  
    
        return train_dataset, valid_dataset

    def get_binding_affinity_train_dataloader(self, pos_df, neg_df):
        pos_train_df, pos_valid_df, pos_test_df = self.__split_df(df=pos_df)
        neg_train_df, neg_valid_df, neg_test_df = self.__split_df(df=neg_df)
        train_df = pd.concat([pos_train_df, neg_train_df])
        valid_df = pd.concat([pos_valid_df, neg_valid_df])
        test_df = pd.concat([pos_test_df, neg_test_df])

        if self.generation_discriminator_split:
            save_dir = join(self.data_dir, 'split_generation_discriminator')
        else:
            save_dir = self.data_dir
        train_df.to_csv(join(save_dir, 'train-seed'+str(self.seed)+'-neg_ratio'+str(self.neg_ratio)+'.csv'), index=False)
        valid_df.to_csv(join(save_dir, 'valid-seed'+str(self.seed)+'-neg_ratio'+str(self.neg_ratio)+'.csv'), index=False)
        test_df.to_csv(join(save_dir, 'test-seed'+str(self.seed)+'-neg_ratio'+str(self.neg_ratio)+'.csv'), index=False)

        self.logger.info(f"{len(pos_train_df)} pos and {len(neg_train_df)} neg in train.")
        self.logger.info(f"{len(pos_valid_df)} pos and {len(neg_valid_df)} neg in valid.")
        self.logger.info(f"{len(pos_test_df)} pos and {len(neg_test_df)} neg in test.")

        train_dataloader = self.__create_dataloader(train_df)
        valid_dataloader = self.__create_dataloader(valid_df)
        test_dataloader = self.__create_dataloader(test_df)

        return train_dataloader, valid_dataloader, test_dataloader

    def get_generation_discriminator_split(self, df):
        save_dir = join(self.data_dir, 'split_generation_discriminator_'+str(self.discriminator_ratio))
        if exists(join(save_dir, 'data_for_generation.csv')):
            self.logger.info(f'Loading existed data from {save_dir} for generation and discriminator...')
            data_for_generation_df = pd.read_csv(join(save_dir, 'data_for_generation.csv'))
            data_for_discriminator_df = pd.read_csv(join(save_dir, 'data_for_discriminator.csv'))
            
            self.logger.info('{} data for generation and {} data for discriminator'.format(
                len(data_for_generation_df), len(data_for_discriminator_df)))
            return data_for_generation_df, data_for_discriminator_df
        else:
            os.makedirs(save_dir)

        self.logger.info('Spliting the data for generation and discriminator separately...')
        epitope_list = list(set(df['epitope']))
        data_for_generation_list, data_for_discriminator_list = [], []
        for epitope in epitope_list:
            epitope_df = df[df['epitope']==epitope]

            data_for_generation_df = epitope_df.sample(frac=1 - self.discriminator_ratio, 
                                                       random_state=self.seed).reset_index(drop=True)
            data_for_discriminator_df = epitope_df.sample(frac=self.discriminator_ratio,
                                                          random_state=self.seed).reset_index(drop=True)

            data_for_generation_list.append(data_for_generation_df)
            data_for_discriminator_list.append(data_for_discriminator_df)

        data_for_generation_df = pd.concat(data_for_generation_list)
        data_for_discriminator_df = pd.concat(data_for_discriminator_list)
        
        self.logger.info('{} data for generation and {} data for discriminator'.format(
                len(data_for_generation_df), len(data_for_discriminator_df)))
        data_for_generation_df.to_csv(join(save_dir, 'data_for_generation.csv'), index=False)
        data_for_discriminator_df.to_csv(join(save_dir, 'data_for_discriminator.csv'), index=False)
        return data_for_generation_df, data_for_discriminator_df

    def _load_data(self):
        pos_pair_df = pd.read_csv(join(self.data_dir, 'full.csv'))
        column_rename_dict = {'epitope': 'epitope', 'cdr3b': 'beta'}
        pos_pair_df.rename(columns=column_rename_dict, inplace=True)
        pos_pair_df['label'] = [1] * len(pos_pair_df)
        self.logger.info(f'{len(pos_pair_df)} positive epitope-beta pairs')

        pos_pair_df = pos_pair_df[(pos_pair_df['epitope'].map(is_valid_aaseq)) &
                                  (pos_pair_df['beta'].map(is_valid_aaseq))]
        self.logger.info(f'After droping invalid sequence, {len(pos_pair_df)} left.')

        pos_pair_df = pos_pair_df.drop_duplicates(subset=['beta'])
        self.logger.info(f'Only keep the unique beta, {len(pos_pair_df)} left.')

        if self.generation_discriminator_split:
            data_for_generation_df, data_for_discriminator_df = self.get_generation_discriminator_split(df=pos_pair_df)
            neg_pair_df = self.__negative_sampling(pos_pair_df=data_for_discriminator_df, discriminator=True)
        else:
            data_for_generation_df, data_for_discriminator_df = None, None
            neg_pair_df = self.__negative_sampling(pos_pair_df=pos_pair_df, discriminator=False)
        
        return pos_pair_df, neg_pair_df, data_for_generation_df, data_for_discriminator_df

    def __load_full_seqs(self):
        self.logger.info(f'Loading full seqs list of epitope, alpha and beta')
        epitope_seq_df = pd.read_csv(join(self.seq_dir, 'epitope.csv'))
        epitope_seq_list = list(epitope_seq_df['epitope'])

        beta_seq_df = pd.read_csv(join(self.seq_dir, 'beta.csv'))
        beta_seq_list = list(beta_seq_df['beta'])

        return epitope_seq_list, beta_seq_list
        
    def __negative_sampling(self, pos_pair_df, discriminator=False):
        if discriminator == True:
            neg_pair_save_dir = join(self.data_dir, 'split_generation_discriminator_'+str(self.discriminator_ratio),
                'neg_pair_df-twosteps-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        else:
            neg_pair_save_dir = join(self.data_dir, 
                'neg_pair_df-twosteps-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        
        if exists(neg_pair_save_dir):
            self.logger.info(f'Loading existed negative pairs from {neg_pair_save_dir}')
            neg_pair_df = pd.read_csv(neg_pair_save_dir)
            self.logger.info(f'Sampling {len(neg_pair_df)} negatives')
            return neg_pair_df

        self.epitope_seq_list, self.beta_seq_list = self.__load_full_seqs()
        assert self.neg_ratio >= 0, "Negative ratio is smaller than 0"
        num_negs = int(round(len(pos_pair_df) * self.neg_ratio))
        self.logger.info(f'Samping {num_negs} negatives')

        pos_pair_filter_df = pos_pair_df.drop(['label'], axis=1)
        neg_pairs = []
        pos_pairs = set(list(pos_pair_filter_df.itertuples(index=False, name=None)))
        self.logger.info(f'Positive pairs example {list(pos_pairs)[0]}')

        epitope_length = len(self.epitope_seq_list)
        beta_length = len(self.beta_seq_list)

        self.logger.info('Negative sampling step 1: for each epitope, get equal negative samples...')
        pos_epitope_list = list(set(pos_pair_filter_df['epitope']))
        for epitope in tqdm(pos_epitope_list):
            sample_num = len(pos_pair_filter_df[pos_pair_filter_df['epitope']==epitope])
            for i in range(sample_num):
                beta_idx = self.rng.integers(beta_length, size=1)[0]
                if len(set([epitope, self.beta_seq_list[beta_idx]]).intersection(pos_pairs)) != 0:
                    continue
                else:
                    neg_pairs.append([epitope, self.beta_seq_list[beta_idx]])

        self.logger.info('Negative sampling step 2: sample from all epitopes...')
        i = 0
        pbar = tqdm(total=num_negs+1)
        while i < num_negs:
            epitope_idx = self.rng.integers(epitope_length, size=1)[0]
            beta_idx = self.rng.integers(beta_length, size=1)[0]
            if self.epitope_seq_list[epitope_idx] in pos_epitope_list:
                continue
            if len(set([self.epitope_seq_list[epitope_idx], self.beta_seq_list[beta_idx]]).intersection(pos_pairs)) != 0:
                continue
            else:
                neg_pairs.append([self.epitope_seq_list[epitope_idx], self.beta_seq_list[beta_idx]])
                i += 1
                pbar.update(1)
        pbar.close()

        neg_pair_df = pd.DataFrame({'epitope': [p[0] for p in neg_pairs],
                                    'beta': [p[1] for p in neg_pairs],
                                    'label': [0] * len(neg_pairs)})
        neg_pair_df.to_csv(neg_pair_save_dir, index=False)
        return neg_pair_df

    def __create_dataloader(self, df, epitope_seq_name='epitope', receptor_seq_name='beta'):
        dataset = ERDataset(
            epitope_seqs=list(df[epitope_seq_name]),
            receptor_seqs=list(df[receptor_seq_name]),
            labels=list(df['label']),
            epitope_split_fun=self.EpitopeTokenizer.split,
            receptor_split_fun=self.ReceptorTokenizer.split,
            epitope_tokenizer=self.epitope_tokenizer,
            receptor_tokenizer=self.receptor_tokenizer,
            epitope_max_len=self.epitope_max_len,
            receptor_max_len=self.receptor_max_len,
            logger=self.logger
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def __split_df(self, df):
        valid_num, test_num = int(len(df) * self.validation_split), int(len(df) * self.test_split)
        test_df = df.sample(n=test_num, random_state=self.seed)
        left_df = df.drop(test_df.index)
        valid_df = left_df.sample(n=valid_num, random_state=self.seed)
        train_df = left_df.drop(valid_df.index)

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, valid_df, test_df

