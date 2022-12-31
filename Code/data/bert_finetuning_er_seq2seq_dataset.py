# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from bert_data_prepare.tokenizer import get_tokenizer
from bert_data_prepare.utility import is_valid_aaseq


class Seq2SeqDataset(Dataset):
    def __init__(self, epitope_seqs,
                       receptor_seqs,
                       epitope_split_fun,
                       receptor_split_fun,
                       epitope_tokenizer,
                       receptor_tokenizer,
                       encoder_input,
                       epitope_max_len,
                       receptor_max_len,
                       logger):
        self.epitope_seqs = epitope_seqs
        self.receptor_seqs = receptor_seqs
        self.epitope_split_fun = epitope_split_fun
        self.receptor_split_fun = receptor_split_fun
        self.epitope_tokenizer = epitope_tokenizer
        self.receptor_tokenizer = receptor_tokenizer
        self.encoder_input = encoder_input
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.logger = logger
        self._has_logged_example = False

        self.logger.info(f"The input to the encoder is {encoder_input}")

    def __len__(self):
        return len(self.epitope_seqs)
        
    def __getitem__(self, i):
        epitope, receptor = self.epitope_seqs[i], self.receptor_seqs[i]

        input_data = {}
        epitope_tensor = self.epitope_tokenizer(self._insert_whitespace(self.epitope_split_fun(epitope)),
                                                padding="max_length",
                                                max_length=self.epitope_max_len,
                                                truncation=True)
        receptor_tensor = self.receptor_tokenizer(self._insert_whitespace(self.receptor_split_fun(receptor)),
                                                  padding="max_length",
                                                  max_length=self.receptor_max_len,
                                                  truncation=True)

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        if self.encoder_input == 'epitope':
            input_data['input_ids'] = epitope_tensor['input_ids']
            input_data['attention_mask'] = epitope_tensor['attention_mask']
            input_data['labels'] = receptor_tensor.input_ids.copy()
            input_data['labels'] = [-100 if token == self.receptor_tokenizer.pad_token_id else token for token in input_data['labels']]
        
        elif self.encoder_input == 'receptor':
            input_data['input_ids'] = receptor_tensor['input_ids']
            input_data['attention_mask'] = receptor_tensor['attention_mask']
            input_data['labels'] = epitope_tensor['input_ids'].copy()
            input_data['labels'] = [-100 if token == self.epitope_tokenizer.pad_token_id else token for token in input_data['labels']]
        else:
            self.logger.info("Wrong encoder input!")
        
        input_data = {k: torch.tensor(v, dtype=torch.long) for k, v in input_data.items()}

        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized epitope: {epitope} -> {epitope_tensor['input_ids']}")
            self.logger.info(f"Example of tokenized receptor: {receptor} -> {receptor_tensor['input_ids']}")
            self.logger.info(f"Example of input_ids {input_data['input_ids']}")
            self.logger.info(f"Example of label: {input_data['labels']}")
            self._has_logged_example = True

        return input_data

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class EpitopeReceptorSeq2SeqDataset(object):
    def __init__(self, logger, 
                       seed,
                       data_dir, 
                       seq_dir,
                       neg_pair_save_dir, 
                       using_dataset, 
                       
                       epitope_vocab_dir,
                       receptor_vocab_dir,
                       epitope_tokenizer_dir,
                       receptor_tokenizer_dir,
                       epitope_tokenizer_name='UAA',
                       receptor_tokenizer_name='UAA',
                       epitope_token_length_list='2,3',
                       receptor_token_length_list='2,3',
                       epitope_max_len=None,
                       receptor_max_len=None,

                       valid_split=0.1,
                       epitope_seq_name='epitope', 
                       receptor_seq_name='beta',
                       encoder_input='epitope',
                       shuffle=True):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.seq_dir = seq_dir
        self.neg_pair_save_dir = neg_pair_save_dir
        self.using_dataset = list(using_dataset.split(','))
        self.epitope_seq_name = epitope_seq_name
        self.receptor_seq_name = receptor_seq_name
        self.valid_split = valid_split
        self.encoder_input = encoder_input

        self.shuffle = shuffle
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        train_pair_df, valid_pair_df, test_pair_df = self._split_dataset()
        self.valid_pair_df = valid_pair_df
        self.test_pair_df = test_pair_df

        self.logger.info(f'Creating {epitope_seq_name} tokenizer...')
        self.EpitopeTokenizer = get_tokenizer(tokenizer_name=epitope_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=epitope_vocab_dir,
                                              token_length_list=epitope_token_length_list)
        self.epitope_tokenizer = self.EpitopeTokenizer.get_bert_tokenizer(
            max_len=self.epitope_max_len, 
            tokenizer_dir=epitope_tokenizer_dir)

        self.logger.info(f'Creating {receptor_seq_name} tokenizer...')
        self.ReceptorTokenizer = get_tokenizer(tokenizer_name=receptor_tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=receptor_vocab_dir,
                                               token_length_list=receptor_token_length_list)
        self.receptor_tokenizer = self.ReceptorTokenizer.get_bert_tokenizer(
            max_len=self.receptor_max_len,
            tokenizer_dir=receptor_tokenizer_dir)

        self.train_dataset = self._get_dataset(pair_df=train_pair_df)
        self.valid_dataset = self._get_dataset(pair_df=valid_pair_df)
        self.test_dataset = self._get_dataset(pair_df=test_pair_df)

    def get_epitope_split_fn(self):
        return self.EpitopeTokenizer.split

    def get_receptor_split_fn(self):
        return self.ReceptorTokenizer.split

    def get_valid_pair_df(self):
        return self.valid_pair_df

    def get_test_pair_df(self):
        return self.test_pair_df

    def get_train_dataset(self):
        return self.train_dataset

    def get_valid_dataset(self):
        return self.valid_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer

    def get_receptor_tokenizer(self):
        return self.receptor_tokenizer

    def _get_dataset(self, pair_df):
        er_dataset = Seq2SeqDataset(epitope_seqs=list(pair_df[self.epitope_seq_name]),
                                    receptor_seqs=list(pair_df[self.receptor_seq_name]),
                                    epitope_split_fun=self.EpitopeTokenizer.split,
                                    receptor_split_fun=self.ReceptorTokenizer.split,
                                    epitope_tokenizer=self.epitope_tokenizer,
                                    receptor_tokenizer=self.receptor_tokenizer,
                                    encoder_input=self.encoder_input,
                                    epitope_max_len=self.epitope_max_len,
                                    receptor_max_len=self.receptor_max_len,
                                    logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_epitopes-seed-'+str(self.seed)+'.csv'))
        self.logger.info(f'Loading unseen epitopes with shape {test_pair_df.shape}')
        test_pair_df = test_pair_df[test_pair_df['label']==1]
        test_pair_df.drop_duplicates(inplace=True)
        test_pair_df.drop(['label'], axis=1, inplace=True)
        self.logger.info(f'After processing, epitopes with shape {test_pair_df.shape}')

        train_valid_pair_df = self.pair_df[~self.pair_df['epitope'].isin(list(test_pair_df['epitope']))]
        self.logger.info('Removing the test epitopes, {} epitopes are used to train and valid.'.format(
            len(set(train_valid_pair_df['epitope']))))

        train_pair_df, valid_pair_df = train_test_split(train_valid_pair_df,
            test_size=self.valid_split, random_state=self.seed)
        self.logger.info(f"{len(train_pair_df)} train and {len(valid_pair_df)} valid.")

        return train_pair_df, valid_pair_df, test_pair_df

    def _create_pair(self):
        # load positive pairs
        if exists(join(self.neg_pair_save_dir, 'pos_pair.csv')):
            pos_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'))
            self.logger.info(f'Loading created positive pairs with {len(pos_pair_df)} records')
        else:
            pos_pair_df = self._load_seq_pairs()
        pos_pair_df.drop(['label'], axis=1, inplace=True)

        if self.shuffle:
            self.logger.info('Shuffling dataset.')
            pos_pair_df = pos_pair_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return pos_pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.epitope_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'epitope': 'epitope'}
        keep_columns = [column_map_dict[c] for c in [self.epitope_seq_name, self.receptor_seq_name]]
        
        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.epitope_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter

    def _load_full_seqs(self):
        self.logger.info(f'Loading full seqs list of {self.epitope_seq_name} and {self.receptor_seq_name}')
        epitope_seq_df = pd.read_csv(join(self.seq_dir, self.epitope_seq_name+'.csv'))
        epitope_seq_list = list(epitope_seq_df[self.epitope_seq_name])

        receptor_seq_df = pd.read_csv(join(self.seq_dir, self.receptor_seq_name+'.csv'))
        receptor_seq_list = list(receptor_seq_df[self.receptor_seq_name])

        return epitope_seq_list, receptor_seq_list