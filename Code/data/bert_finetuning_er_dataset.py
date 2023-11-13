# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
from base import BaseDataLoader
from bert_data_prepare.utility import is_valid_aaseq


class ERDataset(Dataset):
    def __init__(self, epitope_seqs,
                       receptor_seqs,
                       labels,
                       epitope_split_fun,
                       receptor_split_fun,
                       epitope_tokenizer,
                       receptor_tokenizer,
                       epitope_max_len,
                       receptor_max_len,
                       logger):
        self.epitope_seqs = epitope_seqs
        self.receptor_seqs = receptor_seqs
        self.labels = labels
        self.epitope_split_fun = epitope_split_fun
        self.receptor_split_fun = receptor_split_fun
        self.epitope_tokenizer = epitope_tokenizer
        self.receptor_tokenizer = receptor_tokenizer
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        epitope, receptor = self.epitope_seqs[i], self.receptor_seqs[i]
        label = self.labels[i]
        epitope_tensor = self.epitope_tokenizer(self._insert_whitespace(self.epitope_split_fun(epitope)),
                                                padding="max_length",
                                                max_length=self.epitope_max_len,
                                                truncation=True,
                                                return_tensors="pt")
        receptor_tensor = self.receptor_tokenizer(self._insert_whitespace(self.receptor_split_fun(receptor)),
                                                  padding="max_length",
                                                  max_length=self.receptor_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")
        label_tensor = torch.FloatTensor(np.atleast_1d(label))

        epitope_tensor = {k: torch.squeeze(v) for k, v in epitope_tensor.items()}
        receptor_tensor = {k: torch.squeeze(v) for k,v in receptor_tensor.items()}

        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized epitope: {epitope} -> {epitope_tensor}")
            self.logger.info(f"Example of tokenized receptor: {receptor} -> {receptor_tensor}")
            self.logger.info(f"Example of label: {label} -> {label_tensor}")
            self._has_logged_example = True

        return epitope_tensor, receptor_tensor, label_tensor

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class EpitopeReceptorDataset(BaseDataLoader):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       validation_split,
                       test_split,
                       num_workers,
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
                       epitope_seq_name='epitope', 
                       receptor_seq_name='beta',
                       test_epitopes=100,
                       neg_ratio=1.0,
                       shuffle=True,
                       epitope_max_len=None,
                       receptor_max_len=None):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.seq_dir = seq_dir
        self.neg_pair_save_dir = neg_pair_save_dir
        self.using_dataset = list(using_dataset.split(','))
        self.epitope_seq_name = epitope_seq_name
        self.receptor_seq_name = receptor_seq_name

        self.test_epitopes = test_epitopes
        self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        train_valid_pair_df, test_pair_df = self._split_dataset()

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

        train_valid_er_dataset = self._get_dataset(pair_df=train_valid_pair_df)
        super().__init__(train_valid_er_dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)

        test_dataset = self._get_dataset(pair_df=test_pair_df)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer

    def get_receptor_tokenizer(self):
        return self.receptor_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        er_dataset = ERDataset(epitope_seqs=list(pair_df[self.epitope_seq_name]),
                               receptor_seqs=list(pair_df[self.receptor_seq_name]),
                               labels=list(pair_df['label']),
                               epitope_split_fun=self.EpitopeTokenizer.split,
                               receptor_split_fun=self.ReceptorTokenizer.split,
                               epitope_tokenizer=self.epitope_tokenizer,
                               receptor_tokenizer=self.receptor_tokenizer,
                               epitope_max_len=self.epitope_max_len,
                               receptor_max_len=self.receptor_max_len,
                               logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        if exists(join(self.neg_pair_save_dir, 'unseen_epitopes-seed-'+str(self.seed)+'.csv')):
            test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_epitopes-seed-'+str(self.seed)+'.csv'))
            self.logger.info(f'Loading created unseen epitopes for test with shape {test_pair_df.shape}')
        else:
            epitope_list = list(set(self.pair_df['epitope']))
            selected_epitope_index_list = self.rng.integers(len(epitope_list), size=self.test_epitopes)
            self.logger.info(f'Select {self.test_epitopes} from {len(epitope_list)} epitopes')
            selected_epitopes = [epitope_list[i] for i in selected_epitope_index_list]
            test_pair_df = self.pair_df[self.pair_df['epitope'].isin(selected_epitopes)]
            test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_epitopes-seed-'+str(self.seed)+'.csv'), index=False)

        selected_epitopes = list(set(test_pair_df['epitope']))
        train_valid_pair_df = self.pair_df[~self.pair_df['epitope'].isin(selected_epitopes)]
            
        self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_save_dir = join(self.neg_pair_save_dir, 'pair_df-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        if exists(pair_save_dir):
            self.logger.info(f'Loading created pair dataframe with seed {self.seed} and negative ratio {self.neg_ratio}')
            return pd.read_csv(pair_save_dir)
        
        # load positive pairs
        if exists(join(self.neg_pair_save_dir, 'pos_pair.csv')):
            pos_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'))
            self.logger.info(f'Loading created positive pairs with {len(pos_pair_df)} records')
        else:
            pos_pair_df = self._load_seq_pairs()

        neg_pair_df = self._negative_sampling(pos_pair_df=pos_pair_df)
        
        pair_df = pd.concat([pos_pair_df, neg_pair_df])
        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        if not exists(pair_save_dir):
            pair_df.to_csv(pair_save_dir, index=False)

        return pair_df

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

    def _negative_sampling(self, pos_pair_df):
        neg_pair_save_dir = join(self.neg_pair_save_dir, 
            'neg_pair_df-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        
        if exists(neg_pair_save_dir):
            self.logger.info(f'Loading existed negative pairs from {neg_pair_save_dir}')
            neg_pair_df = pd.read_csv(neg_pair_save_dir)
            self.logger.info(f'Sampling {len(neg_pair_df)} negatives')
            return neg_pair_df

        self.epitope_seq_list, self.receptor_seq_list = self._load_full_seqs()
        assert self.neg_ratio >= 0, "Negative ratio is smaller than 0"
        num_negs = int(round(len(pos_pair_df) * self.neg_ratio))
        self.logger.info(f'Samping {num_negs} negatives')

        pos_pair_filter_df = pos_pair_df.drop(['label'], axis=1)
        neg_pairs = []
        pos_pairs = set(list(pos_pair_filter_df.itertuples(index=False, name=None)))
        self.logger.info(f'Positive pairs example {list(pos_pairs)[0]}')
        
        epitope_length = len(self.epitope_seq_list)
        receptor_length = len(self.receptor_seq_list)

        self.logger.info('Negative sampling step 1: for each epitope, get equal negative samples...')
        pos_epitope_list = list(set(pos_pair_filter_df['epitope']))
        for epitope in tqdm(pos_epitope_list):
            sample_num = len(pos_pair_filter_df[pos_pair_filter_df['epitope']==epitope])
            for i in range(sample_num):
                receptor_idx = self.rng.integers(receptor_length, size=1)[0]
                if len(set([epitope, self.receptor_seq_list[receptor_idx]]).intersection(pos_pairs)) != 0:
                    continue
                else:
                    neg_pairs.append([epitope, self.receptor_seq_list[receptor_idx]])
        
        self.logger.info('Negative sampling step 2: sample from all epitopes...')
        i = 0
        pbar = tqdm(total=num_negs+1)
        while i < num_negs:
            epitope_idx = self.rng.integers(epitope_length, size=1)[0]
            receptor_idx = self.rng.integers(receptor_length, size=1)[0]
            if len(set([self.epitope_seq_list[epitope_idx], self.receptor_seq_list[receptor_idx]]).intersection(pos_pairs)) != 0:
                continue
            else:
                neg_pairs.append([self.epitope_seq_list[epitope_idx], self.receptor_seq_list[receptor_idx]])
                i += 1
                pbar.update(1)
        pbar.close()

        neg_pair_df = pd.DataFrame({self.epitope_seq_name: [p[0] for p in neg_pairs],
                                    self.receptor_seq_name: [p[1] for p in neg_pairs],
                                    'label': [0] * len(neg_pairs)})
        neg_pair_df.to_csv(neg_pair_save_dir, index=False)
        return neg_pair_df