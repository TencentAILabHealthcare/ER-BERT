# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
from bert_data_prepare.utility import is_valid_aaseq
from data.bert_finetuning_er_seq2seq_dataset import Seq2SeqDataset
from data.bert_finetuning_er_dataset import ERDataset


class EpitopeReceptorDataset(object):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       data_dir, 
                       shuffle=True,
                       use_part=10000,
                       use_binary=True,
                       use_selected_epitopes=False,
                       neg_ratio=0.5,
                       validation_split=0.1, 
                       test_split=0.1,
                       generation_discriminator_split=0.8,
                       discriminator_ratio=True,

                       epitope_vocab_dir=None,
                       receptor_vocab_dir=None,
                       epitope_tokenizer_dir=None,
                       receptor_tokenizer_dir=None,
                       epitope_tokenizer_name='special',
                       receptor_tokenizer_name='special',
                       epitope_token_length_list='2,3',
                       receptor_token_length_list='2,3',
                       epitope_seq_name='epitope', 
                       receptor_seq_name='beta',
                       epitope_max_len=None,
                       receptor_max_len=None,
                       
                       encoder_input='epitope'):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_binary = use_binary
        self.use_part = use_part
        self.neg_ratio = neg_ratio
        self.validation_split = validation_split
        self.test_split = test_split
        self.generation_discriminator_split = generation_discriminator_split
        self.discriminator_ratio = discriminator_ratio
        self.encoder_input = encoder_input
        
        if use_selected_epitopes:
            self.selected_epitopes = self._load_selected_epitopes()
            self.logger.info(f"Using selected epitopes {self.selected_epitopes}")
        else:
            self.selected_epitopes = None

        self.epitope_seq_name = epitope_seq_name
        self.receptor_seq_name = receptor_seq_name
        self.epitope_max_len = epitope_max_len
        self.receptor_max_len = receptor_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pos_df, self.neg_df = self._load_data()

        '''Get tokenizer'''
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

    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer

    def get_receptor_tokenizer(self):
        return self.receptor_tokenizer

    def get_epitope_split_fn(self):
        return self.EpitopeTokenizer.split

    def get_receptor_split_fn(self):
        return self.ReceptorTokenizer.split

    def get_pos_df(self):
        return self.pos_df

    def get_neg_df(self):
        return self.neg_df

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

    def get_data(self):
        pos_df = self.pos_df

        if self.generation_discriminator_split:
            data_for_generation_df, data_for_discriminator_df = self._get_generation_discriminator_split(df=pos_df)
            neg_pair_df = self.__negative_sampling(pos_pair_df=data_for_discriminator_df, discriminator=True)
        else:
            data_for_generation_df, data_for_discriminator_df = None, None
            neg_pair_df = self.__negative_sampling(pos_pair_df=pos_df, discriminator=False)
        
        return pos_df, neg_pair_df, data_for_generation_df, data_for_discriminator_df

    def _load_data(self):
        if self.use_binary:
            self.logger.info("Using binary dataset.")
            df = pd.read_csv(join(self.data_dir, 'epitope_cdr3_binary.csv'))
        else:
            self.logger.info("Using continuous dataset.")
            df = pd.read_csv(join(self.data_dir, 'epitope_cdr3_reg.csv'))
        
        self.logger.info('Values are larger than 0 are treated as positive.') 
        df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
        df = df[['epitope', 'beta', 'label']]
        df = df.drop_duplicates(subset=['epitope', 'beta'])
        self.logger.info(f"After dropping duplicates, {len(df)} left.")

        pos_df = df[df['label']==1]
        neg_df = df[df['label']==0]
        self.logger.info("Dataset shape {} with {} positive and {} negative".format(
                df.shape, len(pos_df), len(neg_df)
            ))
        return pos_df, neg_df

    def _load_selected_epitopes(self):
        df = pd.read_csv(join(self.data_dir, 'selected_epitopes.csv'))
        return list(df['epitope'])

    def _get_generation_discriminator_split(self, df):
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

    def __negative_sampling(self, pos_pair_df, discriminator=False):
        if discriminator == True:
            neg_pair_save_dir = join(self.data_dir, 'split_generation_discriminator_'+str(self.discriminator_ratio),
                'neg_pair_df-onestep-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        else:
            neg_pair_save_dir = join(self.data_dir, 
                'neg_pair_df-onestep-seed-'+str(self.seed)+'-neg_ratio-'+str(self.neg_ratio)+'.csv')
        
        if exists(neg_pair_save_dir):
            self.logger.info(f'Loading existed negative pairs from {neg_pair_save_dir}')
            neg_pair_df = pd.read_csv(neg_pair_save_dir)
            self.logger.info(f'Sampling {len(neg_pair_df)} negatives')
            return neg_pair_df

        neg_df_list = []
        epitope_list = list(set(pos_pair_df['epitope']))
        for epitope in epitope_list:
            epitope_pos_df = pos_pair_df[pos_pair_df['epitope']==epitope]
            epitope_neg_df = self.neg_df[self.neg_df['epitope']==epitope]
            replace = len(epitope_pos_df)*self.neg_ratio > len(epitope_neg_df)
            epitope_neg_sample_df = epitope_neg_df.sample(
                n=int(len(epitope_pos_df)*self.neg_ratio),
                random_state=self.seed,
                replace=replace
            ).reset_index(drop=True)
            neg_df_list.append(epitope_neg_sample_df)
            self.logger.info(f"Epitope {epitope}, pos {len(epitope_pos_df)}, neg {len(epitope_neg_sample_df)}")
        neg_df = pd.concat(neg_df_list).reset_index(drop=True)

        neg_df.to_csv(neg_pair_save_dir, index=False)
        return neg_df

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
            save_dir = join(self.data_dir, 'split_generation_discriminator_'+str(self.discriminator_ratio))
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

    def get_receptor_binding_train_dataloader(self, epitope):
        self.logger.info(f'Using epitope {epitope}')
        epitope_pos_df = self.pos_df[self.pos_df['epitope']==epitope]
        epitope_neg_df = self.neg_df[self.neg_df['epitope']==epitope]
        
        if self.neg_ratio is not None:
            self.logger.info(f"Generate positive and negative pairs for {epitope} with negative ratio {self.neg_ratio}.")
            replace = len(epitope_neg_df) < len(epitope_pos_df)
            if replace:
                self.logger.info("The number of negative samples is smaller than the positive samples.")
            epitope_neg_sample_df = epitope_neg_df.sample(
                n=int(len(epitope_pos_df) * self.neg_ratio),
                random_state=self.seed,
                replace=replace
            ).reset_index(drop=True)
        else:
            self.logger.info("Using all the negative pairs.")
            epitope_neg_sample_df = epitope_neg_df

        self.logger.info(f"{len(epitope_pos_df)} positive and {len(epitope_neg_sample_df)} negative")
        df = pd.concat([epitope_pos_df, epitope_neg_sample_df]).reset_index(drop=True)

        df = df.drop_duplicates(subset=['epitope', 'beta'])
        self.logger.info(f'After dropping duplicates, {len(df)} left.')

        df = df.sample(frac=1, random_state=self.seed)
        train_df, valid_df, test_df = self._split_df(df=df)
        self.logger.info("{} in train: {} positive and {} negative".format(
            len(train_df), len(train_df[train_df['label']==1]), len(train_df[train_df['label']==0])
        ))
        self.logger.info("{} in valid: {} positive and {} negative".format(
            len(valid_df), len(valid_df[valid_df['label']==1]), len(valid_df[valid_df['label']==0])
        ))
        self.logger.info("{} in test: {} positive and {} negative".format(
            len(test_df), len(test_df[test_df['label']==1]), len(test_df[test_df['label']==0])
        ))

        train_dataloader = self.__create_dataloader(train_df)
        valid_dataloader = self.__create_dataloader(valid_df)
        test_dataloader = self.__create_dataloader(test_df)

        return train_dataloader, valid_dataloader, test_dataloader

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

    def get_binding_affinitiy_test_dataset(self):
        if exists(join(self.data_dir, 'pos_df_bsp_train.csv')):
            pos_df = pd.read_csv(join(self.data_dir, 'pos_df_bsp_train.csv'))
            neg_df = pd.read_csv(join(self.data_dir, 'neg_df_bsp_train.csv'))
            df = pd.concat([pos_df, neg_df])
            df = df.sample(frac=1, random_state=self.seed)
        else:
            if self.selected_epitopes is not None:
                self.logger.info("Using selected epitopes to test.")
                df_list = []
                for epitope in self.selected_epitopes:
                    epitope_pos_df = self.pos_df[self.pos_df['epitope']==epitope]
                    epitope_neg_df = self.neg_df[self.neg_df['epitope']==epitope]
                    if len(epitope_pos_df) > 1000:
                        df_list.append(epitope_pos_df.sample(n=1000, random_state=self.seed).reset_index(drop=True))
                        df_list.append(epitope_neg_df.sample(n=1000, random_state=self.seed).reset_index(drop=True))
                    else:
                        df_list.append(epitope_pos_df.reset_index(drop=True))
                        df_list.append(epitope_neg_df.sample(n=len(epitope_pos_df), random_state=self.seed).reset_index(drop=True))
                df = pd.concat(df_list)
                df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

            elif self.use_part is not None:
                self.logger.info(f'Using {self.use_part} to test.')
                df = self._random_sample_balance()
        
        self.logger.info("{} positive and {} negative".format(
            len(df[df['label']==1]), len(df[df['label']==0])))
    
        er_dataset = ERDataset(
            epitope_seqs=list(df['epitope']),
            receptor_seqs=list(df['beta']),
            labels=list(df['label']),
            epitope_split_fun=self.EpitopeTokenizer.split,
            receptor_split_fun=self.ReceptorTokenizer.split,
            epitope_tokenizer=self.epitope_tokenizer,
            receptor_tokenizer=self.receptor_tokenizer,
            epitope_max_len=self.epitope_max_len,
            receptor_max_len=self.receptor_max_len,
            logger=self.logger)
        
        er_dataloader = DataLoader(dataset=er_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return er_dataloader

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

    def _split_df(self, df):
        valid_num, test_num = int(len(df) * self.validation_split), int(len(df) * self.test_split)
        test_df = df.sample(n=test_num, random_state=self.seed)
        left_df = df.drop(test_df.index)
        valid_df = left_df.sample(n=valid_num, random_state=self.seed)
        train_df = left_df.drop(valid_df.index)

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, valid_df, test_df

    def _random_sample_balance(self):
        self.logger.info('Create sampled data.')
        self.logger.info(f"Negative ratio {self.neg_ratio}")
        if int((round(1-self.neg_ratio, 1))*self.use_part) > len(self.pos_df):
            replace = True
        else:
            replace = False
        pos_df = self.pos_df.sample(n=int((round(1-self.neg_ratio, 1))*self.use_part), 
                                    random_state=self.seed,
                                    replace=replace).reset_index(drop=True)
        neg_df = self.neg_df.sample(n=int(self.neg_ratio*self.use_part), random_state=self.seed).reset_index(drop=True)
        sample_df = pd.concat([pos_df, neg_df])
        sample_df = sample_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return sample_df

