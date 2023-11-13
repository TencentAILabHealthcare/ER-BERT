# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from os.path import join, exists
from tqdm import tqdm
from collections import Counter

class Vocab(object):
    def __init__(self, seq_dir,
                       token_dir, 
                       logger,
                       recreate=False,
                       use_seqs=['epitope', 'alpha', 'beta'],
                       token_len_list=[3, 4, 5],
                       keep_ratio_list=[0.9, 0.9, 0.9]):
        self.seq_dir = seq_dir
        self.token_dir = token_dir
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir)
        self.use_seqs = use_seqs
        self.logger = logger
        self.token_len_list = token_len_list
        self.keep_ratio_list = keep_ratio_list
        self.recreate = recreate

        self.seq_list = self._load_seq()
        if self.recreate:
            self.logger.info('Recreating token list...')
            self.token_df = None
        else:
            self.token_df = self._load_token()
        self._create_vocab()

    def _load_seq(self):
        seq_list = []
        if 'epitope' in self.use_seqs:
            epitope_df = pd.read_csv(join(self.seq_dir, 'epitope.csv'))
            seq_list += list(epitope_df['epitope'])
            del epitope_df
        if 'alpha' in self.use_seqs:
            alpha_df = pd.read_csv(join(self.seq_dir, 'alpha.csv'))
            seq_list += list(alpha_df['alpha'])
            del alpha_df
        if 'beta' in self.use_seqs:
            beta_df = pd.read_csv(join(self.seq_dir, 'beta.csv'))
            seq_list += list(beta_df['beta'])
            del beta_df
        
        assert seq_list != [], "No sequence data is loaded!"
        self.logger.info('{} sequences collected from {}'.format(
            len(seq_list), self.use_seqs))
        return seq_list

    def _load_token(self):
        token_df_list = []
        if 'epitope' in self.use_seqs:
            if not exists(join(self.token_dir, 'total-epitope.csv')):
                self.logger.info('The token file for epitope is not existed!')
            else:
                epitope_df = pd.read_csv(join(self.token_dir, 'total-epitope.csv'), na_filter=False)
                token_df_list.append(epitope_df)
                del epitope_df
        if 'alpha' in self.use_seqs:
            if not exists(join(self.token_dir, 'total-alpha.csv')):
                self.logger.info('The token file for alpha is not existed!')
            else:
                alpha_df = pd.read_csv(join(self.token_dir, 'total-alpha.csv'), na_filter=False)
                token_df_list.append(alpha_df)
                del alpha_df
        if 'beta' in self.use_seqs:
            if not exists(join(self.token_dir, 'total-beta.csv')):
                self.logger.info('The token file for beta is not existed!')
            else:
                beta_df = pd.read_csv(join(self.token_dir, 'total-beta.csv'), na_filter=False)
                token_df_list.append(beta_df)
                del beta_df
        
        if len(token_df_list) == 0:
            return None
        else:
            return pd.concat(token_df_list)

    def _get_token_count(self, token_len, token_df=None):
        if token_df is not None:
            token_len_df = token_df[token_df['token'].str.len() == token_len]
        else:
            token_list = []
            for seq in tqdm(self.seq_list):
                token_list += self._split2tokens(seq=seq, token_len=token_len)
            token_count = dict(Counter(token_list))
            token_len_df = pd.DataFrame({'token': list(token_count.keys()),
                                         'frequency': list(token_count.values())})

        self.logger.info('{} tokens with length {}'.format(
            len(token_len_df), token_len))
        
        return token_len_df

    def _get_top_count(self, token_count_df, keep_ratio):
        df = token_count_df
        df.sort_values(by=['frequency'], ascending=False, inplace=True)
        
        temp_freq = 0
        total_freq = sum(df['frequency'])
        keep_thre_freq = total_freq * keep_ratio
        for i, row in df.iterrows():
            if temp_freq > keep_thre_freq:
                break
            temp_freq += row['frequency']

        df = df.loc[: i, :]

        # z-score normalized
        df['freq_z_normalized'] = (df['frequency'] - df['frequency'].mean()) / df['frequency'].std()
        df.dropna(inplace=True)

        return df

    def _create_vocab(self):
        total_token_df_list = []
        selected_token_df_list = []

        for idx, token_len in enumerate(self.token_len_list):
            temp_token_count_df = self._get_token_count(token_len, token_df=self.token_df)
            total_token_df_list.append(temp_token_count_df)

            # selcted token
            temp_selected_token_df = self._get_top_count(token_count_df=temp_token_count_df,
                                                         keep_ratio=self.keep_ratio_list[idx])
            selected_token_df_list.append(temp_selected_token_df)
            self.logger.info('By keeping the {} frequency of the data with length {}, {} tokens are kept.'.format(
                self.keep_ratio_list[idx], token_len, len(temp_selected_token_df)))
        
        total_token_df = pd.concat(total_token_df_list)
        total_token_save_name = join('../../ProcessedData/vocab', 'total-' + '-'.join(self.use_seqs) + '.csv') 
        if self.recreate or not os.path.exists(total_token_save_name):
            total_token_df.to_csv(total_token_save_name, index=False)
        self.logger.info('In total, {} tokens are generated.'.format(len(total_token_df)))

        save_name = '-'.join(self.use_seqs) + '-' + '-'.join([str(v) for v in self.token_len_list])
        selected_token_df = pd.concat(selected_token_df_list)
        self.logger.info('In total, {} tokens are selected.'.format(len(selected_token_df)))
        selected_token_df.to_csv(join('../../ProcessedData/vocab', save_name+'.csv'), index=False)            

    def _split2tokens(self, seq, token_len):
        if type(seq) == str and len(seq) < token_len:
            return []
        taa_list = []
        start, end = 0, token_len
        while end <= len(seq):
            taa_list.append(seq[start: end])
            start += 1
            end += 1
        return taa_list
