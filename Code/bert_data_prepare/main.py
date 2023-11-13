#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from dataset import BERTDataPrepare
from vocab import Vocab
from tokenizer import FMFMTokenizer

logging.basicConfig(level=logging.DEBUG, filename='BERT_create_vocab.log', filemode='a',
                    format="%(asctime)s - %(name)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('BERT')

myBERTDataPrepare = BERTDataPrepare(logger=logger)

'''['VDJdb', 'IEDB-Receptor', 'IEDB-Epitope',
    'GenePlus-Cancer', 'GenePlus-COVID', 'MAB-COVID',
    'TCRdb', 'PIRD', 'Glanville',
    'Dash', 'McPAS', 'NetTCR', 'huARdb']'''
df = myBERTDataPrepare.get_dataset(data_list=['Dash', 'Glanville', 'McPAS', 'NetTCR', 'PIRD', 'VDJdb'])

EpitopeVocab = Vocab(seq_dir='../../ProcessedData/merged',
                     token_dir='../../ProcessedData/vocab',
                     logger=logger,
                     recreate=True,
                     use_seqs=['epitope'],
                     token_len_list=[2, 3],
                     keep_ratio_list=[0.9]*2)
del EpitopeVocab

BetaVocab = Vocab(seq_dir='../../ProcessedData/merged',
                  token_dir='../../ProcessedData/vocab',
                  logger=logger,
                  recreate=True,
                  use_seqs=['beta'],
                  token_len_list=[2, 3],
                  keep_ratio_list=[0.9]*2)
del BetaVocab

AlphaVocab = Vocab(seq_dir='../../ProcessedData/merged',
                   token_dir='../../ProcessedData/vocab',
                   logger=logger,
                   recreate=True,
                   use_seqs=['alpha'],
                   token_len_list=[2, 3],
                   keep_ratio_list=[1.0]*2)
