# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel

class BERTBinding(nn.Module):
    def __init__(self, EpitopeBert_dir, ReceptorBert_dir, emb_dim):
        super().__init__()
        self.EpitopeBert = BertModel.from_pretrained(EpitopeBert_dir)
        self.ReceptorBert = BertModel.from_pretrained(ReceptorBert_dir)
        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
            nn.Tanh(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, epitope, receptor):
        epitope_encoded = self.EpitopeBert(**epitope).last_hidden_state
        receptor_encoded = self.ReceptorBert(**receptor).last_hidden_state

        '''
        Using the cls (classification) token as the input to get the score if borrowed
        from huggingface NextSentencePrediciton implementation
        https://github.com/huggingface/transformers/issues/7540
        https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html
        '''
        # shape: [batch_size, emb_dim]
        epitope_cls = epitope_encoded[:, 0, :]
        receptor_cls = receptor_encoded[:, 0, :]
        # shape: [batch_size, emb_dim*2]
        concated_encoded = torch.concat((epitope_cls, receptor_cls), dim=1)
        # shape: [batch_size, 1]
        output = self.binding_predict(concated_encoded)

        return output