# -*- coding: utf-8 -*-

import torch.nn.functional as F

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)