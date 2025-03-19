# -*- coding: utf-8 -*-

import torch

def compute_dice_score(pred, target, threshold=0.5):
    smooth = 1e-6
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return dice_score.item()