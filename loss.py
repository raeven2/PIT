# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torchvision import models
from torchmetrics.functional import structural_similarity_index_measure


# Adversarial Loss (GAN Loss)
def adversarial_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


# Pixel-Level L1 Loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def ssim_loss(pred, target):
    """
    Returns a "loss" style metric = 1.0 - SSIM.
    By default, assumes pred & target are in [0,1].
    """
    # If your images are scaled 0~1, you can set data_range=1.
    # If they are scaled 0~255, set data_range=255, etc.
    ssim_val = structural_similarity_index_measure(preds=pred,
                                                   target=target,
                                                   data_range=1.0,
                                                   )
    return 1.0 - ssim_val


# Feature Loss (Using pretrained VGG)
def feature_loss(pred, target):
    vgg = models.vgg19(pretrained=True).features.eval().to(pred.device)

    # If pred, target have shape [B, 1, H, W], repeat channels
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)    # Now [B, 3, H, W]
    if target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)

    # Forward pass through VGG
    with torch.no_grad():
        target_features = vgg(target)
    pred_features = vgg(pred)

    return F.mse_loss(pred_features, target_features)

# Style Loss (Using Gram Matrix)
def gram_matrix(x):
    batch_size, channels, height, width = x.size()
    features = x.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t()) / (channels * height * width)
    return gram


def style_loss(pred, target):
    pred_gram = gram_matrix(pred)
    target_gram = gram_matrix(target)
    return F.mse_loss(pred_gram, target_gram)


# Total Variation Loss
def total_variation_loss(x):
    tv_loss = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + torch.sum(
        torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    return tv_loss


# Knowledge Distillation Loss (combined)
def kd_loss(teacher_model, student_model, images, masks):
    # 前向传播
    with torch.no_grad():
        teacher_output = teacher_model(images)
    student_output = student_model(images)

    # 计算各个损失
    ssim = ssim_loss(student_output, teacher_output)
    feature = feature_loss(student_output, teacher_output)
    style = style_loss(student_output, teacher_output)
    tv = total_variation_loss(student_output)

    # 监督损失
    supervised = l1_loss(student_output, masks)

    # 加权求和
    lambda_ssim = 1.0
    lambda_feature = 0.5
    lambda_style = 0.5
    lambda_tv = 0.1
    lambda_supervised = 100.0

    total_kd_loss = (
            lambda_ssim * ssim +
            lambda_feature * feature +
            lambda_style * style +
            lambda_tv * tv +
            lambda_supervised * supervised
    )
    return total_kd_loss
