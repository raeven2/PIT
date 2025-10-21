# -*- coding: utf-8 -*-

import torch
from metrics import compute_dice_score

def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    dice_scores = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass through model
            pred_masks = model(images)

            # Compute Dice score or other evaluation metrics
            dice_score_val = compute_dice_score(pred_masks, masks)
            dice_scores.append(dice_score_val)

    mean_dice_score = sum(dice_scores) / len(dice_scores)
    print(f"Mean Dice Score: {mean_dice_score}")
    return mean_dice_score

def compute_dice_score(pred_masks, true_masks, threshold=0.5):
    smooth = 1e-6
    pred_masks = (pred_masks > threshold).float()
    true_masks = (true_masks > threshold).float()

    intersection = torch.sum(pred_masks * true_masks)
    union = torch.sum(pred_masks) + torch.sum(true_masks)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return dice_score.item()
