# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch import nn
from loss import kd_loss

def train_teacher_model(teacher_model, discriminator_model, train_loader, device, num_epochs=20):
    # Set models to train mode
    teacher_model.train()
    discriminator_model.train()

    # Optimizers
    optimizer_G = optim.Adam(teacher_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()

    # Loop through epochs
    for epoch in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            # #### Train Discriminator #####
            discriminator_model.zero_grad()

            # Generate fake masks
            fake_masks = teacher_model(images)

            # Real loss
            pred_real = discriminator_model(images, masks)
            real_targets = torch.ones_like(pred_real).to(device)
            loss_real = adversarial_loss_fn(pred_real, real_targets)

            # Fake loss
            pred_fake = discriminator_model(images, fake_masks.detach())
            fake_targets = torch.zeros_like(pred_fake).to(device)
            loss_fake = adversarial_loss_fn(pred_fake, fake_targets)

            # Total Discriminator loss
            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # #### Train Generator #####
            teacher_model.zero_grad()

            # Adversarial loss
            pred_fake = discriminator_model(images, fake_masks)
            real_targets = torch.ones_like(pred_fake).to(device)  # Generator tries to make discriminator believe fake is real
            loss_adv = adversarial_loss_fn(pred_fake, real_targets)

            # L1 loss
            loss_l1 = l1_loss_fn(fake_masks, masks)

            # Total Generator loss
            g_loss = loss_adv + 100 * loss_l1  # Weighting L1 loss
            g_loss.backward()
            optimizer_G.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        epoch_d_loss = running_d_loss / len(train_loader)
        epoch_g_loss = running_g_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}")

    return teacher_model


def train_student_model(student_model, teacher_model, train_loader, device, num_epochs=20):
    student_model.train()
    teacher_model.eval()
    optimizer_S = optim.Adam(student_model.parameters(), lr=5e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            loss_kd = kd_loss(teacher_model, student_model, images, masks)

            optimizer_S.zero_grad()
            loss_kd.backward()
            optimizer_S.step()

            running_loss += loss_kd.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                      f"KD Loss: {loss_kd.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Student Epoch [{epoch + 1}/{num_epochs}], KD Loss: {epoch_loss:.4f}")

    return student_model
