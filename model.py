# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# Helper function for parameter calculation
def calculate_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Convolutional Block
def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, activation="leaky_relu"):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# Transposed Convolutional Block
def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, activation="relu",
                 final=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "sigmoid" and final:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


# Teacher Model with Adjusted Upsampling Layers
class TeacherUNet(nn.Module):
    def __init__(self):
        super(TeacherUNet, self).__init__()
        ngf = 64
        self.down1 = conv_block(1, ngf, norm=False)  # 256 -> 128
        self.down2 = conv_block(ngf, ngf * 2)  # 128 -> 64
        self.down3 = conv_block(ngf * 2, ngf * 4)  # 64 -> 32
        self.down4 = conv_block(ngf * 4, ngf * 8)  # 32 -> 16
        self.down5 = conv_block(ngf * 8, ngf * 8)  # 16 -> 8
        self.down6 = conv_block(ngf * 8, ngf * 8)  # 8 -> 4
        self.down7 = conv_block(ngf * 8, ngf * 8, norm=False)  # 4 -> 2

        self.up1 = deconv_block(ngf * 8, ngf * 8)  # 2 -> 4
        self.up2 = deconv_block(ngf * 16, ngf * 8)  # 4 -> 8
        self.up3 = deconv_block(ngf * 16, ngf * 8)  # 8 -> 16
        self.up4 = deconv_block(ngf * 16, ngf * 8)  # 16 -> 32
        self.up5 = deconv_block(768, 256)  # 32 -> 64
        self.up6 = deconv_block(384, 128)  # 64 -> 128
        self.up7 = deconv_block(192, 1, final=True, activation="sigmoid")  # 128 -> 256

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)  # 256 -> 128
        d2 = self.down2(d1)  # 128 -> 64
        d3 = self.down3(d2)  # 64 -> 32
        d4 = self.down4(d3)  # 32 -> 16
        d5 = self.down5(d4)  # 16 -> 8
        d6 = self.down6(d5)  # 8 -> 4
        d7 = self.down7(d6)  # 4 -> 2

        # Decoder with skip connections
        u1 = self.up1(d7)  # 2 -> 4
        u1 = torch.cat([u1, d6], dim=1)  # 512 + 512 = 1024
        u2 = self.up2(u1)  # 4 -> 8
        u2 = torch.cat([u2, d5], dim=1)  # 512 + 512 = 1024
        u3 = self.up3(u2)  # 8 -> 16
        u3 = torch.cat([u3, d4], dim=1)  # 512 + 512 = 1024
        u4 = self.up4(u3)  # 16 -> 32
        u4 = torch.cat([u4, d3], dim=1)  # 512 + 256 = 768
        u5 = self.up5(u4)  # 32 -> 64
        u5 = torch.cat([u5, d2], dim=1)  # 256 + 128 = 384
        u6 = self.up6(u5)  # 64 -> 128
        u6 = torch.cat([u6, d1], dim=1)  # 128 + 64 = 192
        u7 = self.up7(u6)  # 128 -> 256

        return u7


# Student Model with Adjusted Upsampling Layers
class StudentUNet(nn.Module):
    def __init__(self):
        super(StudentUNet, self).__init__()
        ngf = 16  # 学生模型使用较少的滤波器
        self.down1 = conv_block(1, ngf, norm=False)  # 256 -> 128
        self.down2 = conv_block(ngf, ngf * 2)  # 128 -> 64
        self.down3 = conv_block(ngf * 2, ngf * 4)  # 64 -> 32
        self.down4 = conv_block(ngf * 4, ngf * 8)  # 32 -> 16
        self.down5 = conv_block(ngf * 8, ngf * 8)  # 16 -> 8
        self.down6 = conv_block(ngf * 8, ngf * 8)  # 8 -> 4
        self.down7 = conv_block(ngf * 8, ngf * 8, norm=False)  # 4 -> 2

        self.up1 = deconv_block(ngf * 8, ngf * 8)  # 2 -> 4
        self.up2 = deconv_block(ngf * 16, ngf * 8)  # 4 -> 8
        self.up3 = deconv_block(ngf * 16, ngf * 8)  # 8 -> 16
        self.up4 = deconv_block(ngf * 16, ngf * 8)  # 16 -> 32
        self.up5 = deconv_block(192, 64)  # 32 -> 64（调整输入通道数）
        self.up6 = deconv_block(96, 32)  # 64 -> 128
        self.up7 = deconv_block(48, 1, final=True, activation="sigmoid")  # 128 -> 256

    def forward(self, x):
        # 编码器
        d1 = self.down1(x)  # 256 -> 128
        d2 = self.down2(d1)  # 128 -> 64
        d3 = self.down3(d2)  # 64 -> 32
        d4 = self.down4(d3)  # 32 -> 16
        d5 = self.down5(d4)  # 16 -> 8
        d6 = self.down6(d5)  # 8 -> 4
        d7 = self.down7(d6)  # 4 -> 2

        # 解码器与跳跃连接
        u1 = self.up1(d7)  # 2 -> 4
        u1 = torch.cat([u1, d6], dim=1)  # 128 + 128 = 256
        u2 = self.up2(u1)  # 4 -> 8
        u2 = torch.cat([u2, d5], dim=1)  # 128 + 128 = 256
        u3 = self.up3(u2)  # 8 -> 16
        u3 = torch.cat([u3, d4], dim=1)  # 128 + 128 = 256
        u4 = self.up4(u3)  # 16 -> 32
        u4 = torch.cat([u4, d3], dim=1)  # 128 + 64 = 192
        u5 = self.up5(u4)  # 32 -> 64（现在接收到192通道）
        u5 = torch.cat([u5, d2], dim=1)  # 64 + 32 = 96
        u6 = self.up6(u5)  # 64 -> 128（接收到96通道）
        u6 = torch.cat([u6, d1], dim=1)  # 32 + 16 = 48
        u7 = self.up7(u6)  # 48 -> 256

        return u7


# Discriminator Model
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        ndf = 64
        self.model = nn.Sequential(
            conv_block(2, ndf, norm=False),  # Input: Image + Mask
            conv_block(ndf, ndf * 2),
            conv_block(ndf * 2, ndf * 4),
            conv_block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)  # Output: Patch score
        )

    def forward(self, x, y):
        # x: input image
        # y: target or generated mask
        # Concatenate along channel dimension
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)


# Initialize models
if __name__ == "__main__":
    teacher_model = TeacherUNet()
    student_model = StudentUNet()
    discriminator_model = PatchGANDiscriminator()

    # Calculate Parameters
    teacher_params = calculate_params(teacher_model)
    student_params = calculate_params(student_model)
    discriminator_params = calculate_params(discriminator_model)

    print(f"Teacher Model Parameters: {teacher_params}")
    print(f"Student Model Parameters: {student_params}")
    print(f"Discriminator Model Parameters: {discriminator_params}")
