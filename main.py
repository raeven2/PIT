# -*- coding: utf-8 -*-

import torch
from model import TeacherUNet, StudentUNet, PatchGANDiscriminator
from data_loader import get_dataloader
from train import train_teacher_model, train_student_model
from PIT_UNet.test import test_model

def main():
    # Hyperparameters
    image_dir = "Dataset_UNet/train/images"
    mask_dir = "Dataset_UNet/train/masks"
    batch_size = 32
    num_epochs_teacher = 20
    num_epochs_student = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_loader = get_dataloader(image_dir, mask_dir, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(image_dir, mask_dir, batch_size=batch_size, shuffle=False)

    # Initialize models
    teacher_model = TeacherUNet().to(device)
    student_model = StudentUNet().to(device)
    discriminator_model = PatchGANDiscriminator().to(device)

    # Train Teacher Model
    print("Training Teacher Model...")
    teacher_model = train_teacher_model(teacher_model, discriminator_model, train_loader, device,
                                        num_epochs=num_epochs_teacher)

    # Save the teacher model
    torch.save(teacher_model.state_dict(), "TeacherModelTestCase/teacher_model.pth")
    print("Teacher model saved as 'teacher_model.pth'.")

    # Test Teacher and Student Models
    print("Testing Teacher Model...")
    test_model(teacher_model, test_loader, device)

    # Train Student Model (Knowledge Distillation)
    print("Training Student Model...")
    student_model = train_student_model(student_model, teacher_model, train_loader, device,
                                        num_epochs=num_epochs_student)

    # Save the student model
    torch.save(student_model.state_dict(), "StudentModelTestCase/student_model.pth")
    print("Student model saved as 'student_model.pth'.")

    print("Testing Student Model...")
    test_model(student_model, test_loader, device)


if __name__ == "__main__":
    main()
