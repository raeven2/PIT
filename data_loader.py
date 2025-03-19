# -*- coding: utf-8 -*-

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlaqueDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): directory where images are stored (e.g. "1.png")
            mask_dir (str): directory where masks are stored (e.g. "1_label.png")
            transform: optional transforms to apply
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect a list of images that do NOT contain "_label"
        # e.g. "1.png", "2.png", ...
        self.image_files = []
        for fname in os.listdir(image_dir):
            # Make sure it's actually an image and not already a mask
            if fname.endswith(".png") and "_label" not in fname:
                self.image_files.append(fname)

        # Sort for consistent ordering (optional)
        self.image_files = sorted(self.image_files)

        # Optionally verify that each corresponding mask exists
        # e.g. "1.png" => "1_label.png"
        # If you want to ensure all pairs exist
        for img_name in self.image_files:
            base = img_name[:-4]            # remove ".png", e.g. "1"
            mask_name = base + "_label.png" # e.g. "1_label.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1) Get the image name
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 2) Derive the corresponding mask name
        #    e.g. "1.png" => "1_label.png"
        base = img_name[:-4]               # remove ".png"
        mask_name = base + "_label.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 3) Load images in grayscale
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # 4) Apply transforms (resize, toTensor, etc.)
        if self.transform is None:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            transform = self.transform

        image = transform(image)  # shape: [1, 256, 256]
        mask = transform(mask)    # shape: [1, 256, 256]

        return image, mask

# DataLoader function
def get_dataloader(image_dir, mask_dir, batch_size=32, transform=None, shuffle=True):
    dataset = PlaqueDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
