import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def add_gaussian_noise(image, sigma):
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0.0, 1.0)

def add_salt_pepper_noise(image, amount=0.02, s_vs_p=0.6):
    noisy = image.clone()
    c, h, w = image.shape
    num_salt = int(amount * h * w * s_vs_p)
    num_pepper = int(amount * h * w * (1.0 - s_vs_p))

    for ch in range(c):
        coords = [torch.randint(0, i, (num_salt,)) for i in (h, w)]
        noisy[ch, coords[0], coords[1]] = 1.0

    for ch in range(c):
        coords = [torch.randint(0, i, (num_pepper,)) for i in (h, w)]
        noisy[ch, coords[0], coords[1]] = 0.0

    return noisy

def add_speckle_noise(image, sigma):
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + image * noise, 0.0, 1.1)

class DenoisingDataset(Dataset):
    def __init__(self, image_paths, patch_size=256, sigma_range=(1, 10),
                 max_patches_per_image=10, mode='train'):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.sigma_range = sigma_range
        self.max_patches_per_image = max_patches_per_image
        self.mode = mode

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10)
            ])
        else:
            self.augment = None

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.image_paths) * self.max_patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.max_patches_per_image
        image_path = self.image_paths[img_idx]

        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except:
            # Bozuk bir görüntü geldiyse, bir sonrakine geç
            return self.__getitem__((idx + 1) % self.__len__())

        h, w, _ = image.shape
        if h < self.patch_size or w < self.patch_size:
            return self.__getitem__((idx + 1) % self.__len__())

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        patch_np = image[top:top + self.patch_size, left:left + self.patch_size, :]
        patch = Image.fromarray(patch_np)

        if self.augment:
            patch = self.augment(patch)

        patch_tensor = self.to_tensor(patch)
        clean_tensor = patch_tensor.clone()

        sigma = random.uniform(*self.sigma_range) / 255.0
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])

        if noise_type == 'gaussian':
            noisy_tensor = add_gaussian_noise(patch_tensor, sigma)
        elif noise_type == 'salt_pepper':
            noisy_tensor = add_salt_pepper_noise(patch_tensor, amount=0.01)
        elif noise_type == 'speckle':
            noisy_tensor = add_speckle_noise(patch_tensor, sigma)
        else:
            noisy_tensor = patch_tensor

        clean_norm = self.normalize(clean_tensor)
        noisy_norm = self.normalize(noisy_tensor)

        # Sadece two-valued döndür: (noisy, clean)
        return noisy_norm, clean_norm

def get_image_paths(image_dir):
    return [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

def create_dataloaders(image_dir, batch_size=16, patch_size=128, sigma_range=(1, 10),
                       max_patches_per_image=10, val_split=0.2, num_workers=4,
                       pin_memory=True):
    image_paths = get_image_paths(image_dir)
    train_paths, val_paths = train_test_split(image_paths, test_size=val_split, random_state=42)

    train_dataset = DenoisingDataset(
        train_paths, patch_size, sigma_range, max_patches_per_image, mode='train')
    val_dataset = DenoisingDataset(
        val_paths, patch_size, sigma_range, max_patches_per_image, mode='val')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=2
    )

    return train_loader, val_loader

def show_images(noisy, clean):
    noisy = noisy.permute(1, 2, 0).cpu().numpy()
    clean = clean.permute(1, 2, 0).cpu().numpy()

    noisy = (noisy * 0.5 + 0.5).clip(0, 1)
    clean = (clean * 0.5 + 0.5).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(noisy)
    axes[0].set_title("Noisy")
    axes[0].axis("off")

    axes[1].imshow(clean)
    axes[1].set_title("Clean")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    image_dir = "C:/Users/yasin/Desktop/restoration/esrgan/unet_sr/Flickr2K/HR/train"

    train_loader, val_loader = create_dataloaders(
        image_dir=image_dir,
        batch_size=1,
        patch_size=256,
        sigma_range=(10, 40),
        max_patches_per_image=5,
        val_split=0.1,
        num_workers=4,   # Windows’ta num_workers=0 daha güvenli olabilir
        pin_memory=False
    )

    data_iter = iter(train_loader)

    try:
        # Artık 3 değil, 2 eleman unpack ediliyor:
        noisy1, clean1 = next(data_iter)
        show_images(noisy1[0], clean1[0])
    except StopIteration:
        pass

    try:
        noisy2, clean2 = next(data_iter)
        show_images(noisy2[0], clean2[0])
    except StopIteration:
        pass

if __name__ == "__main__":
    main()
