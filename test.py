import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from models.generator import UNetGenerator
import matplotlib.pyplot as plt

# ==== KULLANICI PARAMETRELERİ ====
image_path = "./img/9.png"
model_path = "./checkpoints/best_gen_epoch014.pth"
output_dir = "./outputs"
device_type = "cuda"  # "cpu" veya "cuda"
sigma_range = (5.0, 50.0)  # Gerçek dünya sigma (0-255 aralığında)
resize_to = 0  # 0: orijinal boyut, >0: yeniden boyutlandır (örn: 512)
patch_size = 256
# ==================================

def process_image_full(model, image_tensor, device):
    _, _, h, w = image_tensor.shape
    pad_h = (256 - h % 256) % 256
    pad_w = (256 - w % 256) % 256
    padding = nn.ZeroPad2d((0, pad_w, 0, pad_h))
    padded = padding(image_tensor)

    with torch.no_grad():
        denoised = model(padded.to(device))

    denoised = denoised[:, :, :h, :w]
    return denoised

def process_image_patches(model, image_tensor, device, patch_size=256):
    _, c, h, w = image_tensor.shape
    denoised = torch.zeros_like(image_tensor)
    count = torch.zeros_like(image_tensor)
    step = patch_size // 2

    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                denoised_patch = model(patch.to(device))
            denoised[:, :, y:y+patch_size, x:x+patch_size] += denoised_patch.cpu()
            count[:, :, y:y+patch_size, x:x+patch_size] += 1

    denoised /= count
    return denoised

def main():
    device = torch.device("cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Model yükle
    generator = UNetGenerator(in_channels=3, out_channels=3, features=64).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint["gen_state_dict"])
    generator.eval()

    # Görüntü oku
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Dönüşümler
    base_transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)) if resize_to > 0 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] → [-1,1]
    ])
    
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # [-1,1] → [0,1]
        transforms.ToPILImage()
    ])

    # Normalize edilmiş temiz görüntü
    clean_tensor = base_transform(image).unsqueeze(0)

    # Gürültü ekle (normalize edilmiş alana)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.0  # normalize sigma
    noise = torch.randn_like(clean_tensor) * sigma
    noisy_tensor = torch.clamp(clean_tensor + noise, -1.0, 1.0)

    # Gürültü giderme işlemi
    _, _, h, w = noisy_tensor.shape
    if h % 256 == 0 and w % 256 == 0:
        print("Tüm görüntü işleniyor (full image mode)...")
        denoised_tensor = process_image_full(generator, noisy_tensor, device)
    else:
        print("Görüntü parça parça işleniyor (patch-based mode)...")
        denoised_tensor = process_image_patches(generator, noisy_tensor, device, patch_size)

    # Görüntüleri geri çevir
    clean_image = reverse_transform(clean_tensor.squeeze(0).cpu())
    noisy_image = reverse_transform(noisy_tensor.squeeze(0).cpu())
    denoised_image = reverse_transform(denoised_tensor.squeeze(0).cpu())

    # Orijinal boyuta döndür
    if resize_to <= 0:
        clean_image = clean_image.resize((orig_w, orig_h), Image.LANCZOS)
        noisy_image = noisy_image.resize((orig_w, orig_h), Image.LANCZOS)
        denoised_image = denoised_image.resize((orig_w, orig_h), Image.LANCZOS)

    # Görselleştirme
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(clean_image)
    plt.title("Temiz Görüntü")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.title(f"Gürültülü (σ={sigma * 255:.1f})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image)
    plt.title("Gürültü Giderilmiş")
    plt.axis("off")

    plt.tight_layout()

    # Kayıt
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path).split('.')[0]
        clean_image.save(os.path.join(output_dir, f"{base}_clean.png"))
        noisy_image.save(os.path.join(output_dir, f"{base}_noisy_sigma{sigma * 255:.1f}.png"))
        denoised_image.save(os.path.join(output_dir, f"{base}_denoised.png"))
        plt.savefig(os.path.join(output_dir, f"{base}_comparison.png"))

    plt.show()

if __name__ == "__main__":
    main()
