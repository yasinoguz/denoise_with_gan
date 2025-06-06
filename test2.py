import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from models.generator import UNetGenerator

# --- Ayarları burada manuel belirle ---
INPUT_IMAGE_PATH     = "./img/9.png"
OUTPUT_NOISY_PATH    = "./img/10_noisy.png"
OUTPUT_DENOISED_PATH = "./img/11_denoised.png"
CHECKPOINT_PATH      = "./checkpoints/best_gen_epoch020.pth"
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE           = 256
OVERLAP              = 64         # PATCH_SIZE=256 için 64 piksel overlap
NOISE_STD_RANGE      = (10/255, 40/255)  # Eğitimle aynı aralıkta (10-40)/255

# ---- Gürültü ekleme fonksiyonları (PIL Image girdi → PIL Image çıktı) ----

def add_gaussian_noise_pil(image: Image.Image, std: float) -> Image.Image:
    """
    RGB PIL görüntüsüne Gaussian gürültü ekler.
    std: 0.0 - 1.0 arası bir değer.
    """
    image_np = np.array(image).astype(np.float32) / 255.0         # [H,W,3], float32
    noise    = np.random.normal(0, std, image_np.shape).astype(np.float32)
    noisy_np = np.clip(image_np + noise, 0.0, 1.0)
    noisy_img = Image.fromarray((noisy_np * 255).astype(np.uint8))
    return noisy_img

def add_salt_pepper_noise_pil(image: Image.Image, amount=0.01, s_vs_p=0.5) -> Image.Image:
    """
    RGB PIL görüntüsüne Salt & Pepper gürültü ekler.
    amount: toplam piksellerin % kaçına gürültü eklensin (örneğin 0.02).
    s_vs_p: gürültü içinde tuz (white) ve karabiber (black) oranı.
    """
    tensor = transforms.ToTensor()(image)          # [3, H, W], [0,1]
    c, h, w = tensor.shape
    noisy = tensor.clone()
    num_salt   = int(amount * h * w * s_vs_p)
    num_pepper = int(amount * h * w * (1.0 - s_vs_p))

    # “Salt” pikselleri
    for ch in range(c):
        coords = [torch.randint(0, i, (num_salt,)) for i in (h, w)]
        noisy[ch, coords[0], coords[1]] = 1.0
    # “Pepper” pikselleri
    for ch in range(c):
        coords = [torch.randint(0, i, (num_pepper,)) for i in (h, w)]
        noisy[ch, coords[0], coords[1]] = 0.0

    noisy_np = noisy.permute(1, 2, 0).cpu().numpy()  # [H, W, 3], [0,1]
    noisy_img = Image.fromarray((noisy_np * 255).astype(np.uint8))
    return noisy_img

def add_speckle_noise_pil(image: Image.Image, std: float) -> Image.Image:
    """
    RGB PIL görüntüsüne Speckle gürültü ekler:
    out = image + image * gaussian_noise
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, image_np.shape).astype(np.float32)
    noisy_np = np.clip(image_np + image_np * noise, 0.0, 1.0)
    noisy_img = Image.fromarray((noisy_np * 255).astype(np.uint8))
    return noisy_img

# ---- Model yükleme / ön işleme / sonrası işleme fonksiyonları ----

def load_model(checkpoint_path, device):
    """
    Model checkpoint'ını yükler ve GPU/CPU’ya taşır.
    """
    gen = UNetGenerator(in_channels=3, out_channels=3, features=64)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen = gen.to(device)
    gen.eval()
    return gen

def preprocess_tensor(image: Image.Image, device: str):
    """
    PIL Image -> normalize edilmiş torch.Tensor [1,3,H,W]
    ([-1,1] aralığında)
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)   # Eğitimde kullanılan normalize ile uyumlu
    ])
    t = tf(image).unsqueeze(0).to(device)  # (1, 3, H, W)
    return t

def postprocess_tensor(tensor: torch.Tensor):
    """
    Normalize edilmiş tensör [1,3,H,W] -> PIL Image
    ([-1,1] → [0,1] aralığına çevirip UInt8)
    """
    t = tensor.squeeze(0).detach().cpu()    # [3, H, W]
    t = t * 0.5 + 0.5                       # [0,1]
    t = torch.clamp(t, 0, 1)
    return transforms.ToPILImage()(t)

def denoise_with_unfold_fold(model, image: Image.Image,
                             patch_size=256, overlap=64, device="cuda") -> Image.Image:
    """
    1) Görüntüyü dinamik pad’le,
    2) Unfold ile patch’leri çıkar,
    3) Model inference (tüm patch’ler batch olarak verilir),
    4) Patch’leri uniform ağırlık matrisiyle ağırlıklandır,
    5) Fold ile birleştir,
    6) Aynı weight matrisiyle fold et,
    7) Çıktıyı weight’e böl → seam’sız blend,
    8) PIL’e dön.
    """
    img_tensor = preprocess_tensor(image, device)  # (1,3,H,W)
    _, c, H, W = img_tensor.shape

    stride = patch_size - overlap

    # Dinamik padding hesapla
    pad_total_h = (stride - (H % stride)) % stride    # Dikey eksik padding
    pad_total_w = (stride - (W % stride)) % stride    # Yatay eksik padding

    # Pad: (left, right, top, bottom)
    padded = F.pad(img_tensor,
                   (overlap, overlap + pad_total_w,
                    overlap, overlap + pad_total_h),
                   mode="reflect")
    _, _, H_pad, W_pad = padded.shape

    # Unfold: (1, c*P*P, num_patches)
    patches = F.unfold(padded, kernel_size=patch_size, stride=stride)
    num_patches = patches.size(2)

    # → [num_patches, c, P, P]
    patches = patches.permute(0, 2, 1).contiguous()
    patches = patches.view(-1, c, patch_size, patch_size)  # (num_patches, c, P, P)

    # Model inference
    with torch.no_grad():
        denoised_patches = model(patches)          # (num_patches, c, P, P)

    # Uniform ağırlık matrisi (Hann yerine)
    hann2d = torch.ones((c, patch_size, patch_size), device=device)  # (c, P, P)

    # Ağırlıklandır
    denoised_patches = denoised_patches * hann2d.unsqueeze(0)       # (num_patches, c, P, P)

    # → [1, c*P*P, num_patches]
    denoised_flat = denoised_patches.view(num_patches, c * patch_size * patch_size)
    denoised_flat = denoised_flat.transpose(0, 1).unsqueeze(0)       # (1, c*P*P, num_patches)

    # Fold ile birleştir: (1, c, H_pad, W_pad)
    denoised_fold = F.fold(
        denoised_flat,
        output_size=(H_pad, W_pad),
        kernel_size=patch_size,
        stride=stride
    )

    # Weight oluşturmak için “1” patch’lerini aynı matrise fold et
    hann_flat = hann2d.view(-1)                                      # (c*P*P,)
    weight_patches = hann_flat.unsqueeze(0).unsqueeze(2).repeat(1, 1, num_patches)
    weight_fold = F.fold(
        weight_patches,
        output_size=(H_pad, W_pad),
        kernel_size=patch_size,
        stride=stride
    )

    # Padding’i çıkar (crop)
    denoised_final = denoised_fold[:, :, overlap:overlap+H, overlap:overlap+W]
    weight_final   = weight_fold[:,   :, overlap:overlap+H, overlap:overlap+W]

    # Seam’sız blend
    out_tensor = denoised_final / (weight_final + 1e-8)

    return postprocess_tensor(out_tensor)

def show_and_wait(image: Image.Image, title: str):
    """
    Matplotlib ile verilen görüntüyü başlıkla açıp bekler.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()  # Kullanıcı pencereyi kapatana kadar bekler

# ================================

def main():
    # 1) Modeli yükle
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # 2) Orijinal görüntüyü aç (RGB)
    original = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    # 3) Gürültü tipini rastgele seç
    noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
    if noise_type == 'gaussian':
        std = random.uniform(*NOISE_STD_RANGE)
        noisy = add_gaussian_noise_pil(original, std=std)
        info_text = f"Gaussian (std={std:.3f})"
    elif noise_type == 'salt_pepper':
        # amount ve s_vs_p değerlerini burada sabit ya da rastgele verebilirsiniz
        noisy = add_salt_pepper_noise_pil(original, amount=0.02, s_vs_p=0.6)
        info_text = "Salt & Pepper (amount=0.02, s_vs_p=0.6)"
    else:  # 'speckle'
        std = random.uniform(*NOISE_STD_RANGE)
        noisy = add_speckle_noise_pil(original, std=std)
        info_text = f"Speckle (std={std:.3f})"

    # 4) Noisy ve Original’ı göster
    show_and_wait(original, "Original Image")
    show_and_wait(noisy, f"Noisy Image → {info_text}")

    # 5) Noisy görüntüyü kaydetmek isterseniz
    os.makedirs(os.path.dirname(OUTPUT_NOISY_PATH), exist_ok=True)
    noisy.save(OUTPUT_NOISY_PATH)
    print(f"Noisy image kaydedildi: {OUTPUT_NOISY_PATH}")

    # 6) Denoise: eğer küçükse direk model, büyükse unfold/fold
    w, h = noisy.size
    if max(w, h) <= PATCH_SIZE:
        # Tek patch olarak
        noisy_tensor = preprocess_tensor(noisy, DEVICE)
        with torch.no_grad():
            out_tensor = model(noisy_tensor)
        denoised = postprocess_tensor(out_tensor)
    else:
        # Unfold/Fold ile seam’sız denoise
        denoised = denoise_with_unfold_fold(
            model,
            noisy,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            device=DEVICE
        )

    # 7) Denoised görüntüyü göster ve kaydet
    show_and_wait(denoised, "Denoised Image")
    os.makedirs(os.path.dirname(OUTPUT_DENOISED_PATH), exist_ok=True)
    denoised.save(OUTPUT_DENOISED_PATH)
    print(f"Denoised image kaydedildi: {OUTPUT_DENOISED_PATH}")

if __name__ == "__main__":
    main()
