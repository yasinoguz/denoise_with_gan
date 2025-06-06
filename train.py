import os
import argparse
from datetime import datetime
import warnings
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset2 import create_dataloaders
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from losses import GANLoss

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def compute_ssim_batch(pred_batch, target_batch):
    pred_np = pred_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

    ssim_vals = []
    for i in range(pred_np.shape[0]):
        h, w = pred_np[i].shape[:2]
        win_size = min(7, h, w)
        win_size = win_size if win_size % 2 == 1 else win_size - 1
        win_size = max(3, win_size)

        try:
            s = ssim(
                pred_np[i], target_np[i],
                channel_axis=2, data_range=1.0, win_size=win_size
            )
            ssim_vals.append(s)
        except ValueError as e:
            warnings.warn(f"SSIM hesaplanamadı: {e}")
            ssim_vals.append(0.0)
            
    return np.mean(ssim_vals)

def calculate_gradient_penalty(disc, real, fake, device):
    batch_size, C, H, W = real.shape
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    disc_interpolates = disc(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def get_args():
    parser = argparse.ArgumentParser(description="GAN-based Image Denoising Training Script")
    
    # Data ve checkpoint dizinleri
    parser.add_argument("--data_dir", type=str, default="C:/Users/yasin/Desktop/restoration/esrgan/unet_sr/Flickr2K/HR/train")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    
    # Eğitim parametreleri
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    
    # Loss ağırlıkları
    parser.add_argument("--lambda_content", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1e-3)
    parser.add_argument("--lambda_percep", type=float, default=1e-2)
    
    # Diğer parametreler
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--sigma_min", type=float, default=10.0)
    parser.add_argument("--sigma_max", type=float, default=40.0)
    parser.add_argument("--max_patches_per_image", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.2)
    
    # GPU ve DataLoader
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    # GAN türü ve gradient penalty
    parser.add_argument("--gan_type", type=str, default="bce", choices=["bce", "wasserstein"])
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    
    # LR Scheduler
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    # Resume checkpoint
    parser.add_argument(
        "--resume", type=str, default="./checkpoints/best_gen_epoch014.pth",
        help="Eğitime devam etmek için checkpoint dosyası (ör. ./checkpoints/best_gen_epoch014.pth)"
    )
    
    return parser.parse_args()

def train(args):
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.save_dir, exist_ok=True)
    sample_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # DataLoader'ları oluştur
    train_loader, val_loader = create_dataloaders(
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        sigma_range=(args.sigma_min, args.sigma_max),
        max_patches_per_image=args.max_patches_per_image,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    # Model, Opt., Scheduler, Loss hazırla
    gen = UNetGenerator(in_channels=3, out_channels=3, features=64).to(device)
    disc = PatchDiscriminator(in_channels=3, features=[64, 128, 256, 512]).to(device)
    
    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_gen, mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )
    
    criterion = GANLoss(
        lambda_content=args.lambda_content,
        lambda_adv=args.lambda_adv,
        lambda_percep=args.lambda_percep,
        device=device,
        gan_type=args.gan_type
    )
    
    scaler = GradScaler()
    
    # Resume kontrolü
    start_epoch = 1
    best_val_psnr = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[INFO] Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Model ağırlıkları
            gen.load_state_dict(checkpoint["gen_state_dict"])
            disc.load_state_dict(checkpoint["disc_state_dict"])
            
            # Optimizer'ların state dict'leri
            opt_gen.load_state_dict(checkpoint["opt_gen_state"])
            opt_disc.load_state_dict(checkpoint["opt_disc_state"])
            
            # Başlangıç epoch'u ayarla
            start_epoch = checkpoint["epoch"] + 1
            
            # En iyi PSNR'ı al
            best_val_psnr = checkpoint.get("best_val_psnr", 0.0)
            
            print(f"[INFO] Resuming from epoch {checkpoint['epoch']} (next: {start_epoch})")
        else:
            print(f"[WARNING] Resume checkpoint bulunamadı: {args.resume}")
            print("[WARNING] Yeni eğitim başlatılıyor.")
    
    # Eğitim döngüsü
    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        disc.train()
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        
        for step, (noisy_patches, clean_patches) in enumerate(
            tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}", leave=False), start=1
        ):
            noisy_patches = noisy_patches.to(device)
            clean_patches = clean_patches.to(device)
            
            # Discriminator Güncelleme
            with autocast():
                fake_patches = gen(noisy_patches)
                disc_real = disc(clean_patches)
                disc_fake = disc(fake_patches.detach())
                
                if args.gan_type == 'bce':
                    _, d_loss, _ = criterion(
                        gen_out=fake_patches,
                        real_img=clean_patches,
                        disc_pred_fake=disc_fake,
                        disc_pred_real=disc_real
                    )
                else:  # wasserstein
                    d_loss_real = criterion.adversarial_loss(disc_real, True)
                    d_loss_fake = criterion.adversarial_loss(disc_fake, False)
                    d_loss = d_loss_real + d_loss_fake
                    
                    # Gradient penalty
                    gp = calculate_gradient_penalty(
                        disc, clean_patches, fake_patches.detach(), device
                    )
                    d_loss += args.lambda_gp * gp
                
            opt_disc.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(opt_disc)
            
            # Generator Güncelleme
            with autocast():
                disc_fake_for_g = disc(fake_patches)
                
                if args.gan_type == 'bce':
                    g_loss, _, _ = criterion(
                        gen_out=fake_patches,
                        real_img=clean_patches,
                        disc_pred_fake=disc_fake_for_g,
                        disc_pred_real=disc_real
                    )
                else:  # wasserstein
                    c_loss = criterion.content_loss(fake_patches, clean_patches)
                    p_loss = criterion.perceptual_loss(fake_patches, clean_patches)
                    g_adv_loss = criterion.adversarial_loss(disc_fake_for_g, True)
                    g_loss = (
                        args.lambda_content * c_loss +
                        args.lambda_percep * p_loss +
                        args.lambda_adv * g_adv_loss
                    )
                
            opt_gen.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(opt_gen)
            
            scaler.update()
            epoch_gen_loss += g_loss.item()
            epoch_disc_loss += d_loss.item()
            
            # Loglama (her 100 step)
            if step % 100 == 0:
                with torch.no_grad():
                    fake_01 = (fake_patches.detach() + 1) / 2.0
                    clean_01 = (clean_patches + 1) / 2.0
                    batch_psnr = compute_psnr(fake_01, clean_01).item()
                    batch_ssim = compute_ssim_batch(fake_01, clean_01)
                
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                    f"[Step {step:04d}/{len(train_loader):04d}] "
                    f"G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}, "
                    f"PSNR: {batch_psnr:.2f}, SSIM: {batch_ssim:.4f}"
                )
        
        # Epoch sonu istatistikler
        avg_gen_loss = epoch_gen_loss / len(train_loader)
        avg_disc_loss = epoch_disc_loss / len(train_loader)
        print(f"\n======== Epoch {epoch:03d} ========")
        print(f"Gen Loss Avg:  {avg_gen_loss:.4f}")
        print(f"Disc Loss Avg: {avg_disc_loss:.4f}")
        
        # Validation
        gen.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for noisy_patches, clean_patches in tqdm(val_loader, desc="Validation", leave=False):
                noisy_patches = noisy_patches.to(device)
                clean_patches = clean_patches.to(device)
                
                fake_patches = gen(noisy_patches)
                fake_01 = (fake_patches + 1) / 2.0
                clean_01 = (clean_patches + 1) / 2.0
                
                val_psnr += compute_psnr(fake_01, clean_01).item()
                val_ssim += compute_ssim_batch(fake_01, clean_01)
            
            val_psnr /= len(val_loader)
            val_ssim /= len(val_loader)
        
        print(f"Validation PSNR: {val_psnr:.2f} dB")
        print(f"Validation SSIM: {val_ssim:.4f}")
        
        # LR Scheduler update
        scheduler_gen.step(val_psnr)
        print(f"Current LR: {opt_gen.param_groups[0]['lr']:.2e}")
        
        # Checkpoint kaydetme (en iyi model)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            ckpt_path = os.path.join(args.save_dir, f"best_gen_epoch{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "gen_state_dict": gen.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "opt_gen_state": opt_gen.state_dict(),
                "opt_disc_state": opt_disc.state_dict(),
                "best_val_psnr": best_val_psnr,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim
            }, ckpt_path)
            print(f"[INFO] Saved Best Checkpoint: {ckpt_path}")
        
        # Örnek görseller (her 2 epoch’ta bir)
        if epoch % 2 == 0:
            sample_noisy, sample_clean = next(iter(val_loader))
            sample_noisy = sample_noisy[:2].to(device)
            sample_clean = sample_clean[:2].to(device)
            
            with torch.no_grad():
                sample_fake = gen(sample_noisy)
            
            sample_noisy_01 = (sample_noisy + 1) / 2.0
            sample_clean_01 = (sample_clean + 1) / 2.0
            sample_fake_01 = (sample_fake + 1) / 2.0
            
            grid = torch.cat([sample_noisy_01, sample_fake_01, sample_clean_01], dim=0)
            grid = make_grid(grid, nrow=2, normalize=False)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_image(grid, os.path.join(sample_dir, f"epoch{epoch:03d}_{timestamp}.png"))
        
        print()

if __name__ == "__main__":
    args = get_args()
    train(args)
