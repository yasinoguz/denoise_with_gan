import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class ContentLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super(ContentLoss, self).__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Geçersiz loss_type: {loss_type}. 'l1' ya da 'mse' olmalı.")
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class AdversarialLoss(nn.Module):
    def __init__(self, use_logits=True, gan_type='bce'):
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        if gan_type == 'bce':
            if use_logits:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCELoss()
        else:  # wasserstein
            self.criterion = None
    
    def forward(self, disc_pred, is_real):
        if self.gan_type == 'bce':
            target_value = 1.0 if is_real else 0.0
            target_tensor = torch.full_like(disc_pred, fill_value=target_value)
            return self.criterion(disc_pred, target_tensor)
        else:  # wasserstein
            return -disc_pred.mean() if is_real else disc_pred.mean()

class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, weights=None, device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.layer_name_mapping = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15,
            'relu4_3': 22, 'relu5_3': 29
        }
        
        layers = layers or ['relu2_2', 'relu3_3', 'relu4_3']
        weights = weights or [1.0 / len(layers)] * len(layers)
        assert len(layers) == len(weights)
        
        self.layers = layers
        self.weights = weights
        self.layer_indices = [self.layer_name_mapping[l] for l in layers]
        self.vgg = nn.Sequential(*list(vgg.children())[:max(self.layer_indices)+1])
        self.device = device
        self.criterion = nn.MSELoss()
        
    def forward(self, pred, target):
        def preprocess_vgg(x):
            x = (x + 1.0) / 2.0
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            return normalize(x)
        
        pred_vgg = preprocess_vgg(pred)
        target_vgg = preprocess_vgg(target)
        
        # Tek geçişte tüm özellikleri çıkar
        features_pred = {}
        features_target = {}
        x_p, x_t = pred_vgg, target_vgg
        
        for i, layer in enumerate(self.vgg):
            x_p = layer(x_p)
            x_t = layer(x_t)
            if i in self.layer_indices:
                features_pred[i] = x_p
                features_target[i] = x_t
        
        # Katman kayıplarını hesapla
        loss = 0.0
        for idx, w in zip(self.layer_indices, self.weights):
            loss += w * self.criterion(
                features_pred[idx], 
                features_target[idx]
            )
            
        return loss

class GANLoss(nn.Module):
    def __init__(self, lambda_content=1.0, lambda_adv=1e-3, 
                 lambda_percep=1e-2, device='cuda', gan_type='bce'):
        super(GANLoss, self).__init__()
        self.lambda_content = lambda_content
        self.lambda_adv = lambda_adv
        self.lambda_percep = lambda_percep
        self.gan_type = gan_type

        self.content_loss = ContentLoss(loss_type='l1').to(device)
        self.adversarial_loss = AdversarialLoss(
            use_logits=True, gan_type=gan_type).to(device)
        self.perceptual_loss = PerceptualLoss(device=device).to(device)

    def forward(self, gen_out, real_img, disc_pred_fake, disc_pred_real):
        c_loss = self.content_loss(gen_out, real_img)
        p_loss = self.perceptual_loss(gen_out, real_img)
        
        if self.gan_type == 'bce':
            adv_loss_gen = self.adversarial_loss(disc_pred_fake, is_real=True)
            d_loss_real = self.adversarial_loss(disc_pred_real, is_real=True)
            d_loss_fake = self.adversarial_loss(disc_pred_fake.detach(), is_real=False)
            total_disc_loss = (d_loss_real + d_loss_fake) * 0.5
        else:  # wasserstein
            adv_loss_gen = self.adversarial_loss(disc_pred_fake, is_real=True)
            d_loss_real = self.adversarial_loss(disc_pred_real, is_real=True)
            d_loss_fake = self.adversarial_loss(disc_pred_fake.detach(), is_real=False)
            total_disc_loss = d_loss_real + d_loss_fake

        total_gen_loss = (
            self.lambda_content * c_loss +
            self.lambda_adv * adv_loss_gen +
            self.lambda_percep * p_loss
        )

        return total_gen_loss, total_disc_loss, {
            'content_loss': c_loss.item(),
            'adv_loss_gen': adv_loss_gen.item(),
            'perceptual_loss': p_loss.item(),
            'disc_loss_real': d_loss_real.item(),
            'disc_loss_fake': d_loss_fake.item()
        }