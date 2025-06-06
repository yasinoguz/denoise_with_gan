import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True):
        super(ConvBlock, self).__init__()
        self.down = down
        self.use_act = use_act
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if down else
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.enc1 = ConvBlock(in_channels, features)
        self.enc2 = ConvBlock(features, features*2)
        self.enc3 = ConvBlock(features*2, features*4)
        self.enc4 = ConvBlock(features*4, features*8)
        self.enc5 = ConvBlock(features*8, features*8)
        self.enc6 = ConvBlock(features*8, features*8)
        self.enc7 = ConvBlock(features*8, features*8)
        
        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(features*8, features*8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        self.dec1 = ConvBlock(features*8 + features*8, features*8, down=False)
        self.dec2 = ConvBlock(features*8 + features*8, features*8, down=False)
        self.dec3 = ConvBlock(features*8 + features*8, features*8, down=False)
        self.dec4 = ConvBlock(features*8 + features*4, features*4, down=False)
        self.dec5 = ConvBlock(features*4 + features*2, features*2, down=False)
        self.dec6 = ConvBlock(features*2 + features, features, down=False)
        
        self.final = nn.Sequential(
            nn.Conv2d(features + in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        b = self.bottleneck(e7)
        
        d1 = self.dec1(torch.cat([b, e6], dim=1))
        d2 = self.dec2(torch.cat([d1, e5], dim=1))
        d3 = self.dec3(torch.cat([d2, e4], dim=1))
        d4 = self.dec4(torch.cat([d3, e3], dim=1))
        d5 = self.dec5(torch.cat([d4, e2], dim=1))
        d6 = self.dec6(torch.cat([d5, e1], dim=1))
        
        out = self.final(torch.cat([d6, x], dim=1))
        return out
# if __name__ == "__main__":
#     # Test
#     model = UNetGenerator(in_channels=3, out_channels=3)
#     x = torch.randn((1, 3, 128, 128))
#     with torch.no_grad():
#         out = model(x)
#     print("Sonuç çıktısı: ", out.shape)
