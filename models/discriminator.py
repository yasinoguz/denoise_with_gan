import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(PatchDiscriminator, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature, 4, 2, 1),
                        nn.LeakyReLU(0.2)
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(features[idx-1], feature, 4, 2, 1),
                        nn.BatchNorm2d(feature),
                        nn.LeakyReLU(0.2)
                    )
                )
        layers.append(nn.Conv2d(features[-1], 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)