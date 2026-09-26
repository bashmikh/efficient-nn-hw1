from torch import nn


class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(32, 64, stride=1, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, stride=1, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, stride=2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, stride=1, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    x = torch.randn(1, 3, 224, 224)
    for layer in ConvNN().eval().model:
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(layer.__class__.__name__, tuple(x.shape))