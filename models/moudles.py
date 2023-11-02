import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=256, out_dim=100):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduce = nn.Sequential(
            nn.Linear(in_features=self.in_dim*self.in_dim, out_features=self.out_dim, bias=True),
            nn.Dropout(p=0.2),
	    nn.ReLU(inplace=True),
            # nn.Linear(in_features=16384, out_features=4096, bias=True),
            # nn.Dropout(p=0.1),
            nn.Linear(in_features=self.out_dim, out_features=self.out_dim, bias=True)
            )

    def forward(self, x):

        out = self.reduce(x)
        return out

class Decoder(nn.Module):
    def __init__(self, in_dim=100):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.res1 = nn.Sequential(
            nn.Conv2d(self.in_dim + 2, self.in_dim, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(100),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(self.in_dim, 100, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(100, 1, kernel_size=1, padding=0, bias=False))

    def forward(self, x):

        x = self.res1(x)
        x = self.res2(x) + x
        out = self.down(x)

        return out

