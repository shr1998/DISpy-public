import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
        self.up = torch.nn.ConvTranspose2d(channel, channel, 2, 2)

    def forward(self, x, feature_map):
        out = self.layer(self.up(x))
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 6, 3, 1, 1)
        print("num_classes: ", 6)

    def forward(self, x):
        R1 = self.c1(x)
        # print(R1.size())
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        # print(R5.size())
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)
        # return F.log_softmax(self.out(O4),dim=1)


def predict(path, num_classes, device, weight_path, transform):
    net = UNet(num_classes).to(device)
    net.load_state_dict(torch.load(weight_path))
    a = Image.open(path)
    img = transform(a)
    img = img.to(device)
    img = img.unsqueeze(0)
    b = net(img)
    out = b.cpu().detach().numpy()
    return out


if __name__ == '__main__':
    from torchsummary import summary

    x = torch.randn(1, 3, 480, 640)
    net = UNet(6)
    summary(net, (3, 480, 640), device="cpu")
    print("shape: ", net(x).shape)
