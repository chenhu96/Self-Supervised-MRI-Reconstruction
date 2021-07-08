import torch
import torch.nn as nn
import torch.nn.functional as F
from mri_tools import rAtA


class SampleCnn(nn.Module):
    def __init__(self):
        super(SampleCnn, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class BasicBlock(nn.Module):
    def __init__(self, rank):
        super(BasicBlock, self).__init__()
        self.rank = rank
        self.lambda_step = nn.Parameter(torch.FloatTensor([0.5]).cuda(self.rank), requires_grad=True)
        self.soft_thr = nn.Parameter(torch.FloatTensor([0.01]).cuda(self.rank), requires_grad=True)

        self.conv_in = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.f = SampleCnn()
        self.ft = SampleCnn()
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, under_img, mask):
        x = x - self.lambda_step * rAtA(x.permute(0, 2, 3, 1).contiguous(), mask).permute(0, 3, 1, 2).contiguous()
        x = x + self.lambda_step * under_img
        x_input = x

        x_D = self.conv_in(x_input)
        x_forward = self.f(x_D)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x_backward = self.ft(x)
        x_G = self.conv_out(x_backward)
        x_pred = x_input + x_G

        x_D_est = self.ft(x_forward)
        symloss = x_D_est - x_D

        return [x_pred, symloss]
