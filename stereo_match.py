from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
# from torch_scatter import scatter_max
print(torch.__version__)

class Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1, activation=True):
        super(Conv1d, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        if activation:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            )
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)

        return x

# TODO: we should replace this with CenterNet feature network
class Feature2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Feature2d, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            Conv2d(out_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            nn.MaxPool2d(2),
            Conv2d(out_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            Conv2d(out_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class SamplePoint(nn.Module):
    def __init__(self, in_ch, width, height):
        super(SamplePoint, self).__init__()
        self.channel = in_ch
        self.width = width
        self.height = height
    
    def forward(self, x, image_num, image_ids, cols, rows):
        #grid_sample requires coords (-1, 1)
        cols_norm = 2 * cols / self.width - 1
        rows_norm = 2 * rows / self.height / image_num - 1
        grid = torch.cat((torch.unsqueeze(cols_norm, -1), torch.unsqueeze(rows_norm, -1)), 1)
        grid = torch.unsqueeze(torch.unsqueeze(grid, 0), 0)

        x = x.permute(1, 0, 2, 3).reshape(1, self.channel, self.height*image_num, self.width)
        x = torch.nn.functional.grid_sample(x, grid).reshape(self.channel, 1, -1).repeat(1, self.width, 1)  # C-W-N

        return x.permute(2, 0, 1)   # return N-C-W

class SamplePointSimple(nn.Module):
    def __init__(self):
        super(SamplePointSimple, self).__init__()

    def forward(self, input, image_ids, centers):
        B, C, H, W = input.shape

        rows = image_ids * H + centers[:, 0]
        cols = centers[:, 1]

        reshape_input = input.permute(1, 0, 2, 3).reshape(1, C, B * H, W)

        output = reshape_input[:, :, rows, cols].permute(2, 1, 0).repeat(1, 1, W)
        return output

class SampleRow(nn.Module):
    def __init__(self, in_ch, width, height):
        super(SampleRow, self).__init__()
        self.channel = in_ch
        self.width = width
        self.height = height

    def forward(self, x, image_num, image_ids, rows):
        rows_norm = 2 * rows / self.height / image_num - 1
        cols_norm = 0 * rows_norm - 1
        grid = torch.cat((torch.unsqueeze(cols_norm, -1), torch.unsqueeze(rows_norm, -1)), 1)
        grid = torch.unsqueeze(torch.unsqueeze(grid, 0), 0)
        x = x.permute(1, 3, 0, 2)       # B-C-H-W -> C-W-B-H
        x = x.reshape(1, self.channel*self.width, image_num*self.height, 1) # C-W-B-H -> 1-C*W-B*H-1
        x = torch.nn.functional.grid_sample(x, grid)    # get 1-C*W-1-N, N is number of match src points
        x = x.reshape(self.channel, self.width, -1)     # C-W-N

        return x.permute(2, 0, 1)   # return N-C-W

class Match1d(nn.Module):
    def __init__(self, in_ch):
        super(Match1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1d(in_ch, in_ch, kernel_size=3, dilation=1, padding=1),
            Conv1d(in_ch, in_ch, kernel_size=3, dilation=2, padding=1),
            Conv1d(in_ch, in_ch, kernel_size=3, dilation=2, padding=1),
            Conv1d(in_ch, in_ch, kernel_size=3, dilation=2, padding=1),
            Conv1d(in_ch, 1, kernel_size=1, dilation=1, padding=0, activation=False)
        )
    
    def forward(self, x):
        return self.conv(x)

class DispRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DispRegression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        tmp = F.softmax(torch.squeeze(x, 1), dim=-1)

        disp = self.disp.repeat(x.size()[0], 1)
        pred = torch.sum(tmp * disp, 1)
        return pred

class StereoMatch(nn.Module):
    def __init__(self, width, height, in_ch=3, feature_ch=32, feature_downsample=4, use_xyz=True):
        super(StereoMatch, self).__init__()
        self.feature = Feature2d(in_ch, feature_ch)
        self.sample_point = SamplePoint(feature_ch, width / feature_downsample, height / feature_downsample)
        self.sample_row = SampleRow(feature_ch, width / feature_downsample, height / feature_downsample)
        self.match1d = Match1d(feature_ch*2)

    def forward(self, images_L, images_R, image_num, image_ids, rows, cols):
        x_L = self.feature(images_L)
        x_R = self.feature(images_R)
        point_feature = self.sample_point(x_L, image_num, image_ids, cols, rows)
        row_feature = self.sample_row(x_R, image_num, image_ids, rows)
        x = torch.cat((point_feature, row_feature), 1)
        
        return self.match1d(x)

if __name__ == "__main__":
    # test SamplePoint and SampleRow
    sample_point = SamplePoint(32, 100, 100)
    sample_row = SampleRow(32, 100, 100)
    feature = torch.rand(3, 32, 100, 100)
    image_num = 3
    cols = torch.tensor([30, 13, 32, 20, 5, 70, 18, 44, 77]).float()
    rows = torch.tensor([0, 1, 30, 20, 50, 20, 15, 80, 99]).float()
    image_ids = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 2])
    point_feature = sample_point.forward(feature, image_num, image_ids, cols, rows)
    row_feature = sample_row.forward(feature, image_num, image_ids, rows)

    cost_volume = torch.cat((point_feature, row_feature), 1)
    match1d = Match1d(64)
    output = match1d(cost_volume)

    disp_reg = DispRegression(100)
    disp = disp_reg(output.cuda())

    groud_truth = torch.tensor([48, 49, 50, 45, 52, 51, 50, 51, 48]).float().cuda()
    loss = F.smooth_l1_loss(disp, groud_truth)
    print(loss)

    # sample_point = SamplePointSimple()
    # cols = torch.tensor([30, 13, 32, 20, 5, 70, 18, 44, 77])
    # rows = torch.tensor([0, 1, 30, 20, 50, 20, 15, 80, 99])
    # centers = torch.cat((torch.unsqueeze(rows, -1), torch.unsqueeze(cols, -1)), -1)
    # point_feature = sample_point(feature, image_ids, centers)