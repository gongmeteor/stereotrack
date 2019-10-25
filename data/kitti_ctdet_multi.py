from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from kitti_ctdet import KittiCtdet
import numpy as np

class KittiCtdetMulti(data.Dataset):
    def __init__(self, opt, split, num):
        super(KittiCtdetMulti, self).__init__()
        self.loaders = []
        for i in range(num):
            self.loaders.append(KittiCtdet(opt, split, i))

    def __getitem__(self, index):
        ret = []
        for i in range(len(self.loaders)):
            ret.append(self.loaders[i].__getitem__(index))

        return ret

    def __len__(self):
        return self.loaders[0].__len__()