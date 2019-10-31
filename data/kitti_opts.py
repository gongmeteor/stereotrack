import math
import numpy as np

class KittiOpt(object):
      def __init__(self, *args, **kwargs):
            self.data_dir = '/home/chris/data'
            self.num_classes = 3
            self.input_w = 1280
            self.input_h = 384

            self.max_objs = 50
            self.class_name = ['__background__', 'Pedestrian', 'Car', 'Cyclist']
            self.cat_ids = {1:0, 2:1, 3:2, 4:-3, 5:-3, 6:-2, 7:-99, 8:-99, 9:-1}

            self.down_ratio = 4
            self.mse_loss = False
            self.hm_gauss = 5

opt = KittiOpt()