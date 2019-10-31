import random
import torch
from data.kitti_ctdet_multi import KittiCtdetMulti
from model import StereoTrack

dataset = KittiCtdetMulti(True, "3dop", "train", 1)
dataset.__getitem__(0)

