from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math
from .image import flip, color_aug
from .image import get_affine_transform, affine_transform
from .image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from .image import draw_dense_reg
from .kitti_base import KittiBase
from .kitti_opts import opt

class KittiCtdet(KittiBase):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), opt.max_objs)

    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    inp = cv2.resize(img, (opt.input_w, opt.input_h)).transpose(2, 0, 1)
    output_h = opt.input_h // opt.down_ratio
    output_w = opt.input_w // opt.down_ratio

    hm = np.zeros((opt.num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((opt.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((opt.max_objs, 2), dtype=np.float32)
    ind = np.zeros((opt.max_objs), dtype=np.int64)
    reg_mask = np.zeros((opt.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((opt.max_objs, opt.num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((opt.max_objs, opt.num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(opt.cat_ids[ann['category_id']])
      bbox[[0, 2]] = np.clip(bbox[[0, 2]] * opt.input_w / width, 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]] * opt.input_h / height, 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = opt.hm_gauss if opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret