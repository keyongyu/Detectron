# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
import cv2
from PIL import Image
from multiprocessing.pool import ThreadPool
import random
from imgaug import augmenters as iaa

# Must happen before importing COCO API (which imports matplotlib)
import detectron.utils.env as envu

envu.set_up_matplotlib()
# COCO API

from detectron.core.config import cfg
from detectron.utils.timer import Timer
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _load_template_json(template_file):
    classes = ['__background__']
    class_to_idx = {classes[0]: 0}

    with open(template_file, "rb") as fp:
        str1 = fp.read().decode('utf-8')
        labels = json.loads(str1)
        label_def = labels.get('categories', None)[0].get("skus")
        for seq, label in enumerate(label_def):
            classes.append(label.get("id"))
            class_to_idx[label.get("id")] = seq + 1
        return classes, class_to_idx


class Sku:
    def __init__(self, sku_code, sku_cls, full_path):
        self.sku_code = sku_code
        self.sku_cls = sku_cls
        self.short_file = full_path.split(os.sep)[-1]
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        w = img.shape[1]
        h = img.shape[0]
        scale = 1.0
        if w > 400:
            scale = 400. / w
            if h * scale > 600:
                scale = 600.0 / h

        w = int(w * scale)
        h = int(h * scale)

        self.img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        self.has_alpha = (img.shape[2] == 4)

    @property
    def shape(self):
        return self.img.shape

    @property
    def aspect(self):
        return float(self.img.shape[1]) / self.img.shape[0]

    def put(self, im, x, y, x2, y2, shape_augmentor, color_augmentor):
        new_x = x
        new_y = y
        new_w = min(x2, im.shape[1]) - x
        new_h = min(y2, im.shape[0]) - y

        if new_w*new_h*1.0 /(im.shape[0]*im.shape[1]) < 1/(20.0*20.0):
            #kx=new_w*1.0/im.shape[1]
            #ky=new_h*1.0/im.shape[0]
            #print("<<<<<<<<<<<<discard small sku>>>>>>>>>>> cls:%02d, kx: %.04f, ky: %.04f" %(self.sku_cls, kx,ky))
            return False

        sku_img = cv2.resize(self.img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        sku_img = shape_augmentor.augment_image(sku_img)

        alpha_channel = sku_img[:, :, 3]
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
        sku_bgr = sku_img[:, :, :3].astype(np.float32) * alpha_factor
        # sku_bgr = color_augmentor.augment_image(sku_bgr)

        alpha = alpha_factor
        bg_img = im[new_y:new_y + new_h, new_x:new_x + new_w]
        # if sku_bgr.shape != bg_img.shape:
        #    print("different shape!!!")
        new_sku_img = sku_bgr + bg_img * (1.0 - alpha)

        new_sku_img = color_augmentor.augment_image(new_sku_img)
        im[new_y:new_h + new_y, new_x:new_w + new_x] = new_sku_img

        # return new_x, new_y, new_w, new_h
        return True

    # @property
    # def shape(self):
    #    return self.rr.shape


def _load_sample_map(sample_dir, code_to_cls_id):
    '''load all png sample in grandson folder of sample_dir'''
    running_threads = []
    sample_map_ = {}

    def load_sku_sample(sku_code):
        print("loading sample sku:" + sku_code)
        my_path = os.path.join(sample_dir, sku_code)
        sample_2nd_map = {}
        for sku_file in [f for f in os.listdir(my_path) if
                         os.path.isfile(os.path.join(my_path, f))
                         and f.endswith(".png")]:
            # print("loading png file ",sku_file)
            sku = Sku(sku_code, code_to_cls_id[sku_code], os.path.join(my_path, sku_file))

            sample_2nd_map[sku.short_file] = sku

        return (sku_code, sample_2nd_map)

    folder_list = [f for f in os.listdir(sample_dir)
                   if os.path.isdir(os.path.join(sample_dir, f))
                   ]

    pool = ThreadPool(8)
    for sku_code, second_map in pool.map(load_sku_sample, folder_list):
        sample_map_[sku_code] = second_map

    return sample_map_


class NPStudioDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, data_path):
        self.name = "npstudio_watson"
        self._data_root_path = data_path
        self._data_path = os.path.join(self._data_root_path, 'photos')
        self._sku_sample_dir = os.path.join(self._data_root_path, "samples")

        assert os.path.exists(self._data_root_path), \
            'Annotation folder \'{}\' not found'.format(self._data_root_path)
        assert os.path.exists(self._data_path), \
            'photos folder \'{}\' not found'.format(self._data_path)
        assert os.path.exists(self._sku_sample_dir), \
            'sku samples folder \'{}\' not found'.format(self._sku_sample_dir)

        self.debug_timer = Timer()
        self.classes, self.category_to_id_map = _load_template_json(
            os.path.join(self._data_root_path, 'templates.json')
        )
        # self._val_count = val_count

        self._sample_map = _load_sample_map(self._sku_sample_dir, self.category_to_id_map)
        sku_width = {}
        sku_width["000"] = 120
        sku_width["001"] = 120
        sku_width["002"] = 120
        sku_width["003"] = 120
        sku_width["004"] = 120
        sku_width["005"] = 160
        sku_width["006"] = 160
        sku_width["007"] = 160
        sku_width["008"] = 125
        sku_width["009"] = 136
        sku_width["010"] = 125
        sku_width["011"] = 84
        sku_width["012"] = 158
        sku_width["013"] = 150
        sku_width["014"] = 133
        sku_width["015"] = 165
        sku_width["016"] = 148
        sku_width["017"] = 158
        sku_width["018"] = 60
        sku_width["019"] = 135
        sku_width["020"] = 120
        sku_width["021"] = 167
        sku_width["022"] = 136
        sku_width["023"] = 100
        sku_width["024"] = 105
        sku_width["025"] = 167
        sku_width["026"] = 130

        self._sku_width = [sku_width[key] for key in sorted(sku_width.keys())]

        ## batch reader ##
        self._perm_idx = None
        self._cur_idx = 0

        logger.debug('Creating: {}'.format(self.name))

        # Set up dataset classes
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = dict(zip(range(1, self.num_classes), range(1, self.num_classes)))
        self.contiguous_category_id_to_json_id = self.json_category_id_to_contiguous_id.copy()
        self._init_keypoints()
        sometimes = lambda aug: iaa.Sometimes(0.6, aug)
        self._shape_augmentor = iaa.Sequential([
            iaa.Crop(percent=(0.0, 0.08)),
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-4, 4),  # rotate by -15 to +15 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                # mode=iaa.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # mode = iaa.Choice(["edge", "symmetric", "reflect", "wrap"])
            )),
            iaa.Sometimes(0.6, iaa.PerspectiveTransform(scale=(0.01, 0.065)))
        ])

        self._color_augmentor = iaa.Sequential([
            iaa.SomeOf((0, 3),
                       [

                           sometimes(iaa.OneOf([
                               iaa.GaussianBlur(sigma=(0, 1.0)),
                               iaa.AverageBlur(k=(2, 4)),
                               iaa.MedianBlur(k=(1, 3))
                           ])),
                           iaa.Add((-30, 15), per_channel=0.5),
                           # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                           iaa.ContrastNormalization((0.7, 1.3), per_channel=0.1),
                           iaa.ContrastNormalization((0.7, 1.3)),
                           # iaa.PerspectiveTransform(scale=(0.01, 0.065)),
                           iaa.Pepper(0.01)
                       ],
                       random_order=True)

        ])

    def _load_roidb(self):
        assert os.path.exists(self._data_path), \
            'folder does not exist: {}'.format(self._data_path)
        from_dir = self._data_path
        roidb = []
        for f in os.listdir(from_dir):
            if os.path.isfile(os.path.join(from_dir, f)) and not f.startswith("."):
                try:
                    im = Image.open(os.path.join(from_dir, f))
                    width, height = im.size
                    entry = {'file_name': f, 'height': height, 'width': width}
                    roidb.append(entry)
                except:
                    continue

        # if self._val_count > 0 and self._val_count < len(image_idx):
        #    image_idx = image_idx[0:self._val_count]
        return roidb

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'

        # _filenames = self._load_image_set_idx()
        # self._sample_bboxes = {}
        # self._labels = self._load_np_annotation()
        # self._sample_map = _load_sample_map(self._sku_sample_dir)

        self._roidb = roidb = self._load_roidb()
        self._labels_by_ind, self._bboxes_to_draw_by_ind = self._load_np_annotation()

        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry in roidb:
                self._add_gt_annotations(entry)
            logger.debug(
                '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
            )
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(self._data_path, entry['file_name'])

        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        # entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )

    def draw_sku_on_image(self, entry):
        idx = entry['file_name']
        im = cv2.imread(entry['image'])
        for bbox_to_draw in self._bboxes_to_draw_by_ind[idx]:
            sku_code, png_file, rect, cls = bbox_to_draw
            assert len(png_file) > 0, "found sku(" + sku_code + ") which doesn't have png file"
            # if sku_code == '001':
            #    print("drawing sample %s" % (png_file))
            sku = self._sample_map[sku_code][png_file]
            x1, y1, x2, y2 = rect
            sku.put(im, x1, y1, x2, y2, self._shape_augmentor, self._color_augmentor)

        # cv2.imwrite("/home/keyong/test/" + idx, im)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_image(self, entry):
        # im = cv2.imread(entry['image'])
        # return im

        return self.draw_sku_on_image(entry)

    def save_image(self, im, new_boxes, idx):
        for box in new_boxes:
            cls, x1, y1, x2, y2 = box
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
        cv2.imwrite("/home/keyong/test/" + idx, im)

    def draw_random_sku(self, im, new_boxes, max_attemps=9):
        new_h = im.shape[0]
        new_w = im.shape[1]
        sku_codes = self._sample_map.values()

        ratio = random.uniform(1/(13*1.5), 1/(13*0.85))/100

        for _ in range(max_attemps):
            sku_with_same_code = sku_codes[random.randint(0, len(sku_codes) - 1)].values()
            sku = sku_with_same_code[random.randint(0, len(sku_with_same_code) - 1)]
            cls = sku.sku_cls

            aspect = sku.aspect
            aspect = random.uniform(0.9 * aspect, 1.1 * aspect)

            #23 will be 12

            w = new_w * ratio

            w = int(self._sku_width[cls]*w)

            #w = int(random.uniform(new_w * 0.04, new_w * 0.125))
            h = int(w / aspect)
            if h > new_h:
                h = new_h
                w = int(h * aspect)
            x1 = random.randint(0, new_w - w)
            y1 = random.randint(0, new_h - h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            good_try = True
            for box in new_boxes:
                l = max(x1, box[1])
                t = max(y1, box[2])
                r = min(x2, box[3])
                b = min(y2, box[4])
                if r > l and b > t:
                    good_try = False
                    break;

            if good_try:
                if sku.put(im, int(x1), int(y1), int(x2), int(y2), self._shape_augmentor, self._color_augmentor):
                    new_boxes.append([cls, x1, y1, x2, y2])

    def get_im_and_lbl(self, entry):

        idx = entry['file_name']
        im = cv2.imread(entry['image'])

        new_boxes = []
        option = random.uniform(0, 1.0)
        if option < 0.2:
            new_entry = {'file_name': idx, 'height': im.shape[0], 'width': im.shape[1]}
            self._prep_roidb_entry(new_entry)
            for bbox_to_draw in self._bboxes_to_draw_by_ind[idx]:
                sku_code, png_file, rect, cls = bbox_to_draw
                assert len(png_file) > 0, "found sku(" + sku_code + ") which doesn't have png file"
                # if sku_code == '001':
                #    print("drawing sample %s" % (png_file))
                sku = self._sample_map[sku_code][png_file]
                x1, y1, x2, y2 = rect
                x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                    x1, y1, x2, y2, im.shape[0], im.shape[1]
                )
                if sku.put(im, int(x1), int(y1), int(x2), int(y2), self._shape_augmentor, self._color_augmentor):
                    new_boxes.append([cls, x1, y1, x2, y2])

            if len(new_boxes) == 0:
                self.draw_random_sku(im, new_boxes)

            self._fill_bboxes_in_entry(new_boxes, new_entry)

        elif option <= 0.6:
            # crop
            scale = random.uniform(0.7, 1.0)
            h, w, _ = im.shape
            new_h, new_w = int(h * scale), int(w * scale)

            draft_x, draft_y = random.randint(0, w - new_w), random.randint(0, h - new_h)
            im = im[draft_y:draft_y + new_h, draft_x:draft_x + new_w]

            new_entry = {'file_name': idx, 'height': im.shape[0], 'width': im.shape[1]}
            self._prep_roidb_entry(new_entry)
            for bbox_to_draw in self._bboxes_to_draw_by_ind[idx]:
                sku_code, png_file, rect, cls = bbox_to_draw
                assert len(png_file) > 0, "found sku(" + sku_code + ") which doesn't have png file"
                x1, y1, x2, y2 = rect

                x1 = x1 - draft_x
                x2 = x2 - draft_x

                y1 = y1 - draft_y
                y2 = y2 - draft_y

                # if sku_code == '001':
                #    print("drawing sample %s" % (png_file))
                # x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                #    x1, y1, x2, y2, im.shape[0], im.shape[1]
                # )
                if x1 < 0 or x2 >= new_w \
                        or y1 < 0 or y2 >= new_h:
                    continue

                sku = self._sample_map[sku_code][png_file]
                if sku.put(im, int(x1), int(y1), int(x2), int(y2), self._shape_augmentor, self._color_augmentor):
                    new_boxes.append([cls, x1, y1, x2, y2])

            if len(new_boxes) == 0:
                self.draw_random_sku(im, new_boxes)

            self._fill_bboxes_in_entry(new_boxes, new_entry)

        else:
            zoom_out = random.uniform(0.8, 1.0)
            h, w, c = im.shape
            im = cv2.resize(im, None, fx=zoom_out, fy=zoom_out)
            new_h, new_w, _ = im.shape

            draft_x, draft_y = random.randint(0, w - new_w), random.randint(0, h - new_h)
            im2 = np.zeros((h, w, c), dtype=im.dtype)
            im2[draft_y:draft_y + new_h, draft_x:draft_x + new_w, :] = im
            im = im2

            new_entry = {'file_name': idx, 'height': im.shape[0], 'width': im.shape[1]}
            self._prep_roidb_entry(new_entry)
            for bbox_to_draw in self._bboxes_to_draw_by_ind[idx]:
                sku_code, png_file, rect, cls = bbox_to_draw
                assert len(png_file) > 0, "found sku(" + sku_code + ") which doesn't have png file"
                x1, y1, x2, y2 = rect

                x1 = x1 * zoom_out + draft_x
                x2 = x2 * zoom_out + draft_x

                y1 = y1 * zoom_out + draft_y
                y2 = y2 * zoom_out + draft_y

                # if sku_code == '001':
                #    print("drawing sample %s" % (png_file))
                # x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                #    x1, y1, x2, y2, im.shape[0], im.shape[1]
                # )

                sku = self._sample_map[sku_code][png_file]
                if sku.put(im, int(x1), int(y1), int(x2), int(y2), self._shape_augmentor, self._color_augmentor):
                    new_boxes.append([cls, x1, y1, x2, y2])


            if len(new_boxes) == 0:
                self.draw_random_sku(im, new_boxes)

            self._fill_bboxes_in_entry(new_boxes, new_entry)

        _add_class_assignments([new_entry])
        new_entry['bbox_targets'] = _compute_targets(new_entry)
        self.save_image(im, new_boxes, idx)
        return im, new_entry

    def _load_np_annotation(self):
        idx_to_annotation = {}
        idx_to_draw_sample = {}
        for entry in self._roidb:
            index = entry["file_name"]
            full_json_file = os.path.join(self._data_path, 'Annotations', index + '.json')
            with open(full_json_file, "rb") as json_fp:
                json_data = json.loads(json_fp.read().decode('utf-8'))
                bboxes = []
                sample_bboxes = []
                for bbox in json_data.get("bndboxes"):
                    w = float(bbox["w"])
                    h = float(bbox["h"])
                    xmin = float(bbox["x"])
                    ymin = float(bbox["y"])
                    sku_code = bbox["id"]

                    xmax = w + xmin
                    ymax = h + ymin

                    xmin = max(xmin, 0.0)
                    ymin = max(ymin, 0.0)

                    assert xmin >= 0.0 and xmin <= xmax, \
                        'Invalid bounding box x-coord xmin {} or xmax {} at {}.json' \
                            .format(xmin, xmax, index)
                    assert ymin >= 0.0 and ymin <= ymax, \
                        'Invalid bounding box y-coord ymin {} or ymax {} at {}.json' \
                            .format(ymin, ymax, index)
                    cls = self.category_to_id_map[sku_code]
                    # bboxes.append([cls, xmin, ymin, xmax, ymax])
                    sample_bbox = (sku_code,
                                   bbox.get("file", ""),
                                   [int(xmin), int(ymin), int(xmax), int(ymax)],
                                   cls
                                   )
                    bboxes.append([cls, int(xmin), int(ymin), int(xmax), int(ymax)])
                    sample_bboxes.append(sample_bbox)

            idx_to_annotation[index] = bboxes
            idx_to_draw_sample[index] = sample_bboxes

        return idx_to_annotation, idx_to_draw_sample

    def _fill_bboxes_in_entry(self, bboxes, entry):
        num_valid_objs = len(bboxes)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        # seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )

        for ix, obj in enumerate(bboxes):
            cls = obj[0]
            boxes[ix, :] = obj[1:5]
            gt_classes[ix] = cls
            # seg_areas[ix] = 100
            is_crowd[ix] = False
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        # entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in self._labels_by_ind[entry["file_name"]]:
            # crowd regions are RLE encoded and stored as dicts

            # if 'ignore' in obj and obj['ignore'] == 1:
            #    continue
            x1, y1, x2, y2 = obj[1:5]
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if x2 > x1 and y2 > y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj[1:5] = x1, y1, x2, y2
                valid_objs.append(obj)
                valid_segms.append([])

        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        # seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = obj[0]
            boxes[ix, :] = obj[1:5]
            gt_classes[ix] = cls
            # seg_areas[ix] = obj['area']
            # seg_areas[ix] = 100
            is_crowd[ix] = False
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True

            gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        # entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_proposals_from_file(
            self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        # if crowd_thresh > 0:
        #    _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        return None


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    # if crowd_thresh > 0:
    #    _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        # entry['seg_areas'] = np.append(
        #     entry['seg_areas'],
        #     np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        # )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


# def _filter_crowd_proposals(roidb, crowd_thresh):
#     """Finds proposals that are inside crowd regions and marks them as
#     overlap = -1 with each ground-truth rois, which means they will be excluded
#     from training.
#     """
#     for entry in roidb:
#         gt_overlaps = entry['gt_overlaps'].toarray()
#         crowd_inds = np.where(entry['is_crowd'] == 1)[0]
#         non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
#         if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
#             continue
#         crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
#         non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
#         iscrowd_flags = [int(True)] * len(crowd_inds)
#         ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
#         bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
#         gt_overlaps[non_gt_inds[bad_inds], :] = -1
#         entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
