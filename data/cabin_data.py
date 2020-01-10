import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np


class CabinData(Dataset):
    def __init__(self, cfg, trans):
        super(CabinData, self).__init__()

        self.input_size = cfg.input_size
        self.front_or_back = cfg.front_or_back
        self.image_path = cfg.train_data_folder + 'images/' + self.front_or_back
        if self.front_or_back == 'front':
            self.label_path = cfg.train_data_folder + 'anno/' + 'txt_front.txt'
        else:
            self.label_path = cfg.train_data_folder + 'anno/' + 'txt_back.txt'
        f = open(self.label_path)
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.trans = trans
        all_data = f.readlines()
        self.dataset = []
        for data in all_data:
            data.strip()
            dataset_dict = dict()
            data_list = data.split()
            dataset_dict['img_name'] = data_list[0]
            dataset_dict['bbox'] = data_list[1:-1]
            dataset_dict['label'] = data_list[-1]

            self.dataset.append(dataset_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_current = self.dataset[item]
        img_path = self.image_path + '/' + data_current['img_name']
        img_numpy = cv2.imread(img_path)

        bbox = list(map(eval, data_current['bbox']))
        label = data_current['label']
        crop_scale_w = np.clip(1 + np.random.rand(), 1, 1.3)
        crop_scale_h = np.clip(1 + np.random.rand(), 1, 1.2)

        c = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)
        c_new = c + np.clip(np.random.randn() * 100, -100, 100)
        new_x1 = np.clip(c_new[0] - crop_scale_w * (bbox[2] - bbox[0]) / 2, 0, bbox[0])
        new_y1 = np.clip(c_new[1] - crop_scale_h * (bbox[3] - bbox[1]) / 2, 0, bbox[1])
        new_x2 = np.clip(c_new[0] + crop_scale_w * (bbox[2] - bbox[0]) / 2, bbox[2], img_numpy.shape[1])
        new_y2 = np.clip(c_new[1] + crop_scale_h * (bbox[3] - bbox[1]) / 2, bbox[3], img_numpy.shape[0])
        cropped_img = img_numpy[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
        new_bbox = [(bbox[1] - new_y1), (bbox[0] - new_x1), (bbox[3] - new_y1),(bbox[2] - new_x1)]
        cv2.rectangle(cropped_img,(int(new_bbox[1]),int(new_bbox[0])),(int(new_bbox[3]),int(new_bbox[2])),(0,255,0))
        cv2.imshow('test',cropped_img)
        cv2.waitKey(0)
        resized_img = cv2.resize(cropped_img, self.input_size)
        resized_img_random_bri = self._brightness(resized_img)
        resized_img_gray = cv2.cvtColor(resized_img_random_bri, cv2.COLOR_BGR2GRAY)

        input_img = (resized_img_gray / 256.0).astype(np.float32)
        input_img = self.trans(input_img)
        scale_x = cropped_img.shape[1] / self.input_size[1]
        scale_y = cropped_img.shape[0] / self.input_size[0]
        # new_bbox = list(map(int, new_bbox))
        if cropped_img.shape[0]<=0:
            print(img_path)
        target = self._bbox2loc(np.array([0, 0, cropped_img.shape[0], cropped_img.shape[1]]).astype(np.float32),
                                np.array(new_bbox).astype(np.float32),img_path)
        target = ((target - np.array(self.loc_normalize_mean, np.float32)
                   ) / np.array(self.loc_normalize_std, np.float32)).astype(np.float32)
        print(target)
        gt = dict()
        gt['target'] = target
        gt['img'] = input_img
        gt['bbox'] = new_bbox
        gt['label'] = label
        return gt

    def _brightness(self, image, min=0.7, max=1.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_br = np.random.uniform(min, max)
        mask = hsv[:, :, 2] * random_br > 255
        v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
        hsv[:, :, 2] = v_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _bbox2loc(self, src_bbox, dst_bbox,imgname):
        """Encodes the source and the destination bounding boxes to "loc".

        Given bounding boxes, this function computes offsets and scales
        to match the source bounding boxes to the target bounding boxes.
        Mathematcially, given a bounding box whose center is
        :math:`(y, x) = p_y, p_x` and
        size :math:`p_h, p_w` and the target bounding box whose center is
        :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
        :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

        * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
        * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
        * :math:`t_h = \\log(\\frac{g_h} {p_h})`
        * :math:`t_w = \\log(\\frac{g_w} {p_w})`

        The output is same type as the type of the inputs.
        The encoding formulas are used in works such as R-CNN [#]_.

        .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
        Rich feature hierarchies for accurate object detection and semantic \
        segmentation. CVPR 2014.

        Args:
            src_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
                These coordinates are
                :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            dst_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`.
                These coordinates are
                :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

        Returns:
            array:
            Bounding box offsets and scales from :obj:`src_bbox` \
            to :obj:`dst_bbox`. \
            This has shape :math:`(R, 4)`.
            The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

        """

        height = src_bbox[2] - src_bbox[0]
        width = src_bbox[3] - src_bbox[1]
        ctr_y = src_bbox[0] + 0.5 * height
        ctr_x = src_bbox[1] + 0.5 * width

        base_height = dst_bbox[2] - dst_bbox[0]
        base_width = dst_bbox[3] - dst_bbox[1]
        base_ctr_y = dst_bbox[0] + 0.5 * base_height
        base_ctr_x = dst_bbox[1] + 0.5 * base_width

        eps = np.finfo(height.dtype).eps
        height = np.maximum(height, eps)
        width = np.maximum(width, eps)

        dy = (base_ctr_y - ctr_y) / height
        dx = (base_ctr_x - ctr_x) / width
        if height<=0:
            print(imgname,height)
        dh = np.log(base_height / height)
        dw = np.log(base_width / width)

        loc = np.vstack((dy, dx, dh, dw)).transpose()
        return loc
