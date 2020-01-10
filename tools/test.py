import numpy as np
import torch
import cv2
import time
import os
from models.Box_reg import Box_reg
import torchvision.transforms as transforms
def main():
    img_path='../bbox_test/'
    loc_normalize_std=[0.1,0.1,0.2,0.2]
    model=Box_reg()
    checkpoint=torch.load('../checkpoint/400__model.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    img_iist=os.listdir(img_path)
    for i in img_iist:
        img_data_numpy=cv2.imread(img_path+i)
        height=img_data_numpy.shape[0]
        width=img_data_numpy.shape[1]
        img_resized=cv2.resize(img_data_numpy,(128,128))
        img_gray=cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
        input=(img_gray/256.0).astype(np.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),

        ])
        input = transform(input)
        input = torch.unsqueeze(input, 0)
        output=model(input)
        print(output)
        output_cpu_with_norm=output.detach().cpu().numpy()
        output_cpu = np.array(loc_normalize_std).astype(np.float32)*output_cpu_with_norm
        bbox=loc2bbox(np.array([[0,0,height,width]]),output_cpu)
        lt=(int(bbox[0][1]),int(bbox[0][0]))
        rb=(int(bbox[0][3]),int(bbox[0][2]))
        cv2.rectangle(img_data_numpy,lt,rb,(0,255,0))
        cv2.imshow('test',img_data_numpy)
        cv2.waitKey(0)
def loc2bbox(src_bbox, loc):

    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]

    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]


    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox
if __name__ == '__main__':
    main()