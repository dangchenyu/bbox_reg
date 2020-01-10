import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import cv2
import numpy
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse
from models.Box_reg import Box_reg
from data.cabin_data import CabinData
from utils.smooth_L1_loss import _smooth_l1_loss


def parse_arg():
    parser = argparse.ArgumentParser('train bbox regression')
    parser.add_argument('--train_data_folder', default='../data/', type=str)
    parser.add_argument('--input_size', default=(128, 128), type=tuple)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--gpus', default=(0,1,2), help='gpus available for training')
    parser.add_argument('--front_or_back', default='', help='choose front or back for training')
    parser.add_argument('--epochs', default=401, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    args = parser.parse_args()
    return args


def main():
    model.train()
    global global_step
    for ind, data in enumerate(train_loader):
        global_step += 1
        optimizer.zero_grad()
        input = data['img']
        output = model(input.to(device))
        output = torch.unsqueeze(output, 1).to(device)
        target = data['target'].to(device)
        loss = _smooth_l1_loss(output, target).to(device)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss, global_step=global_step)
        if ind % 100 == 0:
            print("epoch:{},iteration:{},global_step:{},loss:{}".format(i, ind, global_step, loss))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8'
    cfg = parse_arg()
    global_step=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Box_reg()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9)
    trans = transforms.ToTensor()
    Dataset = CabinData(cfg, trans)
    writer = SummaryWriter()
    train_loader = DataLoader(Dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    if device == 'cuda':
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus).cuda()
        print("Data paralleling...")
        cudnn.benchmark = True
    print('Using:',device)
    print('loading {} items'.format(Dataset.__len__()))
    for i in range(cfg.epochs):
        main()
        if i % 20 == 0:
            state_dict = model.module.state_dict()
            print('saving checkpoint...')
            torch.save(state_dict, '../checkpoint/' + cfg.front_or_back+str(i) + '_' + '_model.pth')
    writer.close()
