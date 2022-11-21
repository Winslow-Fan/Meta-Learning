import pandas as pd
import torch
import numpy as np
import os
from PIL import Image
# import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy.random as random

class MultiSetLoader(Dataset):
    def __init__(self, args, transform=None, mode='train'):
        super(MultiSetLoader, self).__init__()
        self.path = args.data_dir
        self.name = args.dataset_name
        self.mode = mode
        # print(self.name)
        self.transform = transform
        # self.times = args.times
        # self.way = args.way
        self.target = None
        self.target_category = None
        data_train = []
        data_test = []
        data_val = []
        data_name_list = []
        fault_types = []
        data_path = os.path.join(self.path, self.name)
        if self.name is 'NEU':
            i = 0
            for root, dirs, files in os.walk(data_path):
                for f in files:
                    if f.split('.')[-1] == 'bmp':
                        fault_type = f.split('.')[0].split('_')[0]
                        sample_index = int(f.split('.')[0].split('_')[1])
                        if fault_type not in fault_types:
                            fault_types.append(fault_type)
                            # fault_index = fault_types.index(fault_type)
                        fault_index = fault_types.index(fault_type)
                        i += 1
                        data_name_list.append(f)
                        img_path = os.path.join(self.path, self.name, f)
                        if sample_index <= 210:
                            data_train.append([i, sample_index, Image.open(img_path).convert('L'), fault_index])
                        elif sample_index > 210 and sample_index <= 270:
                            data_val.append([i, sample_index, Image.open(img_path).convert('L'), fault_index])
                        else:
                            data_test.append([i, sample_index, Image.open(img_path).convert('L'), fault_index])
        elif self.name == 'mini-imagenet':
            print('Loading mini-ImageNet')
            data_path = '/Users/winslowfan/Documents/Chongqing/Meta-Learning/prototypical-networks/data/miniImagenet/data/'
            data = []
            fault_types = []
            modes = ['train', 'val', 'test']
            data.append(pd.read_csv('/Users/winslowfan/Documents/Chongqing/Meta-Learning/prototypical-networks/data/miniImagenet/splits/ravi/train.csv'))
            data.append(pd.read_csv('/Users/winslowfan/Documents/Chongqing/Meta-Learning/prototypical-networks/data/miniImagenet/splits/ravi/val.csv'))
            data.append(pd.read_csv('/Users/winslowfan/Documents/Chongqing/Meta-Learning/prototypical-networks/data/miniImagenet/splits/ravi/test.csv'))
            for i in range(3):
                mode = modes[i]
                data_list = data[i]
                for j in range(len(data_list['label'])):
                    fault_cls = data_list['label'][j]
                    if fault_cls not in fault_types:
                        fault_types.append(fault_cls)
                        fault_index = fault_types.index(fault_cls)
                        # sample_name = data_list['filename'][j].replace(fault_cls, '')
                        img_dir = os.path.join(data_path, mode, fault_cls)
                        for _, _, files in os.walk(img_dir):
                            for f in files:
                                img_path = os.path.join(img_dir, f)
                                sample_index = int(f.split('.')[0])
                                if i == 0:
                                    data_train.append([j, sample_index, Image.open(img_path).convert('L'), fault_index])
                                elif i == 1:
                                    data_val.append([j, sample_index, Image.open(img_path).convert('L'), fault_index])
                                else:
                                    data_test.append([j, sample_index, Image.open(img_path).convert('L'), fault_index])


        df_train = pd.DataFrame(data_train, columns=['index', 'sample_index', 'image', 'label'])
        df_val = pd.DataFrame(data_val, columns=['index', 'sample_index', 'image', 'label'])
        df_test = pd.DataFrame(data_test, columns=['index', 'sample_index', 'image', 'label'])
        print(df_train.describe())
        if mode is 'train':
            self.data = df_train
        elif mode is 'test':
            self.data = df_test
        else:
            self.data = df_val

        self.num_classes = len(fault_types)
        self.length = len(data_name_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.data.image[int(index)]
        if self.transform:
            x = self.transform(x)
        return x, self.data.label[int(index)]