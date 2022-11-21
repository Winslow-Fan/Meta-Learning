from prototypical_loss import prototypical_loss as loss_fn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
import argparse
from multi_dataset_loader import MultiSetLoader
from torchvision import datasets, transforms
import math
from batch_sampler import BatchSampler
from protonet import ProtoNet
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument('--data_dir', type=str, help='path to dataset',
                        default='/Users/winslowfan/Documents/Chongqing/Meta-Learning/Data/')
    parser.add_argument('--output_dir',type=str, help='path to output of model',
                        default='/Users/winslowfan/Documents/Chongqing/Meta-Learning/ProtoNet/Output/')
    parser.add_argument('--dataset_name',type=str, help='name of dataset',
                        default='NEU-CLS')
    parser.add_argument('--eps', type=int, help='number of epochs to train for', default=100)
    parser.add_argument('--lr', type=float, help='learning rate for the model, default=0.001', default=1e-3)
    parser.add_argument('--lr_decay_step', type=int, help='StepLR learning rate scheduler step, default=20',
                        default=20)
    parser.add_argument('--lr_decay_rate', type=float, help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
    parser.add_argument('--iters', type=int, help='number of episodes per epoch, default=100',
                        default=100)
    parser.add_argument('--cls_iter_tr', type=int, help='number of random classes per episode for training, default=60',
                        default=30)
    parser.add_argument('--cls_iter_val', type=int, help='number of random classes per episode for training, default=60',
                        default=30)
    parser.add_argument('--supp_tr', type=int, help='number of samples per class to use as support for training, default=5',
                        default=5)
    parser.add_argument('--query_tr', type=int, help='number of samples per class to use as query for training, default=5',
                        default=5)
    parser.add_argument('--supp_val', type=int, help='number of samples per class to use as support for validation, default=5',
                        default=5)
    parser.add_argument('--query_val', type=int, help='number of samples per class to use as query for validation, default=15',
                        default=15)
    parser.add_argument('--seed', type=int, help='input for the manual seeds initializations', default=7)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()
    print('a')
    return args

def get_data(args):
    """
    :param args: model settings and options
    :return: training set, valid set and test set. In pd.DataFrame format, columns = [total_index, index_cls, image, label]
    """
    train = MultiSetLoader(args, transform=args.train_transform, mode='train')
    test = MultiSetLoader(args, transform=args.test_transform, mode='test')
    val = MultiSetLoader(args, transform=args.val_transform, mode='val')
    return train, val, test

def sampler(dataset, args, mode):
    return BatchSampler(dataset, args, mode=mode)

def init_protonet(args):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model

def train(model, optimizer, data_tr, lr_scheduler, args, data_val=None):
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    if data_val is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H-%M-%S")

    best_model_path = os.path.join(args.output_dir, dt_string, 'best_model.pth')
    last_model_path = os.path.join(args.output_dir, dt_string, 'last_model.pth')

    for epoch in range(args.eps):
        print('--------Current epochs is {}--------'.format(epoch))
        tr_iter = iter(data_tr)
        model.train()
        for batch in tqdm(tr_iter):
            optimizer.zero_grad()
            x, y = batch
            plt.imshow(x[0,0,:,:])
            print(y)
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=args.supp_tr)
            print("Loss in training is {}, Accuracy is {}".format(loss.item(), acc.item()))

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-args.iterations:])
        avg_acc = np.mean(train_acc[-args.iterations:])
        lr_scheduler.step()

        if data_val is None:
            continue
        val_iter = iter(data_val)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.supp_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            print("Loss in validation is {}, Accuracy is {}".format(loss.item(), acc.item()))
        avg_loss = np.mean(val_loss[-args.iterations:])
        avg_acc = np.mean(val_acc[-args.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    # for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
    #     save_list_to_file(os.path.join(opt.experiment_root,
    #                                    name + '.txt'), locals()[name])
    print(best_acc)
    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


if __name__=='__main__':
    args = get_args()
    if not os.path.exists(args.output_dir):
        print("Creating the output directory...")
        os.mkdir(args.output_dir)

    # Define data pre-processing
    args.train_transform = transforms.Compose([
        transforms.RandomAffine(10, translate=(0.02, 0.02), scale=(0.8, 1.2), shear=0.3 * 180 / math.pi),
        transforms.Resize(105),
        transforms.ToTensor()
    ])
    args.test_transform = transforms.Compose([
        # transforms.RandomAffine(10, translate=(0.02, 0.02), scale=(0.8, 1.2), shear=0.3 * 180/math.pi),
        transforms.Resize(105),
        transforms.ToTensor()
    ])
    args.val_transform = transforms.Compose([
        # transforms.RandomAffine(10, translate=(0.02, 0.02), scale=(0.8, 1.2), shear=0.3 * 180/math.pi),
        transforms.Resize(105),
        transforms.ToTensor()
    ])
    # get datasets
    data_train, data_val, data_test = get_data(args)
    # get dataloader
    # print(type(data_train.data.label))

    train_set = DataLoader(data_train, batch_sampler=sampler(data_train, args, mode='train'))
    val_set = DataLoader(data_val, batch_sampler=sampler(data_val, args, mode='val'))
    test_set = DataLoader(data_test, batch_sampler=sampler(data_test, args, mode='test'))

    # print(train_set)

    model = init_protonet(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   gamma=args.lr_decay_rate,
                                                   step_size=args.lr_decay_step)


    output = train(model, optimizer, train_set, lr_scheduler, args, data_val=val_set)







