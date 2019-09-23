import argparse
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import SMART_FEDERAL as SF
from distillation import *
from train import train
from test import test
import time
from model import CNN_Model_withDropout
import os
import configparser
import csv
import configparser
from utils import XKs, Measure, get_result, adjust_learning_rate
from dataloader import MyDataset

aggregation_method_pool = ["naive", "bsz_average", "weight_average", "distillation"]

parser = argparse.ArgumentParser(description='PyTorch: Deep Mutual Learning')
parser.add_argument('--world_size', type=int, default=4, metavar='N',
                    help='the number of nodes (default: 4)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                    help='number of epochs to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--aggregation_method', choices=aggregation_method_pool, type=str, default='naive',
                    help='choices of aggregation method')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--rank', type=int,
                    help='the id of node. 0 is master and others are servants')
parser.add_argument('--backend', type=str, default='nccl',
                    help='communication protocol')
parser.add_argument('--data_path', type=str, default='/dataset',
                    help='data path')
parser.add_argument('--load_model', type=bool, default=False,
                    help='load exist model or not')
parser.add_argument('--model_path', type=str, default=None,
                    help='model path if load model is True')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                    help='weight decay')
args = parser.parse_args()


def main(world_size, epochs, rank, batch_size=200, backend='nccl', data_path='/dataset',
         lr=1e-3, momentum=0.01, weight_decay=1e-5,no_cuda=False, seed=35, aggregation_method='naive',
         load_model=False, load_path='/data'
         ):
    '''Main Function'''


    use_cuda = not no_cuda and torch.cuda.is_available()  # 使用的使用多个gpu进行训练，需要改进一下
    torch.manual_seed(seed)
    timeline = time.strftime('%m%d%Y_%H:%M', time.localtime())
    device = torch.device("cuda" if use_cuda else "cpu")

    ratio = 0.8551957853612336

    weight = torch.FloatTensor([ratio, 1 - ratio]).to(device)
    Loss = NN.BCELoss(weight=weight)
    start_epoch = 1

    result_dir = 'logs' + '/' + timeline + '/' + 'results' + '/' + '{:s}' + '/'
    model_dir = 'logs' + '/' + timeline + '/' + 'models' + '/' + '{:s}' + '/'
    param_dir = 'logs' + '/' + timeline + '/'
    csvname = '{:s}_log.csv'
    modelname = 'model_{:d}.pth'
    paramname = 'param.csv'

    param = {'world_size':  world_size,
             'batch_size':  batch_size,
             'bachend':     backend,
             'lr':          lr,
             'momentum':    momentum,
             'weight_decay':weight_decay,
             'seed':        seed,
             'aggregation': aggregation_method}

    if rank == 0:
        name = 'master'
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        with open(param_dir + paramname, 'a', newline='') as p:
            fieldnames = param.keys()
            writer = csv.DictWriter(p, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(param)
            p.close()

        if aggregation_method == 'distillation':
            DistillationData = MyDataset(root=data_path, train=True, data_root='distillation.csv')
            distillation_dataloader = DataLoader(dataset=DistillationData, batch_size=batch_size,
                                                 shuffle=True, drop_last=True)

        model_set = []
        for worker_id in range(world_size):
            model_set.append(CNN_Model_withDropout().to(device))

        opt_set = []
        for worker_id in range(world_size):
            opt_set.append(optim.SGD(model_set[worker_id].parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay))

        if load_model:
            if aggregation_method == 'distillation':
                raise ValueError('Unexpected model')
            checkpoint = torch.load(load_path)
            for worker_id in range(world_size):
                model_set[worker_id].load_state_dict(checkpoint['model_state_dict'])
                opt_set[worker_id].load_state_dict(checkpoint['opt_state_dict'])
            start_epoch = checkpoint['epoch']


        model = SF.Master(model=model_set[0], backend=backend, rank=rank, world_size=world_size, learning_rate=lr,
                          device=device, aggregation_method=aggregation_method)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            model.step(model_buffer=model_set)
            model.update(model_set[1:])
            for worder_id in range(world_size):
                adjust_learning_rate(opt_set[worker_id], epoch, lr)
            if aggregation_method == 'distillation':
                distillation(NN_set=model_set[1:], opt_set=opt_set[1:], dataset=distillation_dataloader,
                             world_size=world_size, epoch=epoch, device=device)
            #                 best_idx = choose_best(NN_set=model_set[1:], name=name, dataset=dataloader,world_size=world_size,
    #                                        epoch=epoch, Loss=Loss, time=timeline)
    #                 best_state_dict = model_set[best_idx+1].state_dict()
    #                 model_set[0].load_state_dict(best_state_dict)

    # 这里要回传所有的模型

    else:
        name = 'worker' + str(rank)

        DataSet_train = MyDataset(root=data_path, train=True, data_root='{}.csv'.format(name))
        dataloader_train = DataLoader(dataset=DataSet_train, batch_size=batch_size, shuffle=True,
                                      drop_last=True)
        DataSet_test = MyDataset(root=data_path, train=True, data_root='{}.csv'.format('test'))
        dataloader_test = DataLoader(dataset=DataSet_test, batch_size=batch_size, shuffle=True,
                                     drop_last=True)

        model_set = []
        for worker_id in range(world_size):
            model_set.append(CNN_Model_withDropout().to(device))
        train_model = model_set[0]
        optimizer = optim.SGD(train_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        if load_model:
            checkpoint = torch.load(load_path)
            for worker_id in range(world_size):
                model_set[worker_id].load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            start_epoch = checkpoint['epoch']
            train_model = model_set[0]


        backup_model = CNN_Model_withDropout().to(device)

        model = SF.Servent(model=train_model, backend=backend, rank=rank, world_size=world_size,
                           device=device, aggregation_method=aggregation_method)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            model.step(model_buffer=model_set, rank=rank)

            best_state_dict = train_model.state_dict()
            backup_model.load_state_dict(best_state_dict)
            adjust_learning_rate(optimizer, epoch, lr)

            train(dataloader=dataloader_train, model=train_model, optimizer=optimizer, Loss=Loss,
                  epoch=epoch, time=timeline, result_dir=result_dir.format(name), model_dir=model_dir.format(name),
                  device=device, csvname=csvname.format(name), modelname=modelname)

            model.update(backup_model)

            test(dataloader=dataloader_test, model=train_model, epoch=epoch, Loss=Loss, time=timeline,
                 result_dir=result_dir.format(name), model_dir=model_dir.format(name), csvname=csvname.format(name),
                 modelname=modelname, device=device)


if __name__=='__main__':
    selfname = os.path.basename(os.path.realpath(__file__))
    if 'worker' in selfname:
        rank = selfname.split['worker'][1][0]
    elif 'master' in selfname:
        rank = 0
    else:
        raise ValueError('unexpected filename, please named your file as master.py or worker#.py')

    config = configparser.ConfigParser()
    config.read('config.ini')
    world_size = config['DEFAULTS']['world_size']
    lr = config['DEFAULTS']['lr']
    momentum = config['DEFAULTS']['momentum']
    weight_decay = config['DEFAULTS']['weight_decay']
    batch_size = config['DEFAULTS']['batch_size']
    backend = config['DEFAULTS']['backend']
    aggregation_method = config['DEFAULTS']['aggregation']
    epochs = config['DEFAULTS']['epoch']
    no_cuda = config['DEFAULTS']['no_gpu']
    seed = config['DEFAULTS']['seed']
    data_path = config['DEFAULTS']['datapath']
    model_path = config['DEFAULTS']['modelpath']
    load_model = config['DEFAULTS']['loadmodel']


    # main(world_size=args.world_size, epochs=args.epochs, rank=args.rank, batch_size=args.batch_size,
    #      backend=args.backend, data_path=args.data_path, lr=args.lr, momentum=args.momentum,
    #      no_cuda=args.no_cuda, seed=args.seed, aggregation_method=args.aggregation_method,
    #      weight_decay=args.weight_decay, load_model=args.load_model, load_path=args.load_path)

    main(world_size=world_size, epochs=epochs, rank=rank, batch_size=batch_size,
         backend=backend, data_path=data_path, lr=lr, momentum=momentum,
         no_cuda=no_cuda, seed=seed, aggregation_method=aggregation_method,
         weight_decay=weight_decay, load_model=load_model, load_path=model_path)
