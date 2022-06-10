import argparse
import json
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_class_names, prepare_loader, separate_class
from test import test_v1
import numpy as np

from torch.autograd import Variable
import torch.nn as nn

import torchvision.models as models

def train_v1(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    lr_scheduler.step()
    model.train()

    loss_meter = 0
    acc_meter = 0
    i = 0
    train_loss = 0
    start_time = time.time()
    elapsed = 0
    total = 0
    correct = 0

    for data, target in train_loader:

        data = data.to(device)
        target = target.to(device)


        ####  Functions that were responsible for the MixMatch  ####
        def mixup_data(x, y, alpha=1.0, use_cuda=True):
            '''Returns mixed inputs, pairs of targets, and lambda'''
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = x.size()[0]
            if use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        def mixup_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

        
        #### Implementing MixMatch  ####
        inputs, targets_a, targets_b, lam = mixup_data(data, target, 1, use_cuda=True)

        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        outputs = model(inputs)
       
        criterion = nn.CrossEntropyLoss()

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)


        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc = outputs.max(1)[1].eq(targets_a).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)', end='\r')

    print()

    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_time': elapsed
    }

    return trainres


def get_exp_dir(config):
    exp_dir = f'logs/{config["arch"]}_{config["imgsize"][0]}_{config["epochs"]}_v{config["version"]}'

    if config['finetune']:
        exp_dir += '_finetune'

    os.makedirs(exp_dir, exist_ok=True)

    exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    files = set(map(int, exps))
    if len(files):
        exp_id = min(set(range(1, max(files) + 2)) - files)
    else:
        exp_id = 1

    exp_dir = os.path.join(exp_dir, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)

    json.dump(config, open(exp_dir + '/config.json', 'w'))

    return exp_dir


def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def main(args):
    device = 'cuda' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'imgsize': (args.imgsize, args.imgsize),
        'arch': args.arch,
        'version': args.version,
        'make_loss': args.make_loss,
        'type_loss': args.type_loss,
        'finetune': args.finetune,
        'path': args.path
    }

    exp_dir = get_exp_dir(config)

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())


    #####  Loading the Pretrained ResNet34 models  #####
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 196)  # Change the last layer of the model, from 1000 classes to 196 classes

    model = model.to(device)


    #####  Initializing SGD Optimizer  #####
    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    #####  Initializing LR Scheduler  #####
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  [100, 150],
                                                  gamma=0.1)

    train_loader, test_loader = prepare_loader(config)

    best_acc = 0
    res = []

    train_fn = train_v1
    test_fn = test_v1

    for ep in range(1, config['epochs'] + 1):

        #####  Training the model per epoch  #####
        trainres = train_fn(ep, model, optimizer, lr_scheduler, train_loader, device, config)
        
        #####  Validating the trained model  #####
        valres = test_fn(model, test_loader, device, config)

        trainres.update(valres)

        #####  Saving the model for each epoch  #####
        torch.save(model.state_dict(), exp_dir + '/best' + str(ep) + '.pth')

        res.append(trainres)

    #####  Writing the Accuracy and loss values of both train and loss functions  #####
    res = pd.DataFrame(res)
    res.to_csv(exp_dir + '/history.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification for Cars classification task using ResNet34')

    # training arg
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='training epochs (default: 40)')
    parser.add_argument('--arch', default='resnet34', choices=['resnext50',
                                                                'resnet34',
                                                                'mobilenetv2'],
                        help='Architecture (default: resnext50)')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='Input image size (default: 400)')
    parser.add_argument('--version', default=1, type=int, choices=[1, 2, 3],
                        help='Classification version (default: 1)\n'
                             '1. Cars Model only\n'
                             '2. Cars Model + Make + Car Type')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='whether to finetune from 400x400 to 224x224 (default: False)')
    parser.add_argument('--path',
                        help='required if it is a finetune task (default: None)')

    # optimizer arg
    parser.add_argument('--lr', default=0.01, type=float,
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')

    # multi-task learning arg
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--type-loss', default=0.2, type=float,
                        help='loss$_{type}$ lambda')

    args = parser.parse_args()
    main(args)
