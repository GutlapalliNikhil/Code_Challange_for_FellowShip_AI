import argparse
import json
import os

import torch
import torch.nn.functional as F

import time

from datasets import load_class_names, separate_class, prepare_loader

import torchvision.models as models
import torch.nn as nn

def test_v1(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    runcount = 0
    elapsed = 0
    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)

            loss = F.cross_entropy(pred, target) * data.size(0)
            acc = pred.max(1)[1].eq(target).float().sum()

            loss_meter += loss.item()
            acc_meter += acc.item()
            i += 1
            elapsed = time.time() - start_time
            runcount += data.size(0)

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} ({elapsed:.2f}s)', end='\r')

        print()

        loss_meter /= runcount
        acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_time': elapsed
    }

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    return valres


def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = json.load(open(args.config))
    config['imgsize'] = (args.imgsize, args.imgsize)

    exp_dir = os.path.dirname(args.config)
    modelpath = exp_dir + '/best.pth'

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())


    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 196)

    load_weight(model, modelpath, device)
    model = model.to(device)

    train_loader, test_loader = prepare_loader(config)

    test_fn = test_v1

    test_fn(model, test_loader, device, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for Cars dataset')

    parser.add_argument('--config', required=True,
                        help='path to config.json')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='img size for testing (default: 400)')

    args = parser.parse_args()

    main(args)
