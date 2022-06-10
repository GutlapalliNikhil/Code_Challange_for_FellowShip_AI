import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_class_names(path='devkit/class_names.csv'):
    cn = pd.read_csv(path, header=None).values.reshape(-1)
    cn = cn.tolist()
    return cn


def load_annotations_v1(path):
    ann = pd.read_csv(path, header=None).values
    ret = {}

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        r = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'target': target - 1,
            'filename': imgfn
        }

        ret[idx] = r

    return ret

def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'

        if model == 'Integra Type R':
            model_type = 'Type-R'

        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'

        if 'SRT' in model_type:
            model_type = 'SRT'

        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pd.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr


class CarsDatasetV1(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.annos = load_annotations_v1(anno_path)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]
        target = r['target']

        if idx not in self.cache:
            fn = r['filename']
            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target


def prepare_loader(config):
    train_imgdir = '/DATA/nikhil/cars/stanford-cars-original/cars_train'
    test_imgdir = '/DATA/nikhil/cars/stanford-cars-original/cars_test'

    train_annopath = 'devkit/cars_train_annos.csv'
    test_annopath = 'devkit/cars_test_annos_withlabels.csv'

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    CarsDataset = CarsDatasetV1 

    train_dataset = CarsDataset(train_imgdir, train_annopath, train_transform, config['imgsize'])
    test_dataset = CarsDataset(test_imgdir, test_annopath, test_transform, config['imgsize'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=False,
                              num_workers=12)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return train_loader, test_loader
