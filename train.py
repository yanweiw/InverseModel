from __future__ import print_function, division
import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# label index in the poke data array (29, 1)
# index 0 is the poke index corresponding to starting image
stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
str, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image

class PokeDataset(Dataset):

    def __init__(self, dirname, size=None, transform=None):
        self.dirname = dirname
        self.data = np.loadtxt(dirname + '.txt')
        self.transform = transform
        self.size = size

    def __len__(self):
        if self.size: # in cases where I want to define size
            return self.size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        '''idx should be a single value, not list'''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == -1:
            idx = self.__len__() - 1 # a hack to prevent accessing img1 out of range
        if idx > self.__len__()-1:
            raise ValueError('idx must be smaller than the len-1\
                                in order to get two images')

        img1_name = os.path.join(self.dirname, str(int(self.data[idx, 0]))) + '.png'
        # fix the last idx using identity poke (i.e. zero poke)
        if idx == self.__len__()-1:
            img2_name = img1_name
            poke = np.array([0.0, 0.0, 0.0, 0.0], dtype='float32')
        else:
            img2_name = os.path.join(self.dirname, str(int(self.data[idx, 0])+1)) + '.png'
            poke = np.float32(self.data[idx, stx:edy+1])

        poke = np.array(poke)
        img1 = imread(img1_name)
        img2 = imread(img2_name)
        img = np.vstack([img1, img2])

        if self.transform:
            img = self.transform(img)

        sample = {'img': img,
                  'poke': poke}
        return sample



def run_experiment(seed, bsize, lr, num_epochs, nwork, train_num, valid_num, use_pretrained):
    # set seed
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)

    # build model
    print()
    if use_pretrained:
        print('Using pretrained weights...')
    else:
        print('Not using pretrained weights')
    model = models.resnet18(pretrained=use_pretrained)
    model.fc = nn.Linear(512, 4)

    # specify data folders
    train_dirs = ['data/image_21', 'data/image_22','data/image_23', 'data/image_24',
                      'data/image_25', 'data/image_26','data/image_27', 'data/image_30',
                      'data/image_31', 'data/image_32']

    valid_dirs = ['data/image_20']#, 'data/image_28', 'data/image_38']
    data_dirs = {'train': train_dirs, 'val': valid_dirs}
    train_num_per_dir = train_num
    valid_num_per_dir = valid_num
    # data transform
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])])
    # build DataLoader
    # concatenate train sources into a single dataset
    list_of_train_sets = []
    for data_dir in train_dirs:
        pokeset = PokeDataset(data_dir, size=train_num_per_dir, transform=data_transforms)
        list_of_train_sets.append(pokeset)
    train_sets = ConcatDataset(list_of_train_sets)
    train_loader = DataLoader(train_sets, batch_size=bsize, shuffle=True, num_workers=nwork)
    # valid sources
    list_of_valid_sets = []
    for data_dir in valid_dirs:
        pokeset = PokeDataset(data_dir, size=valid_num_per_dir, transform=data_transforms)
        list_of_valid_sets.append(pokeset)
    valid_sets = ConcatDataset(list_of_valid_sets)
    valid_loader = DataLoader(valid_sets, batch_size=bsize, shuffle=False, num_workers=nwork)
    dataloaders = {'train': train_loader, 'val': valid_loader}

    # write to tensorboard images and model graphs
    writer = SummaryWriter('runs/run'+str(seed))
    dataiter = iter(train_loader)
    images = dataiter.next()['img']
    img_grid = utils.make_grid(images)
    writer.add_image('pokes', img_grid)
    writer.add_graph(model, images)
    writer.close()

    # start to run experiments
    model = model.float().cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # record time
    since = time.time()
    last_time = since
    valid_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 10 # some big number
    running_losses = {}
    print()
    print('batch size ', bsize)
    print('learning rate', lr)
    print('train sets ', train_dirs)
    print()

    # running through epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                running_losses[phase] = 0.0
            else:
                model.eval()
                running_losses[phase] = {}
                for data_dir in valid_dirs:
                    running_losses[phase][data_dir] = 0.0

            # iterate over data
            for batch_iter, batched_sample in enumerate(dataloaders[phase]):
                inputs = batched_sample['img']
                labels = batched_sample['poke']
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero parameter gradients
                optimizer.zero_grad()
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # L1 loss
                    loss = torch.abs(outputs - labels).mean()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # log statistics
                if phase == 'train':
                    running_losses[phase] += loss.item() * inputs.size(0)
                    if batch_iter % 10 == 0:
                        writer.add_scalar('training_loss', running_losses[phase] / \
                        ((batch_iter+1)*bsize), epoch*len(train_loader) + batch_iter)
                        # len(train_loader) gives how many batches are there in a loader
                else:
                    running_losses[phase][data_dir] += loss.item()*inputs.size(0)
                    writer.add_scalar('valid_loss', running_losses[phase][data_dir] / \
                    ((batch_iter+1)*bsize), epoch*len(valid_loader) + batch_iter)

        # print training losses
        epoch_loss = running_losses['train'] / (len(train_dirs)*train_num_per_dir)
        train_loss_history.append(epoch_loss)
        print('Training Loss: {:.4f}'.format(epoch_loss))
        # print validation losses
        total_val_loss = 0.0 # of all the validation sets
        for data_dir in valid_dirs:
            ave_val_loss = running_losses['val'][data_dir] / valid_num_per_dir
            total_val_loss += ave_val_loss
            print('Val {}: {:.4f}'.format(data_dir, ave_val_loss))
            valid_loss_history.append(ave_val_loss)
        if total_val_loss < lowest_loss:
            lowest_loss = total_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print('updating best weights')

        # print running time
        curr_time = time.time()
        epoch_time = curr_time - last_time
        print('Current epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        time_elapsed = curr_time - since
        print('Training so far takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))
        last_time = curr_time
        print()

    # save best model weights
    print("saving model's learned parameters...")
    save_path = 'weights/run' + str(seed) + '_bs' + str(bsize) \
                + '_lr' + str(lr) + '_ep' + str(num_epochs) +'.pth'
    torch.save(model.state_dict(), save_path)
    print("model saved to ", save_path)
    print()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # omit argument to produce no rendering
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--bsize', type=int, default=500, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=30, help='num of epochs')
    parser.add_argument('--nwork', type=int, default=8, help='num of workers')
    parser.add_argument('--train_size', type=int, default=5000, help='num of data from each train dir')
    parser.add_argument('--valid_size', type=int, default=1500, help='num of data from each valid dir')
    parser.add_argument('--use_init', action='store_true', help='use pretrained weights')
    args = parser.parse_args()
    run_experiment(seed=args.seed, bsize=args.bsize, lr=args.lr,
        num_epochs=args.epoch, nwork=args.nwork, train_num=args.train_size,
        valid_num=args.valid_size, use_pretrained=args.use_init)
