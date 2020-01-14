from __future__ import print_function, division
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# label index in the poke data array (23, 1)
#index 0 is the poke index corresponding to starting image
stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 #jpos after poke

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


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 4)
model = model.float().cuda()
model = nn.DataParallel(model)

train_dirs = ['data/image_21', 'data/image_22','data/image_23', 'data/image_24',
                  'data/image_25', 'data/image_26','data/image_27', 'data/image_30',
                  'data/image_31', 'data/image_32']
valid_dirs = ['data/image_20', 'data/image_28', 'data/image_38']
data_dirs = {'train': train_dirs, 'val': valid_dirs}
train_num_per_dir = 5000
valid_num_per_dir = 500
num_epochs=30
bsize = 500 # batch size
nwork = 8 # num of workers

data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                           std=[0.5, 0.5, 0.5])])
train_loaders = {}
valid_loaders = {}

for data_dir in train_dirs:
    pokeset = PokeDataset(data_dir, size=train_num_per_dir, transform=data_transforms)
    loader = DataLoader(pokeset, batch_size=bsize, shuffle=True, num_workers=nwork)
    train_loaders[data_dir] = loader

for data_dir in valid_dirs:
    pokeset = PokeDataset(data_dir, size=valid_num_per_dir, transform=data_transforms)
    loader = DataLoader(pokeset, batch_size=bsize, shuffle=False, num_workers=nwork)
    valid_loaders[data_dir] = loader

dataloaders = {'train': train_loaders, 'val': valid_loaders}

optimizer = optim.Adam(model.parameters(), lr=1e-1)

since = time.time()
last_time = since
valid_loss_history = []
train_loss_history = []
best_model_wts = copy.deepcopy(model.state_dict())
lowest_loss = 10 # some big number
running_losses = {}

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)
    print('train sets ', train_dirs)
    print('batch size ', bsize)

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

        for data_dir in data_dirs[phase]:
            for batch_iter, batched_sample in enumerate(dataloaders[phase][data_dir]):
                inputs = batched_sample['img']
                labels = batched_sample['poke']
                inputs = inputs.cuda()
                labels = labels.cuda()

                """
                im_a, im_b = inputs[0, 0], inputs[0, 1]
                im_a = (im_a.permute(1, 2, 0) + 1.0) / 2.0
                im_b = (im_b.permute(1, 2, 0) + 1.0) / 2.0
                im_a, im_b = im_a.cpu().numpy(), im_b.cpu().numpy()
                plt.figure()
                plt.imshow(im_a)
                plt.figure()
                plt.imshow(im_b)
                print(labels[0])
                assert False
                """

                #print(inputs.min(), inputs.max(), inputs.size())
                #print(labels.size(), labels.min(), labels.max())
                #assert False
                # zero parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
#                     if batch_iter % 50 == 0:
#                         print('gt: {}'.format(labels[0]))
#                         print('pr: {}'.format(outputs[0]))
                    loss = torch.abs(outputs - labels).mean()
#                     criterion = nn.SmoothL1Loss()
#                     criterion = nn.MSELoss()
#                     loss = criterion(outputs, labels)
#                     print(phase, loss.item())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    running_losses[phase] += loss.item() * inputs.size(0)
                else:
                    running_losses[phase][data_dir] += loss.item()*inputs.size(0)



    # print losses
    total_val_loss = 0.0
    for data_dir in valid_dirs:
        ave_val_loss = running_losses['val'][data_dir] / valid_num_per_dir
        total_val_loss += ave_val_loss
        print('Val {} loss: {:.4f}'.format(data_dir, ave_val_loss))
        valid_loss_history.append(ave_val_loss)
    if total_val_loss < lowest_loss:
        lowest_loss = total_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print('updating best weights')

    epoch_loss = running_losses['train'] / (len(train_dirs)*train_num_per_dir)
    train_loss_history.append(epoch_loss)
    print('Loss: {:.4f}'.format(epoch_loss))

    # print time
    curr_time = time.time()
    epoch_time = curr_time - last_time
    print('Epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    time_elapsed = curr_time - since
    print('Training so far takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    last_time = curr_time
    print()

# save best model weights
torch.save(best_model_wts, 'weights/')
