from __future__ import print_function, division
import os
import time
import copy
import torch
import torch.nn as nn
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
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
            img = self.transform(img)

#         sample = {'img1': img1, 'img2': img2, 'poke': poke}
#         img = np.vstack((img1, img2))
#         pokevec = np.array([poke[2]-poke[0], poke[3]-poke[1]])
        sample = {'img': img,
                  'poke': poke}

        return sample


# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
#
# class Siamese(nn.Module):
#     def __init__(self, use_init):
#         super(Siamese, self).__init__()
#         self.base = models.resnet18(pretrained=use_init)
# #         self.base.conv1 = nn.Conv2d(5, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
#         self.base.fc = Identity()
#         self.fc = nn.Sequential(nn.Linear(in_features=2*512, out_features=4))
#
#     def forward(self, x):
#         '''x of size (batch, 2, C, H, W)'''
# #         feat_1, feat_2 = torch.split(x, 1, 1)
# #         feat_1, feat_2 = torch.squeeze(feat_1, 1), torch.squeeze(feat_2, 1)
# #         feat_1 = self.base(feat_1)
# #         feat_2 = self.base(feat_2)
#
# #         feat = torch.cat([feat_1, feat_2], 1)
#         x = x.reshape((-1, x.size(2), x.size(3), x.size(4)))
#         stacked_feat = self.base(x)
#         feat_1, feat_2 = stacked_feat[:x.size(0)//2], stacked_feat[x.size(0)//2:]
#         feat = torch.cat([feat_1, feat_2], 1)
#         out = self.fc(feat)
#         return out


def train(use_4_gpus=True, lr=0.1, bsize=512, nwork=16, num_epochs=30):
    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = Siamese(use_init=use_init)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 4)
    if use_4_gpus:
        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), "GPUs")
            model = model.float().cuda()
            model = nn.DataParallel(model)
    else:
        model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    # data folders
    train_dirs = ['image_21', 'image_22']#,'image_23', 'image_24', 'image_25', 'image_26']#,
    #               'image_27', 'image_30', 'image_31', 'image_32']
    valid_dirs = ['image_20', 'image_28', 'image_38']
    data_dirs = {'train': train_dirs, 'val': valid_dirs}
    train_num_per_dir = 5000
    valid_num_per_dir = 500

    # dataloading
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


    # training
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
                        # if batch_iter % 50 == 0:
                            # print('gt: {}'.format(labels[0]))
                            # print('pr: {}'.format(outputs[0]))
                        # loss = torch.abs(outputs - labels).mean()
                        loss = criterion(outputs, labels)
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
    return valid_loss_history, train_loss_history
    # torch.save(best_model_wts, 'weights/')
    # torch.save(best_model_wts, 'weights/10k_pos_512bsize.pth.tar')

def plot_loss():
    vl1 = valid_loss_history[6::3]
    vl2 = valid_loss_history[7::3]
    vl3 = valid_loss_history[8::3]
    plot_list = [vl1, vl2, vl3]
    plt.plot(list(map(list, zip(*plot_list))))
    plt.plot(train_loss_history[0:])

def record_loss(filename):
    with open(filename, 'a') as f:
        f.write('image_20 ')
        for item in vl1:
            f.write('%s ' % item)
            f.write('\n')
        f.write('image_28 ')
        for item in vl2:
            f.write('%s' % item)
            f.write('\n')
        f.write('image_38 ')
        for item in vl3:
            f.write('%s' % item)
            f.write('\n')
        f.write('training ')
        for item in train_loss_history:
            f.write('%s' % item)
            f.write('\n')

def make_pred(datafolder, savepath):
    import random
    query = random.sample(range(0, 1000), 50)
    print(query)

    pokeset = PokeDataset(datafolder, transform=data_transforms)
    pred_pokes = []
    true_pokes = []
    model.eval()
    for i in query:
        sample = pokeset[i]
        img, poke = sample['img'], sample['poke']
        img = img.expand([1, -1, -1, -1, -1])
        img = img.to(device)
        with torch.no_grad():
            pred_poke = model(img)
        pred_pokes.append(pred_poke.cpu().numpy())
        true_pokes.append(poke)

    pred_pokes = np.squeeze(np.stack(pred_pokes, axis=0))
    true_pokes = np.stack(true_pokes, axis=0)
    pokes = np.hstack([np.array(query).reshape((-1, 1)), pred_pokes, true_pokes])
    np.savetxt(savepath, pokes)


from IPython import embed

embed()
