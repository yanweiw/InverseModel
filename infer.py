from train import *

class MultiPokeSet(Dataset):

    def __init__(self, dirname, img1_name, img2_name, transform=default_transform):
        self.dirname = dirname
        self.transform=transform
        self.img1_name = img1_name
        self.img2_name = img2_name

    def __len__(self):
        return len(os.listdir(self.dirname))

    def __getitem__(self, idx):
        idxpath = str(idx).zfill(3)
        pokepath = os.path.join(self.dirname, idxpath)
        img1_path = os.path.join(pokepath, self.img1_name)
        img2_path = os.path.join(pokepath, self.img2_name)
        img1 = imread(img1_path)
        img2 = imread(img2_path)
        img = np.vstack([img1, img2])
        if self.transform:
            img = self.transform(img)
        sample = {'img': img}
        return sample


class InferOnline():

    def __init__(self, data_dir, model_path, experiment_tag):
        self.data_dir = data_dir
        # decide training objective: predictions in world, joint, or pixel space
        if experiment_tag == 'world':
            start_label, end_label = stx, oby
        elif experiment_tag == 'joint':
            start_label, end_label = js1, je6
        elif experiment_tag == 'pixel':
            start_label, end_label = sr, obc
        elif experiment_tag == 'wpoke':
            start_label, end_label = pang, plen
        else:
            raise Exception("experiment_tag has to be 'world', 'joint', 'pixel', or 'wpoke'")

        self.pokesets = []
        self.testloaders = []
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, end_label-start_label+1)
        self.model = self.model.float().cuda()
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.pred_pokes = None

    def predict(self, attempt_num):
        assert attempt_num >= 0 and attempt_num <= 4
        # add new datesets created online 
        pokeset = MultiPokeSet(self.data_dir, str(attempt_num) + '.png', '5.png')
        testloader = DataLoader(pokeset, batch_size=500, shuffle=False, num_workers=8)
        self.pokesets.append(pokeset)
        self.testloaders.append(testloader)
        assert len(self.pokesets) == attempt_num + 1
        assert len(self.testloaders) == attempt_num + 1

        with torch.no_grad():
            for data in self.testloaders[attempt_num]:
                inputs = data['img'].cuda()
                outputs = self.model(inputs).cpu().numpy()
                if self.pred_pokes is None:
                    self.pred_pokes = outputs
                else:
                    self.pred_pokes = np.vstack((self.pred_pokes, outputs))

        # make sure predictions match pokset in length
        assert len(self.pred_pokes) == len(self.pokesets[attempt_num])

        for i, p in enumerate(self.pred_pokes): 
            save_path = os.path.join(self.data_dir, str(i).zfill(3), 'pokes.txt')
            with open(save_path, 'a') as file:
                file.write('%f %f %f %f\n' % (p[0], p[1], p[2], p[3])) 

        pokelengths = np.linalg.norm(self.pred_pokes[:, :2] - self.pred_pokes[:, 2:4], axis=1)
        print('inferred pokes mean: %.4f, std: %.4f' % (pokelengths.mean(), pokelengths.std()))
        # make sure we clear self.pred_pokes for the subsequent predictions
        self.pred_pokes = None 



def predict(data_num, model_path, experiment_tag, size, transform, random_init):
    data_dir = 'data/image_' + data_num
    # decide training objective: predictions in world, joint, or pixel space
    if experiment_tag == 'world':
        start_label, end_label = stx, oby
    elif experiment_tag == 'joint':
        start_label, end_label = js1, je6
    elif experiment_tag == 'pixel':
        start_label, end_label = sr, obc
    elif experiment_tag == 'wpoke':
        start_label, end_label = pang, plen
    else:
        raise Exception("experiment_tag has to be 'world', 'joint', 'pixel', or 'wpoke'")

    pokeset = PokeDataset(data_dir, start_label, end_label, size, transform)
    testloader = DataLoader(pokeset, batch_size=500, shuffle=False, num_workers=8)

    pred_pokes = []
    true_pokes = []
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, end_label-start_label+1)

    model = model.float().cuda()
    model = nn.DataParallel(model)
    # sanity check that random weights won't predict any meaningful behaviors
    if random_init:
        print('using random weights...')
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        model.apply(weight_reset)
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for data in testloader:
            inputs = data['img'].cuda()
            outputs = model(inputs).cpu().numpy()
            pred_pokes.append(outputs)
            true_pokes.append(data['poke'].numpy())
    pred_pokes = np.vstack(pred_pokes)
    true_pokes = np.vstack(true_pokes)
    indices = np.array(range(size)).reshape(-1, 1)
    pokes = np.hstack([indices, pred_pokes, true_pokes])
    print('saving predictions...')
    save_path = 'prediction/pred_' + data_num + '_' + experiment_tag + '_' + model_path[11:14]+ '.txt'
    if random_init:
        save_path = save_path[:-4] + '_random.txt'
    np.savetxt(save_path, pokes, fmt='%.6f')
    print('saved to ', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data dir num')
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--tag', required=True, help='experiment tag')
    parser.add_argument('--size', type=int, default=2000, help='test case num')
    parser.add_argument('--random', action='store_true', help='use random weights')
    args = parser.parse_args()
    predict(data_num=args.data, model_path=args.model, experiment_tag=args.tag,
        size=args.size, transform=default_transform, random_init=args.random)
