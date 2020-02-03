from train import *

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
