from train import *

def predict(data_num, model_path, experiment_tag, size, transform):
    data_dir = 'data/image_' + data_num
    # decide training objective: predictions in world, joint, or pixel space
    if experiment_tag == 'world':
        start_label, end_label = stx, oby
    elif experiment_tag == 'joint':
        start_label, end_label = js1, je6
    elif experiment_tag == 'pixel':
        start_label, end_label = sr, obc
    else:
        raise Exception("experiment_tag has to be 'world', 'joint', or 'pixel'")

    pokeset = PokeDataset(data_dir, start_label, end_label, size, transform)
    testloader = DataLoader(pokeset, batch_size=500, shuffle=False, num_workers=8)

    pred_pokes = []
    true_pokes = []
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, end_label-start_label+1)
    model = model.float().cuda()
    model = nn.DataParallel(model)
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
    np.savetxt('eval/pred_' + data_num + '_' + experiment_tag + '_' + model_path[11:14]+ '.txt', pokes)
    print('saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data dir num')
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--tag', required=True, help='experiment tag')
    parser.add_argument('--size', type=int, default=2000, help='test case num')
    args = parser.parse_args()
    predict(args.data, args.model, args.tag, args.size, default_transform)
