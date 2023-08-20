import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from gnn import GNN

from tqdm import tqdm
import argparse
import copy

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch_data = batch[0]
        batch_idx = batch[1]

        batch_data = batch_data.to(device)

        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            pred, _ = model(batch_data)
            optimizer.zero_grad()
            loss = multicls_criterion(pred.to(torch.float32), batch_data.y.view(-1, ))
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_data = batch[0]
        batch_idx = batch[1]

        batch_data = batch_data.to(device)

        if batch_data.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(batch_data)

            y_true.append(batch_data.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data[0].x = torch.zeros(data[0].num_nodes, dtype=torch.long)
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--split', type=str, default="species_adaptation",
                        help='dataset split type')
    parser.add_argument('--model_num', type=int, default=1,
                        help='')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    split_idx = dataset.get_idx_split(split_type=args.split)

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train_src"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["train_tar"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, virtual_node=True).to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn_node.parameters(), "lr": 0.001})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": 0.001})
    optimizer = optim.Adam(model_param_group)

    train_curve = []
    best_train_perf = 0
    best_train_epoch = 0
    best_model_train = None

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)

        print({'Train': train_perf})

        if train_perf[dataset.eval_metric] > best_train_perf:
            best_train_perf = train_perf[dataset.eval_metric]
            best_train_epoch = epoch
            best_model_train = copy.deepcopy(model)

        train_curve.append(train_perf[dataset.eval_metric])

    test_perf = eval(best_model_train, device, test_loader, evaluator)

    print('Finished training!')
    print('Best train score: {}'.format(best_train_perf))
    print('With test score: {}'.format(test_perf))
    print('In epoch: {}'.format(best_train_epoch))

    torch.save(best_model_train.state_dict(), 'trained_models/best_src_model_{}.pth'.format(args.model_num))

if __name__ == "__main__":
    main()
