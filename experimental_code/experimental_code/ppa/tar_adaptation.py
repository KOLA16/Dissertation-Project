import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from gnn import GNN

from tqdm import tqdm
import argparse
import numpy as np
import copy

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import lib_patches
from domain_adaptor import AaD
from utils import plot_tsne


def map_index(loader):
    # Map original dataset index to index in memory bank
    indx_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_data = batch[0]
        batch_idx = batch[1]

        indx_list += batch_idx.tolist()

    val = list(range(0, len(indx_list)))
    indx_map = dict(zip(indx_list, val))

    return indx_map


def init_memory_banks(model, device, loader, aad, indx_map, args):
    model.eval()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_data = batch[0]
        batch_idx = batch[1]
        batch_idx_list = batch_idx.tolist()
        memory_bank_idx = list(map(indx_map.get, batch_idx_list))
        batch_data = batch_data.to(device)

        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            with torch.no_grad():
                pred, graph_embed = model(batch_data)
                softmax_score = torch.nn.Softmax(dim=1)(pred)

            aad.init_memory_banks_step(memory_bank_idx, graph_embed, softmax_score, batch_data.y.to(torch.float32))

    torch.save(aad.feature_bank, 'saved_features/{}_tar_features.pt'.format(args.model_num))
    torch.save(aad.true_labels_bank, 'saved_features/{}_tar_labels.pt'.format(args.model_num))
    torch.save(aad.score_bank, 'saved_features/{}_tar_softmax.pt'.format(args.model_num))
    # plot_tsne(filename='{}_src'.format(args.model_num), perplexity=50.0, early_exaggeration=12.0)


def aad_alpha_decay(iter_num, max_iter, alpha, beta):
    p = (iter_num / max_iter)
    decay_fact = (1 + 10 * p) ** (-beta)
    return alpha * decay_fact


def train(model, device, loader, optimizer, lr_scheduler, aad, indx_map, epoch, max_iter, args):
    model.train()
    loss_sum, first_term_sum, second_term_sum = 0, 0, 0

    curr_iter = 0 + (epoch - 1) * len(loader)
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_data = batch[0]
        batch_idx = batch[1]
        batch_idx_list = batch_idx.tolist()
        memory_bank_idx = list(map(indx_map.get, batch_idx_list))

        batch_data = batch_data.to(device)

        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            curr_iter += 1
            pred, graph_embed = model(batch_data)
            softmax_score = torch.nn.Softmax(dim=1)(pred)
            optimizer.zero_grad()
            loss, first_term, second_term = aad.adaptation_step(memory_bank_idx, graph_embed, softmax_score)
            aad.alpha = aad_alpha_decay(curr_iter, max_iter, args.alpha, args.beta)
            loss_sum += loss.item()
            first_term_sum += first_term
            second_term_sum += second_term
            loss.backward()
            optimizer.step()

    # lr_scheduler.step()
    print('  loss sum: {}'.format(loss_sum))
    print('  first term sum: {}'.format(first_term_sum))
    print('  second term sum: {}'.format(second_term_sum))


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
    parser.add_argument('--epochs', type=int, default=15,
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
                        help='which source model to use')

    parser.add_argument('--k', type=int, default=7,
                        help='')
    parser.add_argument('--alpha', type=int, default=1,
                        help='')
    parser.add_argument('--beta', type=int, default=5,
                        help='')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)

    split_idx = dataset.get_idx_split(split_type=args.split)
    target_size = split_idx["train_tar"].size(dim=0)
    aad = AaD(k=args.k, alpha=args.alpha, beta=args.beta, tar_size=target_size)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train_tar"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, virtual_node=True).to(device)

    # load a pre-trained source model
    model.load_state_dict(torch.load('trained_models/best_src_model_{}.pth'.format(args.model_num)))

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn_node.parameters(), "lr": 0.00001})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": 0.0001})
    optimizer = optim.Adam(model_param_group, weight_decay=0)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    max_iter = args.epochs * len(train_loader)

    best_adapt_perf = 0
    best_model_adapt = None

    print("=====Epoch 0")
    print('Initialising memory banks...')
    indx_map = map_index(train_loader)
    init_memory_banks(model, device, train_loader, aad, indx_map, args)
    print('Evaluating...')
    train_perf = eval(model, device, train_loader, evaluator)
    print({'Train (before adaptation)': train_perf})
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, lr_scheduler, aad, indx_map, epoch, max_iter, args)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)

        print({'Train': train_perf})

        if train_perf[dataset.eval_metric] > best_adapt_perf:
            best_adapt_perf = train_perf[dataset.eval_metric]
            best_adapt_epoch = epoch
            best_model_adapt = copy.deepcopy(model)

    print('Finished training!')
    print('Best train score: {}'.format(best_adapt_perf))
    print('In epoch: {}'.format(best_adapt_epoch))

    torch.save(best_model_adapt.state_dict(), 'trained_models/best_adapted_model_{}.pth'.format(args.model_num))


if __name__ == "__main__":
    main()
