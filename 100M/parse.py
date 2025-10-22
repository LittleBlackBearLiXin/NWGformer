
from NWGformer import *

def parse_method(args, c, d, device):
    model = MPNNTs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
                  heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln,
                  bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)

    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')

    # model - MPNNs specific parameters
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers for deep methods')
    parser.add_argument('--local_layers', type=int, default=3,
                        help='number of local layers for MPNNs')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads for attention')
    parser.add_argument('--dropout', type=float, default=0.2)

    # MPNNs architecture flags
    parser.add_argument('--pre_ln', action='store_true', default=True,
                        help='use pre layer normalization')
    parser.add_argument('--pre_linear', action='store_true', default=True,
                        help='use pre linear layer')
    parser.add_argument('--res', action='store_true', default=True,
                        help='use residual connections')
    parser.add_argument('--ln', action='store_true', default=False,
                        help='use layer normalization')
    parser.add_argument('--bn', action='store_true', default=True,
                        help='use batch normalization')
    parser.add_argument('--jk', action='store_true', default=False,
                        help='use jump knowledge connections')
    parser.add_argument('--gnn', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage'],
                        help='type of GNN to use in MPNNs')


    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--model_dir', type=str, default='../../model/')

