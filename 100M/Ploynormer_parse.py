from Polynormer import *

def parse_method(args, c, d, device):
    if args.method == 'Polynormer':
        model = Polynormer(d, args.hidden_channels, c,local_layers=args.local_layers, global_layers=args.globel_layers,
            in_dropout=0.15, dropout=0.3, global_dropout=0.5, heads=1, beta=-1, pre_ln=False).to(device)
    else:
        raise ValueError('Invalid method')
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
    parser.add_argument('--method', type=str, default='Polynormer')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--globel_layers', type=int, default=1,
                        help='number of layers for deep methods')
    parser.add_argument('--local_layers', type=int, default=3,
                        help='number of local layers for MPNNs')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')

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