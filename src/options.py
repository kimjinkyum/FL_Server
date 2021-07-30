import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dcm', type=int, default=1,
                        help="Whether run dcm process")
    parser.add_argument('--data_load', type=int, default=0, help="Whether dataset load or resample (0 : resample 1: "
                                                                 "reload")

    args = parser.parse_args()
    return vars(args)



