import argparse
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets

parse = argparse.ArgumentParser()
parse.add_argument('-data', type=str, default='fashion', help='choose target dataset')  # Like CIFAR10, MNIST, FashionMNIST, CIFAR100
parse.add_argument('-pretrained-model', type=str, default='', help='pretrain model path')
parse.add_argument('-num_iter', type=int, default=100, help='number of iterations')

parse.add_argument('-agg', type=int, default=500, help='Global Aggregation times/ total iterations')
parse.add_argument('-iter_round', type=int, default=1, help='Local Training Times: Iterations')
parse.add_argument('-acc_point', type=int, default=1, help='Accuracy check point')

parse.add_argument('-lr', type=float, default=0.05, help='Learning Rate of the Model')
parse.add_argument('-bs', type=int, default=16, help='Batch Size for model')
parse.add_argument('-bs_test', type=int, default=128, help='Batch Size for test model')
parse.add_argument('-cn', type=int, default=20, help='Client Number')
parse.add_argument('-nn', type=int, default=2, help='Number of Neighbors')

parse.add_argument('-V', type=float, default=0.02, help='Lyapunov V value, constant weight to ensure the average of p(t) close to optimal ')
parse.add_argument('-W', type=float, default=1.0, help='Lyapunov W value, initial queue length')
parse.add_argument('-avg_comm', type=float, default=0.01, help='Average communication cost')
parse.add_argument('-avg_comp', type=float, default=0.25, help='Average computation cost')

parse.add_argument('-seed', type=int, default=13, help='random seed for pseudo random model initial weights')
parse.add_argument('-ns', type=int, default=4, help='Number of seeds for simulation')
parse.add_argument('-ratio', type=float, default=0.1, help='the ratio of non-zero elements that the baseline want to transfer')
parse.add_argument('-quan', type=int, default=8, help='Quantization bits')
parse.add_argument('-dist', type=str, default='Dirichlet', help='Data Distribution Method')
parse.add_argument('-alpha', type=float, default=0.05, help='Alpha value for Dirichlet Distribution')

parse.add_argument('-consensus', type=float, default=0.05, help='Consensus step for CHOCO')
parse.add_argument('-gamma', type=float, default=1.0, help='Discount parameter of residual error for biased estimator')
parse.add_argument('-threshold', type=float, default=1, help='Adapt threshold')
parse.add_argument('-beta', type=float, default=0.9, help='momentum parameter')

parse.add_argument('-algorithm', type=str, default='DEFEAT', help='machine learning algorithm')
parse.add_argument('-control', type=int, default=0, help='Apply control algorithm or not, 0 is not, 1 is true')
parse.add_argument('-compression', type=str, default='quantization', help='compression method')
parse.add_argument('-network', type=str, default='random', help='Network Topology')
parse.add_argument('-test', type=str, default='average', help='test model: average / local')

parse.add_argument('-store', type=int, default=1, help='Store the results or not (1 or 0)')
parse.add_argument('-id', type=int, default=1, help='cuda id (0, 1, 2, 3)')
parse.add_argument('-first_time', type=bool, default=False, help='Is this the first time run the quantization method')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parse.parse_args()
print(', '.join(f'{k}={v}' for k, v in vars(args).items()))

CUDA_ID = args.id
average_comm_cost = args.avg_comm
average_comp_cost = args.avg_comp
V = args.V  # Lyapunov V value
W = args.W  # Lyapunov initial queue length W
# LEARNING_RATE = args.lr

RATIO = args.ratio
CONSENSUS_STEP = args.consensus
QUANTIZE_LEVEL = args.quan

dataset_path = os.path.join(os.path.dirname(__file__), 'data')

if args.data == 'fashion':
    model_name = 'FashionMNIST'
    dataset = 'FashionMNIST'
elif args.data == 'MNIST':
    model_name = 'MNIST'
    dataset = 'MNIST'
elif args.data == 'EMNIST':
    model_name = 'EMNIST'
    dataset = 'EMNIST'
elif args.data == 'QMNIST':
    model_name = 'QMNIST'
    dataset = 'QMNIST'
elif args.data == 'KMNIST':
    model_name = 'KMNIST'
    dataset = 'KMNIST'

elif args.data == 'SVHN':
    model_name = 'SVHN'
    dataset = 'SVHN'

elif args.data == 'CIFAR10':
    model_name = 'CIFAR10Model'
    dataset = 'CIFAR10'
elif args.data == 'CINIC10':
    model_name = 'CIFAR10Model'
    dataset = 'CINIC10'
else:
    raise Exception('Unknown dataset, need to update')

if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10' or dataset == 'CIFAR100':
    data_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip()])
else:
    data_transform = None

Seed_number = args.ns
Seed_up = args.seed
Seed_set = [i for i in range(Seed_up-Seed_number, Seed_up)]

AGGREGATION = args.agg  # This time aggregation equal to total iteration for 1-SGD
BATCH_SIZE = args.bs
BATCH_SIZE_TEST = args.bs_test
CLIENTS = args.cn
NEIGHBORS = args.nn
ROUND_ITER = args.iter_round
CHECK = args.acc_point
ALPHA = args.alpha
DISTRIBUTION = args.dist

ALGORITHM = args.algorithm
COMPRESSION = args.compression
NETWORK = args.network
DISCOUNT = args.gamma
STORE = args.store
FIRST = args.first_time
THRESHOLD = args.threshold

BETA = args.beta
TEST = args.test


if args.control == 0:
    CONTROL = False
elif args.control == 1:
    CONTROL = True
else:
    raise Exception('Unknown control parameter')

LEARNING_RATE = args.lr
