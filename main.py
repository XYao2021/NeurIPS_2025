import matplotlib.pyplot as plt
import torch
import random
import copy
import numpy as np
from torch.utils.data import DataLoader
from model import Model
from util import *
from compression import *
from config import *
from dataset import *
from trans_matrix import *
import time
from datetime import date
import os
from algorithms import Algorithms


if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)
    device = 'cuda:{}'.format(CUDA_ID)

if __name__ == '__main__':
    ACC = []
    LOSS = []
    COMM = []
    ALPHAS = []
    MAXES = []
    for seed in Seed_set:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        "other dataset: EMNIST / KMNIST / SVHN"
        if dataset == 'CINIC10':
            train_data, test_data = loading_CINIC(data_path=dataset_path, device=device)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
        elif dataset == 'FashionMNIST' or 'CIFAR10' or 'MNIST' or 'EMNIST' or 'QMNIST' or 'KMNIST':
            train_data, test_data = loading(dataset_name=dataset, data_path=dataset_path, device=device)
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

        print("......DATA LOADING COMPLETE......")
        Sample = Sampling(num_client=CLIENTS, num_class=10, train_data=train_data, method='uniform', seed=seed, name=dataset)
        if DISTRIBUTION == 'Dirichlet':
            if ALPHA == 0:
                client_data = Sample.DL_sampling_single()
            elif ALPHA > 0:
                client_data = Sample.Synthesize_sampling(alpha=ALPHA)
        else:
            raise Exception('This data distribution method has not been embedded')

        client_train_loader = []
        client_residual = []
        client_compressor = []
        client_partition = []
        Models = []
        client_weights = []
        client_tmps = []
        client_accumulate = []
        neighbor_models = []
        neighbors_accumulates = []
        neighbors_estimates = []
        neighbor_updates = []
        estimate_gossip_error = []
        current_weights = []
        m_hat = []

        neighbor_H = []
        neighbor_G = []
        H = []
        G = []

        if ALGORITHM == 'DEFEAT':
            "FashionMNIST"
            max_value = 0.4066
            min_value = -0.2881
            "KMNIST"
            # max_value = 0.2222
            # min_value = -0.2274
        elif ALGORITHM == 'DCD':
            max_value = 0.4038
            min_value = -0.4095
        elif ALGORITHM == 'CHOCO':
            max_value = 0.30123514
            min_value = -0.21583036
        elif ALGORITHM == 'BEER':
            max_value = 3.6578
            min_value = -3.3810
        elif ALGORITHM == 'DeCoM':
            max_value = 2.2271
            min_value = -2.3342
        elif ALGORITHM == 'CEDAS':
            max_value = 0.0525
            min_value = -0.0233
        elif ALGORITHM == 'MoTEF':
            max_value = 1.9098
            min_value = -2.7054

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network=NETWORK)
        check = Check_Matrix(CLIENTS, Transfer.matrix)
        if check != 0:
            raise Exception('The Transfer Matrix Should be Symmetric')
        else:
            print(NETWORK, 'Transfer Matrix is Symmetric Matrix', '\n')
        eigenvalues, Gaps = Transfer.Get_alpha_upper_bound_theory()

        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
        # Preparation for every vector variables
        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_weights.append(model.get_weights())

            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))

            client_residual.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_models.append([model.get_weights() for i in range(len(Transfer.neighbors[n]))])

            neighbor_updates.append([torch.zeros_like(model.get_weights()) for i in range(len(Transfer.neighbors[n]))])

            if ALGORITHM == 'DCD':
                DISCOUNT = 0
            if COMPRESSION == 'quantization':
                if ALGORITHM == 'DEFEAT':
                    if ADAPTIVE:
                        # DISCOUNT = np.sqrt(QUANTIZE_LEVEL)
                        scale = 2 ** QUANTIZE_LEVEL - 1
                        step = (max_value - min_value) / scale
                        normalization = step
                        model_size = len(client_weights[n])
                        print(step, step**2, (step/2)**2, model_size, step**2 * model_size, (step/2)**2 * model_size)
                        normalization = (step/2)**2
                    else:
                        normalization = 1
                if CONTROL is True:
                    client_compressor.append(Lyapunov_compression_Q(node=n, avg_comm_cost=average_comm_cost, V=V, W=W, max_value=max_value, min_value=min_value))
                    client_partition.append(Lyapunov_Participation(node=n, average_comp_cost=average_comp_cost, V=V, W=W, seed=seed))
                else:
                    if FIRST is True:
                        client_compressor.append(Quantization_I(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device, discount=DISCOUNT))
                    else:
                        client_compressor.append(Quantization_U_1(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device, discount=DISCOUNT))  # Unbiased
                        # client_compressor.append(Quantization_U(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device, discount=DISCOUNT))  # Unbiased 1
                        # client_compressor.append(Quantization(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device, discount=DISCOUNT))  # Biased
            elif COMPRESSION == 'topk':
                normalization = 1
                if CONTROL is True:
                    client_compressor.append(Lyapunov_compression_T(node=n, avg_comm_cost=average_comm_cost, V=V, W=W))
                    client_partition.append(Lyapunov_Participation(node=n, average_comp_cost=average_comp_cost, V=V, W=W, seed=seed))
                else:
                    client_compressor.append(Top_k(ratio=RATIO, device=device, discount=DISCOUNT))
            elif COMPRESSION == 'randk':
                normalization = 1
                client_compressor.append(Rand_k(ratio=RATIO, device=device, discount=DISCOUNT))
            else:
                raise Exception('Unknown compression method, please write the compression method first')

            if ALGORITHM == 'CHOCO':
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            if ALGORITHM == 'DeCoM':
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            if ALGORITHM == 'CEDAS':
                client_tmps.append(model.get_weights().to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
                neighbors_accumulates.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            if ALGORITHM == 'MoTEF' or 'MoTEF_VR' or 'DEFEAT':
                # client_tmps.append(model.get_weights().to(device))
                neighbor_H.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                neighbor_G.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                H.append(torch.zeros_like(model.get_weights()).to(device))
                G.append(torch.zeros_like(model.get_weights()).to(device))
                client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
            if ALGORITHM == 'BEER':
                neighbor_H.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                neighbor_G.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
                H.append(torch.zeros_like(model.get_weights()).to(device))
                G.append(torch.zeros_like(model.get_weights()).to(device))
        # print(model.key_list, model.size_list, sum(model.size_list), len(model.size_list))
        Algorithm = Algorithms(name=ALGORITHM, iter_round=ROUND_ITER, device=device, data_transform=data_transform,
                               num_clients=CLIENTS, client_weights=client_weights, client_residuals=client_residual,
                               client_accumulates=client_accumulate, client_compressors=client_compressor,
                               models=Models, data_loaders=client_train_loader, transfer=Transfer,
                               neighbor_models=neighbor_models, neighbors_accumulates=neighbors_accumulates,
                               client_tmps=client_tmps, neighbors_estimates=neighbors_estimates, client_partition=client_partition,
                               control=CONTROL, alpha_max=0, compression_method=COMPRESSION,
                               estimate_gossip_error=estimate_gossip_error, current_weights=current_weights, m_hat=m_hat,
                               adaptive=ADAPTIVE, threshold=THRESHOLD, H=H, neighbor_H=neighbor_H, G=G, neighbor_G=neighbor_G)
        global_loss = []
        Test_acc = []
        iter_num = 0
        print(ALGORITHM, DISCOUNT, BETA, FIRST)

        while True:
            # print('SEED ', '|', seed, '|', 'ITERATION ', iter_num, 'gamma ', DISCOUNT)
            if ALGORITHM == 'DEFEAT':
                Algorithm.DEFEAT(iter_num=iter_num, normalization=normalization)
            elif ALGORITHM == 'DCD':
                if iter_num == 0:
                    print('Algorithm DCD applied')
                Algorithm.DCD(iter_num=iter_num)
            elif ALGORITHM == 'CHOCO':
                if iter_num == 0:
                    print('Algorithm CHOCO applied')
                Algorithm.CHOCO(iter_num=iter_num, consensus=DISCOUNT)  # replace consensus with gamma
            elif ALGORITHM == 'BEER':  # 1
                if iter_num == 0:
                    print('Algorithm BEER applied')
                Algorithm.BEER(iter_num=iter_num, gamma=DISCOUNT, learning_rate=LEARNING_RATE)
            elif ALGORITHM == 'DeCoM':  # MNIST work, fashionMNIST not work
                if iter_num == 0:
                    print('Algorithm DeCoM applied')
                Algorithm.DeCoM(iter_num=iter_num, gamma=DISCOUNT, beta=BETA, learning_rate=LEARNING_RATE)  # gamma = 0.2, beta = 0.05
            elif ALGORITHM == 'CEDAS':  # 1  (Very interesting, It is possible to improve DEFD according to this algorithm)
                if iter_num == 0:
                    print('Algorithm CEDAS applied')
                Algorithm.CEDAS(iter_num=iter_num, gamma=DISCOUNT, alpha=BETA)  # gamma = 0.2 , alpha = 0.05
            elif ALGORITHM == 'MoTEF':  # 1
                if iter_num == 0:
                    print('Algorithm MOTEF applied')
                Algorithm.MoTEF(iter_num=iter_num, gamma=DISCOUNT, learning_rate=LEARNING_RATE, Lambda=BETA)  # 0.05 / 0.01 / 0.1 # gamma = 0.2 , Lambda = 0.05
            elif ALGORITHM == 'MoTEF_VR':  # 0
                if iter_num == 0:
                    print('Algorithm MOTEF_VR applied')
                Algorithm.MOTEF_VR(iter_num=iter_num, gamma=DISCOUNT, learning_rate=LEARNING_RATE, Lambda=BETA)
            else:
                raise Exception('Unknown algorithm, please update the algorithm codes')

            iter_num += 1
            "Need to change the testing model to local model rather than global averaged model"
            if TEST == 'average':
                test_weights = average_weights([Algorithm.models[i].get_weights() for i in range(CLIENTS)])  # test with global averaged model
            elif TEST == 'local':
                test_weights = Algorithm.models[1].get_weights()  # test with local model

            train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader, device=device)

            global_loss.append(train_loss)
            Test_acc.append(test_acc)
            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |',
                  train_acc, '| Test Accuracy |', test_acc, '\n')

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss
                ALPHAS += Algorithm.error_mag
                MAXES += Algorithm.error_ratio
                print([client_compressor[i].discount_parameter for i in range(CLIENTS)])
                break
        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    if STORE == 1:
        if FIRST is True:
            Maxes = []
            Mines = []
            for i in range(CLIENTS):
                Maxes.append(max(client_compressor[i].max))
                Mines.append(min(client_compressor[i].min))
            txt_list = [Maxes, '\n', Mines, '\n', ACC, '\n', LOSS]
        else:
            txt_list = [ACC, '\n', LOSS, '\n', eigenvalues, '\n', Gaps]

        if COMPRESSION == 'quantization':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, QUANTIZE_LEVEL, DISCOUNT, TEST, dataset, LEARNING_RATE, CONSENSUS_STEP, BETA, CLIENTS, NEIGHBORS, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        elif COMPRESSION == 'topk' or 'randk':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, RATIO, DISCOUNT, TEST, dataset, LEARNING_RATE, CONSENSUS_STEP, BETA, CLIENTS, NEIGHBORS, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        else:
            raise Exception('Unknown compression method')

        for item in txt_list:
            f.write("%s\n" % item)
    else:
        print('NOT STORE THE RESULTS THIS TIME')
