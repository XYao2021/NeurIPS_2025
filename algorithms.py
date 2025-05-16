import copy
import numpy as np
import torch
import random
import copy
import time

class Algorithms:
    def __init__(self, name=None, iter_round=None, device=None,
                 data_transform=None, num_clients=None, client_weights=None,
                 client_residuals=None, client_accumulates=None, client_compressors=None,
                 models=None, data_loaders=None, transfer=None, neighbor_models=None,
                 neighbors_accumulates=None, client_tmps=None, neighbors_estimates=None,
                 client_partition=None, control=False, alpha_max=None, compression_method=None,
                 estimate_gossip_error=None, current_weights=None, m_hat=None, adaptive=False, threshold=None,
                 H=None, G=None, neighbor_H=None, neighbor_G=None):
        super().__init__()
        self.algorithm_name = name
        self.local_iter = iter_round
        self.device = device
        self.data_transform = data_transform
        self.num_clients = num_clients

        self.client_weights = client_weights
        self.client_residuals = client_residuals
        self.client_residuals_g = copy.deepcopy(client_residuals)
        self.client_compressor = client_compressors
        self.models = models
        self.data_loaders = data_loaders
        self.transfer = transfer
        self.neighbors = self.transfer.neighbors
        self.neighbor_models = neighbor_models
        self.client_partition = client_partition
        self.client_history = self.client_residuals
        self.initial_error_norm = None

        "CHOCO"
        self.client_accumulates = client_accumulates
        self.neighbor_accumulates = neighbors_accumulates
        self.client_tmps = client_tmps
        self.neighbors_estimates = neighbors_estimates

        "BEER"
        self.neighbor_H = neighbor_H
        self.neighbor_G = neighbor_G
        self.H = H
        self.G = G
        self.V = []
        self.neighbor_V = copy.deepcopy(neighbor_G)
        self.previous_gradients = []

        "DeCoM"
        self.gradients = []
        self.gradients_tmp = []
        self.client_theta_hat = client_weights  # initial is model weights
        self.neighbors_theta_hat = neighbor_models  # initials are model weights
        self.client_g_hat = client_accumulates  # initials are zero
        self.neighbors_g_hat = neighbors_accumulates  # initials are zeros
        self.v = []  # gradient estimate
        self.previous_V = []
        self.previous_X = client_weights

        "CEDAS"
        self.diffusion = client_accumulates  # zeros
        self.h = []  # initial model weights
        self.hw = []  # h_omega
        self.updates = neighbors_accumulates

        "MOTEF"
        self.M = []
        self.previous_M = client_accumulates

        "Testing parameter"
        self.Alpha = []
        self.alpha_max = alpha_max
        self.compression_method = compression_method
        self.changes_ratio = []
        "Debugging parameters"
        self.max = []
        self.change_iter_num = []
        self.error_mag = []
        self.error_ratio = []
        self.logger()
        self.gamma = 1
        # self.coefficient = torch.ones_like(self.models[0])

    def logger(self):
        print(' compression method:', self.compression_method, '\n',
              'running algorithm: ', self.algorithm_name, '\n')

    def _training(self, data_loader, client_weights, model):
        model.assign_weights(weights=client_weights)
        model.model.train()
        for i in range(self.local_iter):
            images, labels = data_loader
            images, labels = images.to(self.device), labels.to(self.device)
            if self.data_transform is not None:
                images = self.data_transform(images)

            model.optimizer.zero_grad()
            pred = model.model(images)
            loss = model.loss_function(pred, labels)
            loss.backward()
            model.optimizer.step()

        trained_model = model.get_weights()  # x_t - \eta * gradients
        return trained_model

    def _average_updates(self, updates):
        Averaged_weights = []
        for i in range(self.num_clients):
            Averaged_weights.append(sum(updates[i]) / len(updates[i]))
        return Averaged_weights

    def _check_weights(self, client_weights, neighbors_weights):
        checks = 0
        for n in range(self.num_clients):
            neighbors = self.neighbors[n]
            neighbors_models = neighbors_weights[n]

            check = 0
            for m in range(len(neighbors)):
                if torch.equal(neighbors_models[m], client_weights[neighbors[m]]):
                    check += 1
                else:
                    pass
            if check == len(self.neighbors[n]):
                checks += 1
            else:
                pass
        if checks == self.num_clients:
            return True
        else:
            return False

    "New idea without gradient tracking and momentum"
    def DEFEAT(self, iter_num, normalization):
        Averaged_weights = self._average_updates(updates=self.neighbor_models)

        learning_rate = self.models[0].learning_rate
        error_ratio_i = []
        epsilon = 0.000000000001  # noise: make sure not divide or multiple with zero.

        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            Vector_update = self._training(data_loader=[images, labels],
                                           client_weights=self.client_weights[n], model=self.models[n])
            Vector_update -= self.client_weights[n]  # -eta*G(X_t)

            gradient = Vector_update
            gradient_norm = torch.sum(torch.square(Vector_update)).item()
            gradient_and_error_norm = torch.sum(torch.square(Vector_update + self.client_residuals[n])).item()

            Vector_update += Averaged_weights[n]  # X_tW - eta*G(X_t)
            Vector_update -= self.client_weights[n]  # X_t(W-I) - eta*G(X_t)

            residual_errors = (1 - self.client_compressor[n].discount_parameter) * self.client_residuals[n]  # Equals to zero if gamma equals to 1.0

            "Compression Operator"  # Vector_update = v_t = C(b_t) | client_residual = e_(t+1) = b_t - v_t
            Vector_update, self.client_residuals[n] = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=Vector_update, w_residual=self.client_residuals[n], device=self.device, neighbors=self.neighbors[n])

            self.client_residuals[n] += residual_errors  # e_(t+1) = b_t - v_t + (1-gamma)*e_t

            self.client_weights[n] += Vector_update  # x_(t+1) = x_t + v_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update

    def DCD(self, iter_num):
        Averaged_weights = self._average_updates(updates=self.neighbor_models)

        for n in range(self.num_clients):
            if self.control:
                pass
            else:
                images, labels = next(iter(self.data_loaders[n]))
                Vector_update = self._training(data_loader=[images, labels],
                                               client_weights=self.client_weights[n],
                                               model=self.models[n])
                Vector_update -= self.client_weights[n]  # gradient
                Vector_update += Averaged_weights[n]

            Vector_update -= self.client_weights[n]  # Difference between averaged weights and local weights

            Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(iter=iter_num,
                                                                                     w_tmp=Vector_update,
                                                                                     w_residual=
                                                                                     self.client_residuals[n],
                                                                                     device=self.device,
                                                                                     neighbors=self.neighbors[n])
            self.client_weights[n] += Vector_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_models[m][self.neighbors[m].index(n)] += Vector_update

    def _averaged_choco(self, updates, update):
        Averaged = []
        for i in range(self.num_clients):
            summation = torch.zeros_like(update[0])
            for j in range(len(updates[i])):
                summation += (1/len(updates[i])) * (updates[i][j] - update[i])
            Averaged.append(summation)
        return Averaged

    def CHOCO(self, iter_num, consensus):
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            self.client_tmps[n] = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])

            Vector_update = self.client_weights[n] - self.client_accumulates[n]
            Vector_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, iter=iter_num,
                                                                                     w_residual=self.client_residuals[n],
                                                                                     device=self.device, neighbors=self.neighbors[n])
            self.client_accumulates[n] += Vector_update  # Vector Update is q_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_accumulates[m][self.neighbors[m].index(n)] += Vector_update

        Averaged_accumulate = self._averaged_choco(updates=self.neighbor_accumulates, update=self.client_accumulates)

        for n in range(self.num_clients):
            self.client_weights[n] = self.client_tmps[n] + consensus * Averaged_accumulate[n]

    def BEER(self, iter_num, gamma, learning_rate):
        weighted_H = self._averaged_choco(updates=self.neighbor_H, update=self.H)
        weighted_G = self._averaged_choco(updates=self.neighbor_G, update=self.G)

        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                # initial_gradients = self.client_weights[n] - training_weights
                self.V.append(initial_gradients)
                self.previous_gradients.append(initial_gradients)

            self.client_weights[n] = self.client_weights[n] + gamma * weighted_H[n] - learning_rate * self.V[n]
            H_update = self.client_weights[n] - self.H[n]
            H_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=H_update, iter=iter_num,
                                                                                     w_residual=self.client_residuals[n],
                                                                                     device=self.device,
                                                                                     neighbors=self.neighbors[n])
            self.H[n] += H_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += H_update

            images, labels = next(iter(self.data_loaders[n]))
            next_train_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
            next_gradients = (self.client_weights[n] - next_train_weights) / learning_rate
            # next_gradients = self.client_weights[n] - next_train_weights

            self.V[n] = self.V[n] + gamma * weighted_G[n] + next_gradients - self.previous_gradients[n]
            self.previous_gradients[n] = next_gradients

            G_update = self.V[n] - self.G[n]
            G_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=G_update, iter=iter_num,
                                                                                w_residual=self.client_residuals[n],
                                                                                device=self.device,
                                                                                neighbors=self.neighbors[n])
            self.G[n] += G_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += G_update

    def DeCoM(self, iter_num, gamma, learning_rate, beta):  # Have problem with Quantization compression
        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                                  model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                self.v.append(initial_gradients)
                self.previous_V.append(initial_gradients)
                self.gradients.append(initial_gradients)
                self.gradients_tmp.append(initial_gradients)

            'client_weights --- theta'
            self.client_tmps[n] = self.client_weights[n] - learning_rate * self.gradients[n]
            theta_update = self.client_tmps[n] - self.client_theta_hat[n]
            theta_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=theta_update, iter=iter_num,
                                                                                    w_residual=self.client_residuals[n],
                                                                                    device=self.device,
                                                                                    neighbors=self.neighbors[n])
            # theta_update = random_quantize_mat(theta_update, s=s)[0]
            self.client_theta_hat[n] += theta_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbors_theta_hat[m][self.neighbors[m].index(n)] += theta_update

        weighted_theta = self._averaged_choco(updates=self.neighbors_theta_hat, update=self.client_theta_hat)
        for n in range(self.num_clients):
            # next_dataloader = copy.deepcopy(self.data_loaders[n])
            images, labels = next(iter(self.data_loaders[n]))
            f_hat_current = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                           model=self.models[n])
            f_hat_current = (self.client_weights[n] - f_hat_current) / learning_rate

            self.client_weights[n] = self.client_tmps[n] + gamma * weighted_theta[n]
            f_hat_next = self._training(data_loader=[images, labels], client_weights=self.client_weights[n],
                                        model=self.models[n])
            f_hat_next = (self.client_weights[n] - f_hat_next) / learning_rate

            self.v[n] = beta * f_hat_next + (1 - beta) * (self.v[n] + f_hat_next - f_hat_current)
            # self.v[n] = f_hat_next + (1 - beta) * (self.v[n] - f_hat_current)

            self.gradients_tmp[n] = self.gradients[n] + self.v[n] - self.previous_V[n]
            self.previous_V[n] = self.v[n]

            g_update = self.gradients_tmp[n] - self.client_g_hat[n]
            g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=g_update, iter=iter_num,
                                                                                w_residual=self.client_residuals[n],
                                                                                device=self.device,
                                                                                neighbors=self.neighbors[n])
            # g_update = random_quantize_mat(theta_update, s=s)[0]
            self.client_g_hat[n] += g_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbors_g_hat[m][self.neighbors[m].index(n)] += g_update

        weighted_g = self._averaged_choco(updates=self.neighbors_g_hat, update=self.client_g_hat)
        for n in range(self.num_clients):
            self.gradients[n] = self.gradients_tmp[n] + gamma * weighted_g[n]

    def CEDAS(self, iter_num, alpha, gamma):
        Trained_weights = []
        Y_hat_plus = []
        for n in range(self.num_clients):
            if iter_num == 0:
                self.h.append(self.client_weights[n])
                self.hw.append(self.client_weights[n])
                images, labels = next(iter(self.data_loaders[n]))
                self.client_weights[n] = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])

            Trained_weights.append(trained_weights)
            y = trained_weights - self.diffusion[n]
            "COMM start"
            q = y - self.h[n]
            q, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=q, iter=iter_num,
                                                                         w_residual=self.client_residuals[n],
                                                                         device=self.device,
                                                                         neighbors=self.neighbors[n])
            y_hat_plus = self.h[n] + q
            Y_hat_plus.append(y_hat_plus)
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.updates[m][self.neighbors[m].index(n)] = q

        Averaged_updates = self._average_updates(updates=self.updates)

        for n in range(self.num_clients):
            yw_hat_plus = self.hw[n] + Averaged_updates[n]

            self.h[n] = (1-alpha) * self.h[n] + alpha * Y_hat_plus[n]
            self.hw[n] = (1-alpha) * self.hw[n] + alpha * yw_hat_plus  # here
            "COMM end"

            self.diffusion[n] += (gamma / 2) * (Y_hat_plus[n] - yw_hat_plus)
            self.client_weights[n] = Trained_weights[n] - self.diffusion[n]
            # self.client_weights[n] = Trained_weights[n]

    "MOTEF and MOTEF_VR require large batch size? No"
    def MoTEF(self, iter_num, gamma, learning_rate, Lambda):  # Binary classification?
        weighted_H = self._averaged_choco(updates=self.neighbor_H, update=self.H)
        weighted_G = self._averaged_choco(updates=self.neighbor_G, update=self.G)
        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                training_weights = self._training(data_loader=[images, labels], client_weights=self.client_weights[n], model=self.models[n])
                initial_gradients = (self.client_weights[n] - training_weights) / learning_rate
                # initial_gradients = self.client_weights[n] - training_weights
                self.V.append(initial_gradients)
                self.M.append(initial_gradients)
                # self.M.append(torch.zeros_like(initial_gradients))

            self.client_weights[n] += gamma * weighted_H[n] - learning_rate * self.V[n]
            Q_h_update = self.client_weights[n] - self.H[n]
            Q_h_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_h_update, iter=iter_num,
                                                                                 w_residual=self.client_residuals[n],
                                                                                 device=self.device,
                                                                                 neighbors=self.neighbors[n])
            self.H[n] += Q_h_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += Q_h_update

            # print(self.previous_M[n], self.M)
            self.previous_M[n] = copy.deepcopy(self.M[n])

            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.client_weights[n],
                                             model=self.models[n])
            gradients = (self.client_weights[n] - trained_weights) / learning_rate

            self.M[n] = (1 - Lambda) * self.M[n] + Lambda * gradients
            self.V[n] += gamma * weighted_G[n] + self.M[n] - self.previous_M[n]

            Q_g_update = self.V[n] - self.G[n]
            Q_g_update, _ = self.client_compressor[n].get_trans_bits_and_residual(w_tmp=Q_g_update, iter=iter_num,
                                                                                  w_residual=self.client_residuals[n],
                                                                                  device=self.device,
                                                                                  neighbors=self.neighbors[n])

            self.G[n] += Q_g_update
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += Q_g_update
