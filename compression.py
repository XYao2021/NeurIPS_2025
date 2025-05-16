import copy
import random
from abc import ABC
import numpy.random
import torch
import numpy as np
import abc
from config import *


class Top_k(abc.ABC):
    def __init__(self, ratio=1.0, device=None, discount=0.0):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual  # w_tmp + gamma*e_t

        full_size = w_tmp.size()[0]
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)
        trans_bits = int(self.ratio * full_size)

        trans_indices, not_trans_indices = bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:]
        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp
        return w_tmp, w_tmp_residual

class Quantization(abc.ABC):  # Biased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, device=None, discount=0.0):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))
        assignments = torch.argmin(distances, dim=1)

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_U(abc.ABC):  # Unbiased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, discount=0.0, device=None):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        # print(self.discount_factor)
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))

        sorted_distance_value = torch.sort(distances, dim=1).values
        sorted_distances_index = torch.argsort(distances, dim=1)

        first_choice_value = torch.flatten(sorted_distance_value[:, :1]).tolist()
        first_choice_index = torch.flatten(sorted_distances_index[:, :1])

        second_choice_index = torch.flatten(sorted_distances_index[:, 1:2])
        sorted_distance_value = sorted_distance_value[:, :2]

        summation = torch.sum(sorted_distance_value, dim=1).tolist()
        random_choice = np.random.uniform(high=np.array(summation))

        decision = random_choice > np.array(first_choice_value)

        assignments = copy.deepcopy(second_choice_index)
        indexes = torch.tensor(np.where(decision)[0])
        assignments[indexes] = first_choice_index[indexes]

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_I(abc.ABC):  # Initialize setup for max and min value
    def __init__(self, num_bits=4, max_value=0, min_value=0, device=None, discount=0.0):
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.max = []
        self.min = []
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual
        max_value = torch.max(w_tmp)
        min_value = torch.min(w_tmp)
        self.max.append(max_value)
        self.min.append(min_value)
        # print(max_value, min_value)

        step = (max_value - min_value) / self.scale

        centroids = []
        value = min_value
        centroids.append(value)
        while len(centroids) < 2 ** self.num_bits:
            value = value + step
            centroids.append(value)

        centroids = torch.tensor(centroids).to(device)
        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
        assignments = torch.argmin(distances, dim=1)

        w_tmp_quantized = torch.tensor([centroids[i] for i in assignments])
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Quantization_U_1(abc.ABC):  # Unbiased quantization
    def __init__(self, num_bits=8, max_value=0, min_value=0, discount=0.0, device=None):
        self.device = device
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        self.discount_parameter = discount
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization).to(self.device)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        # print(iter, self.discount_parameter)
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))

        sorted_distance_value = torch.sort(distances, dim=1).values
        sorted_distances_index = torch.argsort(distances, dim=1)

        first_choice_value = torch.flatten(sorted_distance_value[:, :1])
        first_choice_index = torch.flatten(sorted_distances_index[:, :1])

        second_choice_value = torch.flatten(sorted_distance_value[:, 1:2])
        second_choice_index = torch.flatten(sorted_distances_index[:, 1:2])

        # print(first_choice_value, second_choice_value)
        summation_1 = first_choice_value + second_choice_value

        choices_2 = second_choice_value / summation_1
        choices_2 = torch.bernoulli(choices_2)

        assignments = copy.deepcopy(first_choice_index)
        assignments[torch.where(choices_2 > 0)[0]] = second_choice_index[torch.where(choices_2 > 0)[0]]

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Rand_k(abc.ABC):
    def __init__(self, ratio=1.0, device=None, discount=0.0):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.discount_parameter = discount

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, neighbors):
        if w_tmp is None:
            w_tmp = self.discount_parameter * w_residual  # w_residual is e_t
        else:
            w_tmp += self.discount_parameter * w_residual

        full_size = w_tmp.size()[0]
        # print(iter, full_size, self.ratio)
        trans_bits = int(self.ratio * full_size)
        indices = random.sample(range(full_size), trans_bits)

        w_trans = torch.zeros_like(w_tmp)
        w_tmp_residual = copy.deepcopy(w_tmp)

        w_trans[indices] = w_tmp[indices]
        w_tmp = w_trans  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp
        return w_tmp, w_tmp_residual
