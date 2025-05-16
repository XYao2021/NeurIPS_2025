import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import time

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'arial'

def smooth(input):

    x1 = (input[0] + input[1] + input[2]) / 3
    x2 = (input[0] + input[1] + input[3]) / 3
    x3 = (input[0] + input[2] + input[3]) / 3
    x4 = (input[1] + input[2] + input[3]) / 3
    output = np.array([x1, x3, x2, x4])

    return output

def moving_average(input_data, window_size):
    moving_average = [[] for i in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                if type(input_data[i][j + 1]) == str:
                    input_data[i][j + 1] = float(input_data[i][j + 1])
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
            else:
                moving_average[i].append(sum(input_data[i][j - window_size + 1:j + 1]) / len(input_data[i][j - window_size + 1:j + 1]))
    moving_average_means = []
    for i in range(len(moving_average[0])):
        sum_data = []
        for j in range(len(moving_average)):
            sum_data.append(moving_average[j][i])
        moving_average_means.append(sum(sum_data) / len(sum_data))
    return np.array(moving_average), moving_average_means

def data_to_fig(file, ax, color_list, mission, iteration, times, agg, window_size):
    for i in range(len(file)):
        x = pd.read_csv(file[i], header=None)
        x_acc, x_loss = x.values
        x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in
                                                                                range(times)]
        if mission == 'acc':
            x_area = np.stack(x_acc)
            ax.set_ylabel('Test Accuracy', fontsize=14)
        elif mission == 'loss':
            x_area = np.stack(x_loss)
            ax.set_ylabel('Global Loss', fontsize=14)

        x_area, x_means = moving_average(input_data=x_area, window_size=window_size)
        x_stds = x_area.std(axis=0, ddof=1)

        name = file[i].split('|')[0]
        ratio = file[i].split('|')[2]

        if name == 'CHOCO':
            name = 'CHOCO'
        elif name == 'DEFEAT':
            name = 'DEFEAT (ours)'
        elif name == 'DCD':
            name = 'DCD'
        elif name == 'MoTEF':
            name = 'MoTEF'
        elif name == 'BEER':
            name = 'BEER'

        ax.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
        ax.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=0.1, color=color_list[i])

        if mission == 'acc':
            print(name, ratio, x_means[-1], x_stds[-1], '\n')

        ax.set_xlabel('Aggregations', fontsize=14)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


"FashionMNIST Ring 0.05"

plot_list = []  # Add the simulation files here.

color_list = ['blue', 'orange', 'green', 'red', 'gray', 'purple', 'brown', 'pink', 'yellow', 'cyan', 'olive', 'black']

# times = 4
times = 3

agg = 1000
# agg = 20000

# compare = True
compare = False

# Ada_vs_fixed = True
Ada_vs_fixed = False

Fair = 1  # Fair comparison
# Fair = 0  # comparison in terms of number of aggregations

alpha1 = 0.05

iteration = range(agg)
if agg == 1000:
    dataset = "fashion"
elif agg == 20000:
    dataset = "CIFAR"

missions = ['acc', 'loss']


if agg == 1000:
    window_size = 20
elif agg == 20000:
    window_size = 250

fig, axs = plt.subplots(figsize=(12, 6), nrows=2, ncols=3)  # wider, shorter figure
plt.subplots_adjust(hspace=0.3)

for j in range(len(plot_list)):
    for mission in missions:
        index = int(missions.index(mission))
        x_means_max = 0
        folder = plot_list[j]
        for i in range(len(folder)):
            file = folder[i]
            name = file.split('|')[0]
            alpha = file.split('|')[1]
            compression = file.split('|')[2]
            nodes = file.split('|')[-3]

            if compression == '4':
                method = 'Quantization'
            elif compression == '6':
                method = 'Quantization'
            elif compression == '0.05':
                method = 'Top-k'
            elif compression == '0.1':
                method = 'Top-k'

            x = pd.read_csv(file, header=None)

            x_acc, x_loss = x.values

            x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)]
            if len(x_acc) > times:
                x_acc, x_loss = x_acc[1:], x_loss[1:]

            if mission == 'acc':
                x_area = np.stack(x_acc)
            elif mission == 'loss':
                x_area = np.stack(x_loss)

            x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

            x_means = x_area.mean(axis=0)
            x_stds = x_area.std(axis=0, ddof=1)

            if name == 'CHOCO':
                name = 'CHOCO'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif name == 'DCD':
                name = 'DCD'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif name == 'CEDAS':
                name = 'CEDAS'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif name == 'BEER':
                if Fair == 1:
                    name = 'BEER'
                    y = int(len(x_means))
                    x_means = x_means[:int(y / 2)]
                    x_stds = x_stds[:int(y / 2)]
                    x_b = np.arange(0, y, 2)
                    axs[index][j].plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
                elif Fair == 0:
                    name = 'BEER'
                    axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif name == 'MoTEF':
                if Fair == 1:
                    name = 'MoTEF'
                    x_means = x_means[:int(agg / 2)]
                    x_stds = x_stds[:int(agg / 2)]
                    x_b = np.arange(0, agg, 2)
                    axs[index][j].plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
                elif Fair == 0:
                    name = 'MoTEF'
                    axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                    axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif name == 'DEFEAT':
                name = 'DEFEAT (ours)'
                axs[index][j].plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                axs[index][j].fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])

            if mission == 'acc':
                print(j, name, sum(x_means[-20:])/len(x_means[-20:]), sum(x_stds[-20:])/len(x_stds[-20:]))

        if index == 1:
            axs[index][j].set_xlabel('Number of Iterations', fontsize=12)
        if j == 0:
            if index == 0:
                axs[index][j].set_ylabel('Test Accuracy', fontsize=12)
            else:
                axs[index][j].set_ylabel('Training Loss', fontsize=12)
        axs[index][j].set_title('Ring Network with {} nodes'.format(nodes))
        if mission == 'acc':
            if dataset == 'CIFAR':
                axs[index][j].set_ylim([0.3, 0.72])
            elif dataset == 'fashion':
                axs[index][j].set_ylim([0.65, 0.82])

        elif mission == 'loss':
            if dataset == 'CIFAR':
                axs[index][j].set_ylim([0.005, 0.014])
            elif dataset == 'fashion':
                axs[index][j].set_ylim([0.003, 0.008])
        axs[index][j].grid()

axs[0][1].legend(bbox_to_anchor=(1.8, 1.35), loc='upper right', ncol=6)
plt.savefig('./Plots/Ring_0.1.pdf')
plt.show()
