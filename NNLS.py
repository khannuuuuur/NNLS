###########################################
#         ECE 174 MINI PROJECT 2
###########################################
# AUTHOR: Conner Hsu
# PID: A16665092
###########################################
import os.path
from pathlib import Path
import sys

import numpy as np
import math

import matplotlib.pyplot as plt
import time

def save_np(path_name, arr):
    if '/' not in path_name:
        np.save(path_name, arr)
        return
    last_slash = path_name.rindex('/')
    path = path_name[0:last_slash+1]
    name = path_name[last_slash:]
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path_name, arr)
def list_latex(list):
    mean = np.average(list)
    result = ''
    for n in list:
        if n >= 1000:
            result += '{:.2e}'.format(n)
        else:
            result += '{:.2f}'.format(n)
        result += ' & '
    if mean >= 1000:
        result += '{:.2e}'.format(mean)
    else:
        result += '{:.2f}'.format(mean)
    return result  + ' \\\\'
def precision_truncate(list, order=1):
    list2 = (np.array(list)*order).astype('int64').astype('float64')/order
    end = np.where(list2==list2[-1])[0][0]
    return list[:end]

# Creates N sample input and output data points for g where g maps from R^3 to R
def get_data(g, N, name, range=1, noise=0, new=False, save=True):
    filename = name + str(N) + '.npy'

    if not new and Path(filename).is_file():
        dataX = np.load(filename)
        print('Loaded', filename)
    else:
        dataX = np.random.uniform(-1, 1, size=(N, 3))
        if save:
            save_np(filename, dataX)
    dataX = dataX*range

    dataY = np.apply_along_axis(g, 1, dataX).T
    if noise != 0:
        filename_noise = name + '_noise' + str(N) + '.npy'
        if not new and Path(filename_noise).is_file():
            dataNoise = np.load(filename_noise)
        else:
            dataNoise = np.random.uniform(-1, 1, size=dataY.shape)
            if save:
                save_np(filename_noise, dataNoise)
        dataY += dataNoise*noise

    return dataX, dataY

class Trainer:
    phi = lambda x: np.tanh(x)
    phidot = lambda x: 1-np.square(np.tanh(x))
    def add_bias(A):
        return np.append(A, np.ones((A.shape[0], 1)), axis=1)
    def get_rms_error(w, dataX, dataY):
        N = dataX.shape[0]
        return math.sqrt(1/N*np.linalg.norm(Trainer.f(w, dataX)-dataY)**2)
    def f(w, dataX):
        x = Trainer.add_bias(dataX)
        temp = np.array([w[1:5], w[6:10], w[11:15]])
        phiInputs = np.matmul(x, temp.T)
        phiOutputs = Trainer.phi(phiInputs)

        phiOutputs = np.append(phiOutputs, np.ones((phiOutputs.shape[0],1)), axis=1)
        output = np.matmul(phiOutputs, np.array([w[0], w[5], w[10], w[15]]))
        return output

    def __init__(self, w1, trainX, trainY, lambda0=1e-5, gamma1=1e-5):
        self.trainX = trainX
        self.trainY = trainY
        self.lam = lambda0
        self.w1 = w1
        self.gamma1 = gamma1

    def r(self, w):
        return Trainer.f(w, self.trainX) - self.trainY
    def Dr(self, w):
        x = Trainer.add_bias(self.trainX)
        temp = np.array([w[1:5], w[6:10], w[11:15]])
        phiInputs = np.matmul(x, temp.T)

        phiRows = Trainer.phi(phiInputs)

        phidotOutputs = Trainer.phidot(phiInputs)
        temp = np.diag([w[0], w[5], w[10]])
        temp = np.matmul(phidotOutputs, temp)
        temp = np.repeat(temp, 4, axis=1)
        phidotRows = np.multiply(temp, np.tile(x, (1, 3)))

        result = (phiRows[:,0:1], phidotRows[:,:4],
                  phiRows[:,1:2], phidotRows[:,4:8],
                  phiRows[:,2:3], phidotRows[:,8:],
                  np.ones((self.trainX.shape[0], 1)))

        return np.concatenate(result, axis=1)
    def Dw(self, w):
        return np.identity(w.shape[0])
    def h(self, w):
        return np.concatenate((self.r(w), math.sqrt(self.lam)*w))
    def Dh(self, w):
        return np.concatenate((self.Dr(w), math.sqrt(self.lam)*self.Dw(w)))
    def l(self, w):
        return np.linalg.norm(self.h(w))**2
    def get_next_iterate(self, w, gamma):
        Dh = self.Dh(w)

        b = np.matmul(Dh, w) - self.h(w)
        b = np.concatenate((b, math.sqrt(gamma)*w))

        A = np.concatenate((Dh, math.sqrt(gamma)*np.identity(16)))
        pinvA = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)

        return np.matmul(pinvA, b)
    def get_optimality(self, w):
        optimalityCondResidual = np.linalg.norm(2*np.matmul(self.Dr(w).T, self.r(w)) + 2*self.lam*self.w)
        return optimalityCondResidual

    def print_init(self):
        print('='*110)
        print('lambda=' + '{:.2e}'.format(self.lam),
              '\t gamma1=' + '{:.2e}'.format(self.gamma1),
              '\t w1=', str(self.w1[:2])[:-1], '...', str(self.w1[-2:])[1:])
        print('='*110)

    def train(self, g, display=False):
        wk = self.w1; gammak = self.gamma1

        losses = []; loss = self.l(self.w1)
        rms_error = Trainer.get_rms_error(wk, self.trainX, self.trainY)

        stagnate = 0; k = 1; start_time = time.time()
        while stopping_criterion(loss, rms_error, stagnate, time.time()-start_time, k):
            losses.append(loss)

            wk_next = self.get_next_iterate(wk, gammak)
            loss_next = self.l(wk_next)

            if loss_next < loss:
                wk = wk_next
                loss = loss_next
                gammak = 0.8*gammak
            else:
                gammak = 2*gammak
                stagnate += 1

            k+=1
            rms_error = Trainer.get_rms_error(wk, self.trainX, self.trainY)
            if display:
                print(k, '\t', '{:.2f}'.format(time.time()-start_time),
                         '\t Loss:', '{:.8f}'.format(loss),
                         '\t RMS Error:', '{:.8f}'.format(rms_error),
                         #'\t Gamma:', '{:.2e}'.format(gammak),
                         end=' '*15+'\r' )

        if display:
            print()

        return wk, losses

# 3a, make it plot multiple lines for multiple noises
def plot_loss_wrt_k(g, N, w1, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        wk, losses = trainer.train(g, display=True)

        line, = ax.plot(losses)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\gamma=10^{-5}$, $\mathbf{w}_1\in[-0.1,0.1)$')
    if noise_levels != [0]:
        ax.legend()
    ax.set_xlabel('Number of iterations, $k$', fontsize=12)
    ax.set_ylabel('Loss, $l(\mathbf{w}_k)$', fontsize=12)
    plt.grid()
    plt.show()


# 3a, make it plot multiple lines for multiple noises
def plot_loss_wrt_gamma1(g, N, w1, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 1e-7
    gamma1s = np.linspace(1e-4, 1, 10)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        losses = []
        for gamma1 in gamma1s:
            trainer.gamma1 = gamma1
            unused, losses_i = trainer.train(g)
            losses.append(losses_i[-1])

        line, = ax.plot(gamma1s, losses)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\mathbf{w}_1\in[-0.1,0.1)$')
    if noise_levels != [0]:
        ax.legend()
    ax.set_xlabel('$\gamma_1$', fontsize=12)
    ax.set_ylabel('Loss, $l(\mathbf{w}_{k_{max}})$', fontsize=12)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.show()
def plot_loss_wrt_w1(g, N, w1, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 0.01
    scales = np.linspace(0.5, 2, 10)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        losses = []
        for scale in scales:
            trainer.w1 = w1*scale
            unused, losses_i = trainer.train(g)
            losses.append(losses_i[-1])


        line, = ax.plot(scales, losses)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\gamma_1=10^{-5}$')
    if noise_levels != [0]:
        ax.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('Spread of $\mathbf{w}_1$', fontsize=12)
    ax.set_ylabel('Loss, $l(\mathbf{w}_{k_{max}})$', fontsize=12)
    plt.grid()
    plt.show()


# 3b, different initializations of LM
def plot_error_wrt_gamma1(g, N, w1, NT, type, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 1e-7
    gamma1s = np.linspace(1e-5-delta, 1e-5+delta, 10)
    dataX, dataY = get_data(g, NT, type)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        errors = []
        for gamma1 in gamma1s:
            trainer.gamma1 = gamma1
            wk_final, unused = trainer.train(g)
            errors.append(Trainer.get_rms_error(wk_final, dataX, dataY))

        line, = ax.plot(gamma1s, errors)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\mathbf{w}_1\in[-0.1,0.1)$')
    if noise_levels != [0]:
        ax.legend()
    ax.set_xlabel('$\gamma_1$', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.show()
def plot_error_wrt_w1(g, N, w1, NT, type, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 0.01
    scales = np.linspace(1-delta, 1+delta, 10)
    dataX, dataY = get_data(g, NT, type)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        errors = []
        for scale in scales:
            trainer.w1 = w1*scale
            wk_final, unused = trainer.train(g)
            errors.append(Trainer.get_rms_error(wk_final, dataX, dataY))

        line, = ax.plot(scales, errors)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\gamma_1=10^{-5}$')
    if noise_levels != [0]:
        ax.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('Spread of $\mathbf{w}_1$', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    plt.grid()
    plt.show()

# 3b, different lambda and Gamma
def plot_error_wrt_lambda(g, N, w1, NT, type, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 1e-7
    lambdas = np.logspace(-6, -4, 50)
    #lambdas = np.linspace(1e-5-delta, 1e-5+delta, 100)
    dataX, dataY = get_data(g, NT, type)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        errors = []
        for lambda0 in lambdas:
            trainer.lam = lambda0
            wk_final, unused = trainer.train(g)
            errors.append(Trainer.get_rms_error(wk_final, dataX, dataY))

        line, = ax.plot(lambdas, errors)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\gamma_1=10^{-5}$, $\mathbf{w}_1\in[-0.1,0.1)$')
    if noise_levels != [0]:
        ax.legend()
    ax.set_xlabel('$\lambda$', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_xscale('log')

    plt.grid()
    plt.show()
def plot_error_wrt_Gamma(g, N, w1, NT, type, noise_levels=[0]):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)

    delta = 0.5
    scales = np.linspace(1-delta, 1+delta, 200)
    dataX, dataY = get_data(g, NT, type)

    for noise in noise_levels:
        trainX, trainY = get_data(g, N, 'train', noise=noise)
        trainer = Trainer(w1, trainX, trainY)
        wk_final, unused = trainer.train(g)

        errors = []
        for scale in scales:
            new_dataX = dataX*scale
            errors.append(Trainer.get_rms_error(wk_final, new_dataX, np.apply_along_axis(g, 1, new_dataX).T))

        line, = ax.plot(scales, errors)
        if noise_levels != [0]:
            line.set_label('$\epsilon=$ ' + str(noise))

    ax.set_title(gname + ', $\lambda=10^{-5}$, $\gamma_1=10^{-5}$, $\mathbf{w}_1\in[-0.1,0.1)$')
    if noise_levels != [0]:
        ax.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('Testing $\Gamma$', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    plt.grid()
    plt.show()

def ThreeA(g, N, w1, noise_levels=[0]):
    #plot_loss_wrt_k(g, N, w1, noise_levels=noise_levels)
    plot_loss_wrt_gamma1(g, N, w1, noise_levels=noise_levels)
    plot_loss_wrt_w1(g, N, w1, noise_levels=noise_levels)

def ThreeB(g, N, w1, NT, noise_levels=[0]):
    plot_error_wrt_gamma1(g, N, w1, N, 'train', noise_levels=noise_levels)
    plot_error_wrt_w1(g, N, w1, N, 'train', noise_levels=noise_levels)

    plot_error_wrt_gamma1(g, N, w1, NT, 'test', noise_levels=noise_levels)
    plot_error_wrt_w1(g, N, w1, NT, 'test', noise_levels=noise_levels)

    #plot_error_wrt_lambda(g, N, w1, N, 'train', noise_levels=noise_levels)
    #plot_error_wrt_lambda(g, N, w1, NT, 'test', noise_levels=noise_levels)

    #plot_error_wrt_Gamma(g, N, w1, NT, 'test', noise_levels=noise_levels)

def train_alot(g, N, noise=0):
    trainX, trainY = get_data(g, N, 'train', noise=noise)
    w1 = np.random.uniform(-0.1, 0.1, size=16)
    trainer = Trainer(w1, trainX, trainY)
    trainer.print_init()
    for i in range(20):
        unused, errors = trainer.train(g, display=True)
        trainer.w1 = np.random.uniform(-0.1, 0.1, size=16)
def stopping_criterion(loss, rms_error, stagnate, time, k):
    return rms_error > 0.006 and stagnate < 1200 # used for 3a
    #return rms_error > 0.17 and stagnate < 1000 # used for 3b
    #return k < 1000 # used for 3e

g1name = '$g(\mathbf{x})=x_1x_2+x_3$'
g2name = '$g(\mathbf{x})=69\sin(x_1)+x_2x_3$'
gname = g2name
def main():
    N=500
    NT=100
    g1 = lambda x: x[0]*x[1]+x[2];
    g2 = lambda x: 69*math.sin(x[0])+x[1]*x[2];

    #train_alot(g1, N)
    #w1=np.random.uniform(-0.1,0.1,size=16)

    # initial weights for g1
    w1g1 = np.array([-0.03907819,  0.06158748, -0.08008263, -0.06117632, -0.00110027,  0.06879582,
                     -0.00660737, -0.08597136, -0.06461364, -0.03900993,  0.07414459,  0.07428109,
                     0.06751018,  0.02483472,  0.07676263, -0.07073906])

    w1g2 = np.array([-0.03236776, -0.02194876, -0.00409352,  0.07917061, -0.07397286, -0.05966028,
                     0.03283233, -0.01403717,  0.09412561,  0.0943238,  -0.04830366,  0.05694761,
                     0.02113755, -0.07613354, -0.09833506, -0.02382732])
    # 3a
    ThreeA(g1, N, w1g1)

    # 3b
    #ThreeB(g1, N, w1g1, NT)

    # 3c
    #ThreeA(g2, N, w1g2)
    #ThreeB(g2, N, w1g2, NT)

    # 3e
    #noise_levels = [0, 0.25, 0.5, 1]
    #ThreeA(g1, N, w1g1, noise_levels=noise_levels)
    #ThreeB(g1, N, w1g1, NT, noise_levels=noise_levels)

if __name__ == '__main__':
    main()
