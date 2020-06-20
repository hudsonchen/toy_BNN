import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import distributions as dist

eps = 1e-4

def log_gaussian(x, mean, var):
    return (-torch.log(var) * 2 - torch.pow((x - mean) / var, 2)).sum()

class Bayes_Linear(nn.Module):
    def __init__(self, input_dim, out_dim, sigma):
        super(Bayes_Linear, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma = sigma

        self.W_mu = nn.Parameter(torch.Tensor(self.input_dim, self.out_dim).uniform_(-1, 1))

        self.W_logvar = nn.Parameter(torch.Tensor(self.input_dim, self.out_dim).uniform_(-2, -1))

        self.b_mu = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-1, 1))
        self.b_logvar = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-2, -1))

        self.qlog = 0
        self.plog = 0

    def forward(self, x, sample):
        if not sample:
            out = torch.mm(x, self.W_mu.data) + self.b_mu.data.unsqueeze(0).expand(x.shape[0], -1)
            return out, 0, 0

        else:
            eps_W = Variable(torch.randn_like(self.W_mu.data))
            eps_b = Variable(torch.randn_like(self.b_mu.data))

            std_w = 1e-6 + torch.exp(self.W_logvar)
            std_b = 1e-6 + torch.exp(self.b_logvar)

            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b
            """
            print('w', W[:5,0])
            print('w_mu', self.W_mu[:5,0])
            print('std', std_w[:5,0])
            print('eps', eps_W[:5,0])
            """
            out = torch.mm(x, W) + b.unsqueeze(0).expand(x.shape[0], -1)
            qlog = log_gaussian(W, self.W_mu, std_w) + log_gaussian(b, self.b_mu, std_b)
            plog = log_gaussian(W, torch.zeros_like(W), torch.ones_like(W) * self.sigma) + log_gaussian(b, torch.zeros_like(b), torch.ones_like(b) * self.sigma)
            return out, qlog, plog

class Bayes_Multi_L(nn.Module):
    def __init__(self, input_dim, out_dim, layer, Sigma):
        super(Bayes_Multi_L, self).__init__()
        self.layers = nn.ModuleList([])
        self.input_dim = input_dim
        self.out_dim = out_dim

        layer_temp = Bayes_Linear(input_dim, layer[0], Sigma[0])
        self.layers.append(layer_temp)
        for i in range(len(layer)):
            if i == (len(layer) - 1):
                layer_temp = Bayes_Linear(layer[i], out_dim, Sigma[i+1])
            else:
                layer_temp = Bayes_Linear(layer[i], layer[i+1], Sigma[i+1])
            self.layers.append(layer_temp)
        self.act = nn.Sigmoid()

    def forward(self, x, sample):
        qlog = 0
        plog = 0
        current_data = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        for i in range(len(self.layers)):
            if (i==(len(self.layers) - 1)):
                layer = self.layers[i]
                current_data, qlog_, plog_ = layer.forward(current_data, sample)
                qlog += qlog_
                plog += plog_                             
            else:
                layer = self.layers[i]
                current_data, qlog_, plog_ = layer.forward(current_data, sample)
                current_data = self.act(current_data)
                qlog += qlog_
                plog += plog_
        return current_data, qlog, plog

    def get_loss(self, qlog, plog, y, y_hat, batch_num, epsilon):
        data_likelihood = log_gaussian(y_hat, y, torch.ones_like(y_hat) * epsilon).sum()

        #print('data_likelihood', data_likelihood)
        return (1. / batch_num * (qlog - plog) - data_likelihood, data_likelihood)

    def prediction(self, x_test, Nsamples):
        y_mean, qlog, plog = self.forward(x_test, False)
        y_test_hat = torch.zeros([Nsamples, x_test.shape[0]])
        for i in range(Nsamples):
            y_dummy, qlog, plog = self.forward(x_test, True)
            y_test_hat[i,:] = y_dummy.squeeze()
        return y_mean, y_test_hat


def demo(y, y_mean, x_test, y_train, x_train): # NSample * testdata_num
    y_lower = np.percentile(y.detach().numpy(), 25, axis = 0)
    y_higher = np.percentile(y.detach().numpy(), 75, axis = 0)
    plt.figure()
    plt.ylim(-1.9,1.4)
    plt.xlim(-0.3, 1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_test.detach().numpy(), y_mean.detach().numpy())
    plt.scatter(x_train, y_train)
    plt.fill_between(x_test.detach().numpy(), y_lower, y_higher, interpolate=True, alpha=0.3)
    plt.plot()
    plt.show()





