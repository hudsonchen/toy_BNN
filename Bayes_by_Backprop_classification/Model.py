import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

eps = 1e-4

def log_gaussian(x, mean, var):
    return (-torch.log(var) * 2 - torch.pow((x - mean) / var, 2)).sum()

class Bayes_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Bayes_Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_mu = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-3, -2))

        self.qlog = 0
        self.plog = 0

    def forward(self, x):
        if not self.training:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.output_dim)
            return output, 0, 0

        else:
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b

            output = torch.mm(x, W) + b.unsqueeze(0).expand(x.shape[0], -1)

            qlog = log_gaussian(W, self.W_mu, std_w) + log_gaussian(b, self.b_mu, std_b)
            plog = log_gaussian(W, torch.zeros_like(W), torch.ones_like(W)) + log_gaussian(b, torch.zeros_like(b), torch.ones_like(b))
            return output, qlog, plog

class Bayes_2L(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):
        super(Bayes_2L, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bfc1 = Bayes_Linear(input_dim, hid_dim)
        self.bfc2 = Bayes_Linear(hid_dim, hid_dim)
        self.bfc3 = Bayes_Linear(hid_dim, output_dim)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        qlog = 0
        plog = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, qlog_, plog_ = self.bfc1(x)
        qlog = qlog + qlog_
        plog = plog + plog_
        # -----------------
        x = self.act(x)
        # -----------------
        x, qlog_, plog_ = self.bfc2(x)
        qlog = qlog + qlog_
        plog = plog + plog_
        # -----------------
        x = self.act(x)
        # -----------------
        y_hat, qlog_, plog_ = self.bfc3(x)
        qlog = qlog + qlog_
        plog = plog + plog_

        y_hat = torch.softmax(y_hat, dim = 1)
        return y_hat, qlog, plog

    def get_loss(self, qlog, plog, y, y_hat, batch_num):
        data_likelihood = (y * torch.log(y_hat + eps) + (1. - y) * torch.log(1. - y_hat + eps)).sum()
        return 1. / batch_num * (qlog - plog) - data_likelihood

    def prediction(self, x_test, y_test):
        y_hat, qlog, plog = self.forward(x_test)
        test_loss = - (y_test * torch.log(y_hat + eps) + (1. - y_test) * torch.log(1. - y_hat + eps)).sum() / y_test.shape[0]
        y_pred = y_hat.max(dim = 1)[1]
        err = y_pred.ne(y_test.max(dim = 1)[1]).sum()
        return test_loss, err







