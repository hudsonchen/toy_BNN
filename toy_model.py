import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

def generate_data():
	x = np.linspace(-10,10, num = 1000)
	y = np.sin(x)* 10
	noise = np.random.randn(*y.shape)
	return y + noise

def sample(mean, logvar):
	var = torch.exp(logvar)
	epsilon = torch.randn_like(mean)
	return (mean + var * epsilon , epsilon)

class toy_BNN(nn.Module):
	def __init__(self, hidden_dim_1, hidden_dim_2):
		super(toy_BNN, self).__init__()
		self.flow = nn.Sequential(nn.Linear(1, hidden_dim_1),
			nn.Tanh(),nn.Linear(hidden_dim_1, hidden_dim_2), nn.Tanh())
		
		self.get_mean = nn.Linear(hidden_dim_2, 1)
		self.get_logvar = nn.Linear(hidden_dim_2, 1)


	def forward(self, x):
		tp = self.flow(x)
		return self.get_mean(tp), self.get_logvar(tp)

	def Likelihood_Loss(self, x, mean, logvar):
		var = torch.exp(logvar)
		LL = torch.sum(- torch.log(var) - 0.5 * torch.pow((x - mean) / var, 2))
		return LL

	def KLD(self, mean, logvar, M):
		KLD_tp = torch.zeros(M).double()
		var = torch.exp(logvar)
		f_D, eps = sample(mean, var) 
		for i in range(M):
			log_q = -torch.log(var) - 0.5 * torch.pow(eps ,2)
			log_p = -0.5 * f_D ** 2
			KLD_tp[i] = torch.sum(log_q - log_p)
		return torch.mean(KLD_tp)


data = generate_data()
data_tensor = Variable(torch.tensor(data.reshape([-1,1])))
hidden_dim_1 = 200
hidden_dim_2 = 100
epoch_num = 100
Sampling_num = 50
BNN = toy_BNN(hidden_dim_1, hidden_dim_2).double()
optimizer = torch.optim.Adam(BNN.parameters(), lr = 1e-4)


for i in range(epoch_num):
	mean, logvar = BNN.forward(data_tensor)
	LL = BNN.Likelihood_Loss(data_tensor, mean, logvar)
	KLD = BNN.KLD(mean, logvar, Sampling_num)
	Loss = -(LL - KLD)
	#print('ELBO: {} Log-likelihood:{} KL-Divergence:{}'.format(-Loss, LL, KLD))
	optimizer.zero_grad()
	Loss.backward()
	optimizer.step()
	"""
	if(i % 5 == 0):
		X_hat = sample(mean, logvar)[0]
		plt.plot(X_hat.detach().numpy()[:30])
		plt.show()
	"""

BNN.eval()
mean, logvar = BNN.forward(data_tensor)
X_hat = sample(mean, logvar)[0]
plt.figure(1)
plt.title('original data')
plt.plot(data)
plt.figure(2)
plt.title('after training with KL-Divergence')
plt.plot(X_hat.detach().numpy())


BNN_NoKL = toy_BNN(hidden_dim_1, hidden_dim_2).double()
optimizer_NoKL = torch.optim.Adam(BNN_NoKL.parameters(), lr = 1e-4)


for i in range(epoch_num):
	mean, logvar = BNN_NoKL.forward(data_tensor)
	LL = BNN_NoKL.Likelihood_Loss(data_tensor, mean, logvar)
	Loss_NoKL = -LL
	#print('ELBO: {} Log-likelihood:{}'.format(-Loss, LL))
	optimizer_NoKL.zero_grad()
	Loss_NoKL.backward()
	optimizer_NoKL.step()
	"""
	if(i % 5 == 0):
		X_hat = sample(mean, logvar)[0]
		plt.plot(X_hat.detach().numpy()[:30])
		plt.show()
	"""

BNN_NoKL.eval()
mean, logvar = BNN_NoKL.forward(data_tensor)
X_hat_NoKL = sample(mean, logvar)[0]
plt.figure(3)
plt.title('after training without KL-Divergence')
plt.plot(X_hat_NoKL.detach().numpy())
plt.show()