import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import BNN_BBP_regression
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import math
import matplotlib.pyplot as plt

'''------------------------Hyper-parameter------------------------'''
batch_size = 1000
batch_num = 10
input_dim = 1
layer = [100, 100]
Sigma = [10, 10, 10]
output_dim = 1
epoch_num = 10
eps = 1e-4
test_size = 5000
Nsamples = 100
epsilon = 1e-3

'''------------------------generate data------------------------'''
def generate_data(epsilon, size):
	x = np.linspace(0, 0.4, size)
	noise = np.random.normal(0, epsilon, size)
	y = x + 0.3* np.sin(2 *math.pi * (x + noise)) + 0.3 * np.sin(4 * math.pi * (x + noise)) + noise
	return x, y

x_train_0, y_train_0 = generate_data(epsilon, size = batch_num * batch_size)
seq = np.arange(batch_num * batch_size)
np.random.shuffle(seq)

plt.figure()
plt.plot(x_train_0, y_train_0)
plt.show()

'''------------------------shuffling training data-----------------------'''
x_train = x_train_0.reshape([-1,1])[seq].reshape([batch_num,batch_size])
y_train = y_train_0.reshape([-1,1])[seq].reshape([batch_num,batch_size])


'''------------------------find device-----------------------'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

'''------------------------Construct model------------------------'''

model = BNN_BBP_regression.Bayes_Multi_L(input_dim, output_dim, layer, Sigma)
model.to(device)

'''------------------------Optimizer------------------------'''
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma = 1.0)

'''------------------------Training BNN------------------------'''
print("start training BNN")
#x_test = Variable(torch.Tensor(x_test)).to(device)
x_train = x_train.reshape([batch_num, batch_size])
y_train = y_train.reshape([batch_num, batch_size])

for epoch in range(epoch_num):
	scheduler.step()
	for batch_idx in range(batch_num):
		x = Variable(torch.Tensor(x_train[batch_idx,:].reshape([batch_size,1]))).to(device)
		y = Variable(torch.Tensor(y_train[batch_idx,:])).to(device)

		y_hat, qlog, plog = model.forward(x, True)
		(Loss, LL) = model.get_loss(qlog, plog, y, y_hat, batch_num, epsilon)

		optimizer.zero_grad()
		Loss.backward()
		optimizer.step()
	#(y_mean, y_test_hat) = model.prediction(x_test, Nsamples)
	#BNN_BBP_regression.demo(y_hat, y_mean, x_test, y_train, x_train)
	print('Train epoch: {} Training Loss: {}'.format((epoch + 1), Loss.detach()))

