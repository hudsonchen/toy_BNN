import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import BNN_Bayes_by_backprop
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

'''------------------------Hyper-parameter------------------------'''
batch_size = 100
batch_num = 500
input_dim = 784
hid_dim = 200
output_dim = 10
epoch_num = 20
eps = 1e-4
test_size = 5000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
'''------------------------Loading training data------------------------'''
train_image = np.load("train_data.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, input_dim])
train_label = np.load("train_label.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, output_dim])

test_image = np.load("test_data.npy")[:5000,:]
test_label = np.load("test_label.npy")[:5000,:]
'''------------------------Construct model------------------------'''

model = BNN_Bayes_by_backprop.Bayes_2L(input_dim, output_dim, hid_dim)
model.to(device)

'''------------------------Optimizer------------------------'''
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma = 0.9)

'''------------------------Training BNN------------------------'''
print("start training BNN")
x_test = Variable(torch.Tensor(test_image)).to(device)
y_test = Variable(torch.Tensor(test_label)).to(device)

for epoch in range(epoch_num):
	scheduler.step()
	for batch_idx in range(batch_num):
		x = Variable(torch.Tensor(train_image[batch_idx,:,:])).to(device)
		y = Variable(torch.Tensor(train_label[batch_idx,:,:])).to(device)

		y_hat, qlog, plog = model.forward(x)
		Loss = model.get_loss(qlog, plog, y, y_hat, batch_num)

		optimizer.zero_grad()
		Loss.backward()
		optimizer.step()

	test_loss, err = model.prediction(x_test, y_test)
	print('Train epoch: {} Training Loss: {}'.format((epoch + 1), Loss.detach()))
	print('Train epoch: {} Test Loss: {} error: {}'.format((epoch + 1), test_loss.detach(), err.detach()))


