#!/usr/bin/env python
import sys
sys.path.append('../')
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import itertools, copy, torch, scipy
from scipy import integrate, signal
from enum import Enum
import matplotlib.pyplot as plt
import torch.onnx
from helpers.networkarch import NeuralNetwork
np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

dtype = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1
torch.manual_seed(seed)
np.random.seed(seed = seed)
# torch.autograd.set_detect_anomaly(True)
# torch.set_num_threads(8)

DT_DATA_DEFAULT = 0.05
DT_CTRL_DEFAULT = 0.1


class L3():

	def __init__(self, N_x = 1, N_z = 16, N_e = 10, epochs = 50, batch_size = 512, retrain: bool=True,):
		self.n_x = N_x
		self.n_z = N_z
		self.n_e = N_e
		self.epochs = epochs
		self.batch_size = batch_size

		self.retrain = retrain
		self.optimizer = None
		self.model_fn = 'model'
		self.model = None

	def step(self, Fx_arr_batch: torch.Tensor, model: torch.nn.Module, loss_fn: Callable):
		# Send data to GPU if applicable
		Fx_arr_batch = Fx_arr_batch[0].to(device)
		# Fx_arr_batch is the data [batch, time_steps, states]

		# There are three kinds of losses that we are interested in:
		# 1. The decoding loss
		# 2. The prediction error for multiple time-steps into the future
		Fx_arr_batch_data = Fx_arr_batch[:,:,:self.n_x]
		Fx_arr_batch_ctrl = Fx_arr_batch[:,:,self.n_x:]
		x_hat, eta_hat = model(Fx_arr_batch_data[:,0,:],Fx_arr_batch_ctrl[:,0,:])
		multi_step_loss_state = loss_fn(Fx_arr_batch_data[:, 1, :], x_hat)
		multi_step_loss_eta = loss_fn(model.enc(Fx_arr_batch_data[:, 1, :]), eta_hat)
		xi_hat = torch.cat((x_hat, eta_hat), 1)
		time_weight = 0.95
		for time_step in range(Fx_arr_batch_data.shape[1]-2):
			#print('timestep:',time_step)
		#for time_step in range(3):
			#x_hat, eta_hat = model(x_hat)
			x_hat, eta_hat = model.ldm(xi_hat,Fx_arr_batch_ctrl[:,time_step+1,:])
			xi_hat = torch.cat((x_hat, eta_hat), 1)
			multi_step_loss_state += time_weight**(time_step+1)*loss_fn(Fx_arr_batch_data[:, time_step+2, :], x_hat)
			multi_step_loss_eta += time_weight**(time_step+1)*loss_fn(model.enc(Fx_arr_batch_data[:, time_step+2, :]), eta_hat)
		#print("range:",Fx_arr_batch_data.shape[1]-2)
		#return (1/self.n_x)*multi_step_loss_state + (0.1/self.n_e)*multi_step_loss_eta, (1/self.n_x)*multi_step_loss_state
		return multi_step_loss_state + 100*multi_step_loss_eta, multi_step_loss_state
		#return 0.001*decoding_loss + (1/3)*multi_step_loss_state + multi_step_loss_eta

	def train_model(self, model: torch.nn.Module, Fx_arr: torch.Tensor, title: str=None):
		model.cuda()
		# Reshape x and y to be vector of tensors
		# x = torch.transpose(x,0,1)
		# y1 = torch.transpose(y1,0,1)
		# y2 = torch.transpose(y2,0,1)
		# y3 = torch.transpose(y3,0,1)
		# y4 = torch.transpose(y4,0,1)
		# Split dataset into training and validation sets
		N_train = int(3*Fx_arr.shape[0]/5)
		#dataset = torch.utils.data.TensorDataset(x, y1)
		dataset = torch.utils.data.TensorDataset(Fx_arr)
		train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [N_train,Fx_arr.shape[0]-N_train])

		# Construct dataloaders for batch processing
		#batch_size = 512
		batch_size = self.batch_size
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
		val_loader   = torch.utils.data.DataLoader(dataset=val_dataset  , batch_size=batch_size)

		# Define learning hyperparameters
		#loss_fn       = torch.nn.MSELoss(reduction='sum')
		#loss_fn = torch.nn.MSELoss(reduction='mean')
		loss_fn = torch.nn.MSELoss(reduction='mean')
		#learning_rate = .001
		#n_epochs      = 250
		#learning_rate = .00025
		#learning_rate = .001
		#learning_rate = .002
		#learning_rate = .002
		#learning_rate = .025
		learning_rate = .0002
		#learning_rate = .00005
		#learning_rate = .000005
		#learning_rate = 0.0000005
		#learning_rate = 0.0005
		#n_epochs       = 100
		#n_epochs = 50
		print('learning_rate: ', learning_rate)
		n_epochs = self.epochs
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		optimizer     = self.optimizer
		lr_exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.999)
		lr_exp_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,500], gamma=0.5)

		# Initialize arrays for logging
		training_losses = []
		training_state_losses = []
		validation_losses = []
		validation_state_losses = []

		# Main training loop
		try:
			for t in range(n_epochs):
				# Validation
				with torch.no_grad():
					losses = []
					state_losses = []
					for data_slice in val_loader:
						loss, state_loss = self.step(data_slice, model, loss_fn)
						state_losses.append(state_loss.item())
						losses.append(loss.item())
					validation_losses.append(np.mean(losses))
					validation_state_losses.append(np.mean(state_losses))

				# Terminating condition
				# if t>50 and np.mean(validation_losses[-20:-11])<=np.mean(validation_losses[-10:-1]):
				# 	break

				# Training
				losses = []
				state_losses = []
				for data_slice in train_loader:
					loss, state_loss = self.step(data_slice, model, loss_fn)
					losses.append(loss.item())
					state_losses.append(state_loss.item())
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				training_losses.append(np.mean(losses))
				training_state_losses.append(np.mean(state_losses))

				lr_exp_scheduler.step()
				lr_exp_scheduler_2.step()

				pstr = f"[{t+1}] Training loss: {training_losses[-1]:.3f}\t Validation loss: {validation_losses[-1]:.6f}\t Validation state loss: {validation_state_losses[-1]:.6f}"

				print(pstr)

				if t % 100 == 0:
					torch.save({'model_dict': model.state_dict()},'model_'+str(t)+'.pt')
					torch.save({'opt_dict': optimizer.state_dict()},'optimizer_'+str(t)+'.pt')

			self.plot_losses(training_losses, validation_losses,title='Full Losses')
			self.plot_losses(training_state_losses, validation_state_losses, title='State Losses')
		except KeyboardInterrupt:
			print('Stopping due to keyboard interrupt. Save and continue (Y/N)?')
			ans = input()
			if ans[0].upper() == 'N':
				exit(0)

		model.eval()

		return model

	def plot_losses(self, training_losses, validation_losses, title = None):
		fig, axs = plt.subplots(1,1)
		axs.semilogy(range(len(  training_losses)),   training_losses, label=  'Training Loss')
		axs.semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
		axs.set_xlabel('Epoch')
		axs.set_ylabel('Loss')
		axs.legend()
		if title is not None:
			axs.set_title(title)
		plt.savefig(title+'.png')

		plt.show()



	def learn(self, data: np.array):
		# Copy data for manipulation
		data = copy.deepcopy(data)
		data = torch.from_numpy(data).type(dtype)
		data.to('cuda:0')

		# Format data
		# x_minus = torch.transpose(torch.from_numpy(data['x'  ]['minus']).type(dtype), 0,1)
		# x_plus1  = torch.transpose(torch.from_numpy(data['x'  ]['plus1' ]).type(dtype), 0,1)
		# x_plus2  = torch.transpose(torch.from_numpy(data['x'  ]['plus2' ]).type(dtype), 0,1)
		# x_plus3  = torch.transpose(torch.from_numpy(data['x'  ]['plus3' ]).type(dtype), 0,1)
		# x_plus4  = torch.transpose(torch.from_numpy(data['x'  ]['plus4' ]).type(dtype), 0,1)

		# Initialize model
		if self.model is None:
			self.model = NeuralNetwork(N_x = self.n_x, N_e = self.n_e, N_h = self.n_z)
		# x_minus.to('cuda:0')
		# x_plus1.to('cuda:0')
		# x_plus2.to('cuda:0')
		# x_plus3.to('cuda:0')
		# x_plus4.to('cuda:0')
		# Train/load model
		if self.retrain:
			#self.model = self.train_model(self.model, x_minus, x_plus1, x_plus2, x_plus3, x_plus4).to('cuda:0')
			self.model = self.train_model(self.model, data).to('cuda:0')
			torch.save({'model_dict': self.model.state_dict()},'{}.pt'.format(self.model_fn))
			torch.save({'opt_dict': self.optimizer.state_dict()},'optimizer.pt')


			# x = torch.randn(batch_size, 2, 224, 224, requires_grad=True)

			
		else:
			self.model.load_state_dict(torch.load('{}.pt'.format(self.model_fn)))

			self.trained = True

	# def augmented_state(x):
	# 	x_shape = x.shape
	# 	if len(x_shape)==3:
	# 		x = x.reshape(-1, x.shape[-1])

	# 	x = torch.from_numpy(x.T).type(dtype)

		
	# 	xs = x
		
	# 	eta = self.model.g(xs)

	# 	xs = torch.cat((xs,eta), 0)
	# 	self.augmented_state = augmented_state

	# 	return xs.detach().numpy()

	def regress_new_LDM(self, data):
		# Copy data for manipulation
		data = copy.deepcopy(data)

		p_data = self.generate_data_ldm(data)

		# Format data
		x_minus = self.flatten_trajectory_data(data['x'  ]['minus'])
		z_minus = self.flatten_trajectory_data(data['eta']['minus'])
		x_plus  = self.flatten_trajectory_data(data['x'  ]['plus' ])
		z_plus  = self.flatten_trajectory_data(data['eta']['plus' ])

		xs_minus = []
		xs_plus = []
		for i in range(len(x_minus)):
			xa_minus = self.augmented_state(x_minus[i])
			xa_plus  = self.augmented_state(x_plus [i])

			xs_minus.append(np.concatenate((xa_minus),0))
			xs_plus.append(xa_plus)

		xs_minus = np.asarray(xs_minus)
		xs_plus  = np.asarray(xs_plus )

		ldm = np.linalg.lstsq(xs_minus,xs_plus,rcond=None)[0].T

		self.model.A.weight.data = torch.from_numpy(ldm[                 :self.n_x         ,:]).type(dtype)
		self.model.H.weight.data = torch.from_numpy(ldm[self.n_x,        :                 ,:]).type(dtype)

		return ldm


	def calc_eig(self):
		A_ = copy.deepcopy(self.model.A.weight.data.detach().numpy())
		H_ = copy.deepcopy(self.model.H.weight.data.detach().numpy())

		K = np.array([[A_],
					[H_]])

		w,v = eig(K)
		print('E-value: ',w)
		print('E-vector: ', v)

		return w, v


	# def generate_data_ldm(self, data):
	# 	self.model.


	def flatten_trajectory_data(data: np.ndarray):
		return data.reshape(-1, data.shape[-1])
