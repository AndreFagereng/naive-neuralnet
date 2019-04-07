
import numpy as np


class net:

	def __init__(self, layerdimensions):

		self.dimensions = layerdimensions
		self.weights, self.bias = self.initiate_network(layerdimensions)

	def __repr__(self):
		return f'Weights\n{self.weights}\nBias\n{self.bias}' 

	def initiate_network(self, layerdimensions):
		
		l = len(self.dimensions)
		weights = dict()
		bias    = dict()

		# Lambda expression for making arrays with np.random.normal()
		make_arr = lambda x,y: np.random.normal(loc=0, scale=np.sqrt(2/x) ,size=(x,y))

		for depth in range(1,l):
			weights['W'+str(depth)] = make_arr(layerdimensions[depth-1], layerdimensions[depth])
			bias['b'+str(depth)]    = np.ones((layerdimensions[depth],1))

		return weights, bias

	##TODO implement sigmoid among other activations
	def activation(self, Z, activation_function):

		if activation_function == 'relu':
			return np.maximum(0.0,Z)
		elif activation_function == 'sigmoid':

			pass
		else:
			print("Error: Unimplemented activation function: {}", activation_function)
		return None

	def activation_derivative(self, Z, activation_function):

		if activation_function == 'relu':
			Z[Z>=0]  = 1
			Z[Z<0]   = 0

			return Z

		else:
			print('This activation function is not implemented')
			return None

	def softmax(self,Z):

		Z       = Z.T
		softmax = np.zeros_like(Z.T)

		for i in range(Z.shape[0]):

			exp     = np.exp(Z[i, :] - max(Z[i,:]))
			exp_sum = np.sum(np.exp(Z[i, :] - max(Z[i,:])))
			t       = exp/exp_sum

			softmax[:,i] = exp/exp_sum
		return softmax


	def forward_pass(self, X_batch, is_training=False, activation_function='relu'):
	
		prev_A = X_batch.T
		l = len(self.dimensions)
		
		activations = {'A_0':X_batch}

		for idx in range(1, l-1):
				
			A = prev_A
			Z = np.dot(self.weights['W'+str(idx)].T,A) + self.bias['b'+str(idx)]
			prev_A = self.activation(Z, activation_function)
			
			if is_training:
				activations['Z_'+str(idx)] = Z
				activations['A_'+str(idx)] = prev_A

		#Last layer
		Z = np.dot(self.weights['W'+str(l-1)].T, prev_A) + self.bias['b'+str(l-1)]
		predicted = self.softmax(Z)

		if is_training:
			activations['Z_'+str(l-1)] = Z
			activations['A_'+str(l-1)] = predicted

		return predicted, activations

	def backward_pass(self, conf, Y_predicted, Y_target, activations):
		
		l = len(conf['layer_dimensions'])
		m = Y_predicted.T.shape[0]
		grad_params = {}

		#### Last layer ####
		# Weights
		jZ_L = (Y_predicted - Y_target)
		d_w  = np.dot(activations['A_' + str(l-2)], jZ_L.T)
		d_w  = d_w/m
		grad_params['grad_W_'+str(l-1)] = d_w

		# Biases
		grad_params['grad_b_'+str(l-1)] = (np.dot(jZ_L, np.ones((m,1)))/m)

		#for key,val in activations.items():
		#	print(key,': ',val.shape)
		#print('l:' , l)
		for i in range(l-1, 1, -1):

			# Weights
			g_z  = self.activation_derivative(activations['Z_'+str(i-1)], conf['activation_function']) 
			
			tmp = np.dot(self.weights['W'+str(i)], jZ_L)

			jZ_L = g_z * tmp
        	
			d_w  = np.dot(activations['A_' + str(i-2)].T, jZ_L.T)
			d_w = d_w/m

			grad_params['grad_W_'+str(i-1)] = d_w

			# Biases
			grad_params['grad_b_'+str(i-1)] = np.dot(jZ_L, np.ones((m,1)))/m

		return grad_params

	def gradient_descent_update(self,conf, grad_params):

		lr = conf['learning_rate']
		updated_weights = {}
		updated_bias    = {}
		l = len(self.weights)

		for i in range(1,l+1):
        
			b_tmp = self.bias['b'+str(i)] - lr * grad_params['grad_b_'+str(i)]  
			w_tmp = self.weights['W'+str(i)] - lr * grad_params['grad_W_'+str(i)]

			updated_weights['W'+str(i)] = w_tmp
			updated_bias['b'+str(i)] = b_tmp

		return updated_weights, updated_bias


	def cross_entropy_cost(self, Y_predicted, Y_target):

		num_correct = 0
		cost        = None
		Y_k_idx     = []
		m           = Y_predicted.T.shape[0]

		a = Y_predicted * Y_target

		for i in range(Y_predicted.shape[1]):
          
			Y_k_idx.append(np.argmax(a.T[i]))

			if np.argmax(Y_predicted.T[i]) == np.argwhere(Y_target.T[i] == 1):
				num_correct += 1 

		log = -np.log(Y_predicted.T[range(m), Y_k_idx])
		cost = np.sum(log) / m

		return cost, num_correct



#data = np.array(([[10,5], [1,1], [5,10], [2,1], [1,10], [2,5],[1,1]]))

#layers = [2,5,2]
#nn = net(layers)
#print(nn)


#predicted, features = nn.forward_pass(X_batch=data,  is_training=True)
#print(features['A_2'])
