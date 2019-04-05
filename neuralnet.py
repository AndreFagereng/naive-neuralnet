
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

	def softmax(self,Z):

		Z       = Z.T
		softmax = np.zeros_like(Z.T)

		for i in range(Z.shape[0]):

			exp     = np.exp(Z[i, :] - max(Z[i,:]))
			exp_sum = np.sum(np.exp(Z[i, :] - max(Z[i,:])))
			t       = exp/exp_sum

			softmax[:,i] = exp/exp_sum
		return softmax


	def forward_pass(self, X_batch, is_training=True, activation_function='relu'):
	
		prev_A = X_batch.T
		l = len(self.dimensions)
		
		features = {'A_0':X_batch}

		for idx in range(1, l-1):
				
			A = prev_A
			Z = np.dot(self.weights['W'+str(idx)].T,A) + self.bias['b'+str(idx)]
			prev_A = self.activation(Z, activation_function)
			
			if is_training:
				features['Z_'+str(idx)] = Z
				features['A_'+str(idx)] = prev_A

		#Last layer
		Z = np.dot(self.weights['W'+str(l-1)].T, prev_A) + self.bias['b'+str(l-1)]
		predicted = self.softmax(Z)

		if is_training:
			features['Z_'+str(l-1)] = Z
			features['A_'+str(l-1)] = predicted

		return predicted, features

	def backward_pass(self):
		pass
	def update_gradient(self):
		pass
	def cross_entropy_cost(self,Y_predicted, Y_target):
		pass


data = np.array(([[10,5], [1,1], [5,10], [2,1], [1,10], [2,5],[1,1]]))

layers = [2,5,2]
nn = net(layers)
print(nn)


predicted, features = nn.forward_pass(X_batch=data,  is_training=True)
print(features['A_2'])
