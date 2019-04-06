from neuralnet import net
from import_data import load_mnist
from utils import one_hot_encode
from matplotlib import pyplot as plt

from optparse import OptionParser

parser = OptionParser()

parser.add_option('-v','--verbose', dest='verbose', action='store_true', help='Show verbose training-text', default=False)
(options, args) = parser.parse_args()


conf = {
	'layer_dimensions': [784,128,10],
	'learning_rate'   : 0.01,
	'epochs'		  : 10,
	'batch_size'	  : 128,
	'activation_function' : 'relu'
}


model = net(layerdimensions=conf['layer_dimensions'])

X, y, X_test, y_test = load_mnist('data_mnist')
b = len(X) // conf['batch_size']

print('One hot encoding target values..\n')
y      = one_hot_encode(y)
y_test = one_hot_encode(y_test)

print('Scaling pixel data..\n')
X      = X/255
X_test = X_test/255

print('Starting training..\n')

loss_list_train = []
loss_list_test  = []
accuracy_train  = []
accuracy_test   = []

for i in range(2):

	total_correct_train = 0

	for j in range(1,b):
		
		X_batch = X[(j-1)*conf['batch_size']:(j)*conf['batch_size'], :] 
		y_batch = y[:, (j-1)*conf['batch_size']:(j)*conf['batch_size']]

		predictions, activations = model.forward_pass(X_batch, is_training=True)

		loss, batch_correct = model.cross_entropy_cost(predictions, y_batch)

		gradients = model.backward_pass(conf, predictions, y_batch, activations)

		updated_weights, updated_bias = model.gradient_descent_update(conf, gradients)
		
		model.weights = updated_weights
		model.bias    = updated_bias


		total_correct_train += batch_correct

	predictions, activations = model.forward_pass(X_test, is_training=False)
	loss_test, total_correct_test  = model.cross_entropy_cost(predictions, y_test)
	
	accuracy_train.append(round((total_correct_train/len(X))*100, 2))
	accuracy_test.append(round(total_correct_test/len(X_test)*100, 2))
	loss_list_train.append(loss)
	loss_list_test.append(loss_test)

	if options.verbose:
		print('-'*50)
		print(f'Epoch: {i}\n')
		print(f'Train loss: {round(loss, 2)}')
		print(f'Train accuracy: {round((total_correct_train/len(X))*100, 2)}\n')
		print(f'Test loss: {round(loss_test, 2)}')
		print(f'Train accuracy: {round(total_correct_test/len(X_test)*100, 2)}\n')
	else:
		print(f'Epoch: {i}')

print('Training completed')
print('-'*50)
print(f'Train loss: {round(loss, 2)}')
print(f'Train accuracy: {round((total_correct_train/len(X))*100, 2)}%\n')
print(f'Test loss: {round(loss_test, 2)}')
print(f'Train accuracy: {round(total_correct_test/len(X_test)*100, 2)}%\n')


fig = plt.figure(1)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

ax1.plot(accuracy_train, 'b')
ax1.plot(accuracy_test, 'r')

ax2.plot(loss_list_train, 'b')
ax2.plot(loss_list_test, 'r')


plt.show()
