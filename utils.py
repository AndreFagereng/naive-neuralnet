import numpy as np

'''
	Utilities

'''

test = [5,2,8,10,4]

def one_hot_encode(vec):
	
	
	length = len(vec)
	size   = len(np.unique(vec))

	one_hot = np.zeros((size, length))
	
	one_hot[vec, np.arange(length)] = 1	
	
	return one_hot

    #Y_tilde[Y, np.arange(m)] = 1
    #return Y_tilde
	




#one_hot_encode(test)