import random
import numpy as np

class NeuralNetwork(object):

	def __init__(self,sizes):
		#sizes: number of neurons in respective layers:
		#[input_layer, hidden_layer1, hidden_layer2, ..., output_layer]

		self.numLayers = len(sizes)

		self.sizes = sizes

		#doesn't set bias for the first layer, assuming that is the input layer
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]

		self.weights = [np.random.randn(y,x)
						for x,y in zip(sizes[:-1], sizes[1:])]

	def feedForward(self, a):
		"""
			Return the output of the network with the input "a" 
		"""

		for b,w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w,a)+b)

		return a

	def sgd(self, trainingData, epochs, miniBatchSize, eta, testData=None):
		"""
			Train the network using the stochastic gradient desent method with mini-bash.
			- trainingData is a list of tuples (x,y) representing, respectively, the input and the (desired) output.
			- the others required parameters are self-explanatory.
			- testData is used to, when provided, evaluate the network after each epoch. It's useful for tracking progress, but be warned: this slows down the process!

		"""

		if testData: nTtest = len(testData)

		n = len(trainingData)

		for j in xrange(epochs):
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in xrange(0,n,miniBatchSize)]

			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch,eta)

			if testData:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(testData),nTtest)
			else: 
				print "Epoch {0} - complete".format(j)

	def updateMiniBatch(self, miniBatch, eta):
		"""
			Update the network weights and biases using backpropagation to a miniBatch.
			- miniBatch is a list of tuples (x,y) from the training data.
			- eta is the learning rate.
		"""
		n = len(miniBatch)
		nablaB = [np.zeros(b.shape) for b in self.biases]
		nablaW = [np.zeros(w.shape) for w in self.weights]

		for x,y in miniBatch:
			deltaNablaB, deltaNablaW = self.backpropagation(x,y)
			nablaB = [nb+dnb for nb, dnb in zip(nablaB,deltaNablaB)]
			nablaW = [nw+dnw for nw, dnw in zip(nablaW,deltaNablaW)]

		self.weights = [w-(eta/n)*nw for w,nw in zip(self.weights, nablaW)]
		self.biases = [b-(eta/n)*nb for b,nb in zip(self.biases,nablaB)]

	def backpropagation(self, x, y):
        
	    nabla_b = [np.zeros(b.shape) for b in self.biases]
	    nabla_w = [np.zeros(w.shape) for w in self.weights]
	    
	    activation = x #activation of the current layer
	    activations = [x] #list of all activations, layer by layer
	    zs = [] #list of the "z" vectors
	    for b, w in zip(self.biases, self.weights):
	        z = np.dot(w, activation)+b
	        zs.append(z)
	        activation = sigmoid(z)
	        activations.append(activation)
	    
	    delta = self.cost_derivative(activations[-1], y) * \
	        sigmoid_prime(zs[-1])
	    nabla_b[-1] = delta
	    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

	    for l in xrange(2, self.numLayers):
	        z = zs[-l]
	        sp = sigmoid_prime(z)
	        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	        nabla_b[-l] = delta
	        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	    return (nabla_b, nabla_w)

	def evaluate(self, test_data):
	    test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
	    return sum(int(x == y) for (x, y) in test_results)

   	def cost_derivative(self, output_activations, y):
   		return (output_activations-y)

def sigmoid(z):
	    """The sigmoid function."""
	    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))