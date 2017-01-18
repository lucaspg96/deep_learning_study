from src import mnist_loader
from src import neuralNetwork1 as NN1

print "Loading data..."
trainingData, validationData, testData = mnist_loader.load_data_wrapper('data/mnist.pkl.gz')
print "Data loaded."

print "Training Data: ",len(trainingData)
print "Validation Data: ",len(validationData)
print "Test Data: ",len(testData)

print "Creating Neural Network..."
net = NN1.NeuralNetwork([784,30,10])

print"Starting gradient..."
net.sgd(trainingData,30,10,3.0,testData=testData)