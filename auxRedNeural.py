import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable


# In [12]
class NeuralNetwork(nn.Module):

    def __init__(self, neuronasPorCapas, alpha, maxPesosRand):
        super().__init__()
        self.inodes = neuronasPorCapas[0]
        self.hnodes = neuronasPorCapas[1]
        self.onodes = neuronasPorCapas[2]
        # learning rate
        self.alpha = alpha
        # define the layers and their sizes, turn off bias
        self.linear_ih = nn.Linear(neuronasPorCapas[0], neuronasPorCapas[1], bias=False)
        self.linear_ho = nn.Linear(neuronasPorCapas[1], neuronasPorCapas[2], bias=False)
        # define activation function
        self.activation = nn.Sigmoid()
        # create error function
        self.error_function = torch.nn.MSELoss(size_average=False)
        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), self.alpha)


    def forward(self, inputs_list):
        # convert list to a 2-D FloatTensor then wrap in Variable
        # also shift to GPU, remove .cuda. if not desired
        inputs = Variable(torch.cuda.FloatTensor(inputs_list).view(1, self.inodes))
        # combine input layer signals into hidden layer
        hidden_inputs = self.linear_ih(inputs)
        # apply sigmiod activation function
        hidden_outputs = self.activation(hidden_inputs)
        # combine hidden layer signals into output layer
        final_inputs = self.linear_ho(hidden_outputs)
        # apply sigmiod activation function
        final_outputs = self.activation(final_inputs)
        return final_outputs


    def train(self, inputs_list, targets_list):
        # calculate the output of the network
        output = self.forward(inputs_list)
        # create a Variable out of the target vector, doesn't need gradients calculated
        # also shift to GPU, remove .cuda. if not desired
        target_variable = Variable(torch.cuda.FloatTensor(targets_list).view(1, self.onodes), requires_grad=False)
        # calculate error
        loss = self.error_function(output, target_variable)
        # print(loss.data[0])
        # zero gradients, perform a backward pass, and update the weights.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

# In[13]

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork([input_nodes, hidden_nodes, output_nodes], learning_rate, 5)

# move neural network to the GPU, delete if not desired
n.cuda()

# In[14]


# load the mnist training data CSV file into a list
training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# In [15]

# %%timeit -n1 -r1 -c

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 10
inputFinal = []
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    #print("Para epoch {} \n Pesos Oculta {} \n Pesos Salida {} \n".format(e, n.linear_ih, n.linear_ho))

a = "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,255,103,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,205,253,253,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,205,253,253,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,44,233,253,244,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,253,253,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,153,253,240,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,208,253,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,253,253,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,110,253,235,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,223,235,130,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,253,235,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,145,253,231,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,220,231,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,205,253,176,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,125,253,185,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,214,231,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,253,225,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,205,207,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,249,233,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,253,89,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
a = a.split(',')
a = (numpy.asfarray(a[1:]) / 255.0 * 0.99) + 0.01
print(n.forward(a))