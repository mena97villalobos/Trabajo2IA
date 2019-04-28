import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable


class RedNeuralMNIST(nn.Module):

    def _init_(self, neuronasPorCapas, alpha):
        super()._init_()
        self.D = neuronasPorCapas[0]
        self.M = neuronasPorCapas[1]
        self.K = neuronasPorCapas[2]
        self.alpha = alpha
        self.linear_ih = nn.Linear(self.D, self.M, bias=False)
        self.linear_ho = nn.Linear(self.M, self.K, bias=False)
        self.activacion = nn.Sigmoid()
        self.error_function = torch.nn.MSELoss(size_average=False)
        self.optimiser = torch.optim.SGD(self.parameters(), self.alpha)
        self.perdida = torch.Tensor([0.0])


    def forward(self, inputs_list):
        caracteristicas_entrada = Variable(torch.cuda.FloatTensor(inputs_list).view(1, self.D))
        # Combianación lineal de las entradas con los pesos respectivos
        entrada_oculta = self.linear_ih(caracteristicas_entrada)
        salida_oculta = self.activacion(entrada_oculta)
        # Combianación lineal de la capa oculta con los pesos respectivos
        entrada_capa_salida = self.linear_ho(salida_oculta)
        salida_final = self.activacion(entrada_capa_salida)
        return salida_final


    def entrenar(self, inputs_list, targets_list):
        # Calcular la salida actual de la red
        salida_pasada_adelante = self.forward(inputs_list)
        target_variable = Variable(torch.cuda.FloatTensor(targets_list).view(1, self.K), requires_grad=False)
        # Calculando error
        loss = self.error_function(salida_pasada_adelante, target_variable)
        self.perdida += loss
        # print("Error en la iteración: {}".format(loss))
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def calcular_accuracy(self):
        # TODO
        test_data_file = open("mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()
        self.perdida = torch.Tensor([0.0])
        size = 0
        for data in test_data_list[:500]:
            values = data.split(',')
            inputData = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            # Creando valores de targets para los 10 nodos de salida, todos en 0.01 menos el deseado
            target = numpy.zeros(10) + 0.01
            # Actualizando el target deseado
            target[int(all_values[0])] = 0.99
            salida_pasada_adelante = self.forward(inputData)
            target_variable = Variable(torch.cuda.FloatTensor(target).view(1, self.K), requires_grad=False)
            loss = self.error_function(salida_pasada_adelante, target_variable)
            self.perdida += loss
            size += 1
        # Promedio de los errores creo que debería ser comparar targets bien evaluados vs mal evaluados
        return self.perdida / size


n = RedNeuralMNIST([784, 200, 10], 0.4)
n.cuda()
# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 3
inputFinal = []
plotError = []
for e in range(epochs):
    print("Epoch: {}".format(e))
    # Recorrer el dataset
    for record in training_data_list[:500]:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Creando valores de targets para los 10 nodos de salida, todos en 0.01 menos el deseado
        targets = numpy.zeros(10) + 0.01
        # Actualizando el target deseado
        targets[int(all_values[0])] = 0.99
        n.entrenar(inputs, targets)
    plotError += [[e, n.perdida[0]]]
    n.perdida = torch.Tensor([0.0])
x = [i[0] for i in plotError]
y = [i[1] for i in plotError]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'r')
ax.set_title('Gráfica del Error Acumulador por Epoch')
plt.show()
# print("Accuracy: {}".format(n.calcular_accuracy()))
print("Pesos capa oculta a capa salida {}".format(n.linear_ho.weight))
print("Pesos capa entrada a capa oculta {}".format(n.linear_ih.weight))
