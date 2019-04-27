import torch
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
import _pickle as cPickle
import gzip

def load_data():
    # f = gzip.open('data/mnist/mnist.pkl.gz', 'rb')
    # training_data, validation_data, test_data = cPickle.load(f)
    # f.close()
    # return (training_data, validation_data, test_data)
    X, y = loadlocal_mnist(images_path ='data/mnist/train-images.idx3-ubyte',labels_path='data/mnist/train-labels.idx1-ubyte')
    return (X, y)
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class RedNeural:

    def __init__(self, neuronasPorCapa, alpha, maxPesosRand):
        self.D = neuronasPorCapa[0]
        self.M = neuronasPorCapa[1]
        self.K = neuronasPorCapa[2]
        self.delta_oculta = torch.zeros(self.M)
        self.delta_salida = torch.zeros(self.K)
        self.learningRate = alpha
        self.maxPesosRand = maxPesosRand
        # Inicializando los pesos de manera aleatoria
        # Valor adicional del bias
        self.pesos_oe : torch.Tensor = torch.rand(self.M, self.D)
        self.pesos_so : torch.Tensor = torch.rand(self.K, self.M)
        self.salidaFinal = None
        self.salidaOculta = None
        self.entrada = None
        self.etiqueta = None
        self.bias_oculto : torch.Tensor = torch.rand(self.K,1)
        self.bias_entrada : torch.Tensor = torch.rand(self.M,1)
        self.errorXIteracion = [[],[]]



    def dSigmoid(self, x : torch.Tensor):
        return x * (1 - x)


    def calcularPasadaAdelante(self, entrada):
        # Añadiendo elemento para que se conserve el bias
       # entrada += [[1]]
        #entrada+=[1]
        entrada = torch.Tensor(entrada)
        self.entrada = entrada
        salidaOculta = torch.mm(self.pesos_oe,entrada.reshape(entrada.shape[0],1)) #+1

        salidaOculta += self.bias_entrada

        #salidaOculta = torch.mm(self.pesos_oe, entrada)
        salidaOculta = salidaOculta.reshape(salidaOculta.shape[0])
        salidaOculta : torch.Tensor = torch.sigmoid(salidaOculta)
        # Añadiendo elemento para que se conserve el bias
        #salidaOculta = torch.cat((salidaOculta, torch.Tensor([1])), 0)
        self.salidaOculta = salidaOculta
        salidaFinal = torch.mm(self.pesos_so, salidaOculta.reshape(salidaOculta.shape[0],1)) #+ 1 #AQUI SA EL BIAS

        salidaFinal += self.bias_oculto


        salidaFinal = salidaFinal.reshape(salidaFinal.shape[0])
        self.salidaFinal = torch.sigmoid(salidaFinal.reshape(salidaFinal.shape[0]))
        return self.salidaFinal


    def calcularPasadaAtras(self):

        self.delta_salida = (self.salidaFinal - self.etiqueta) * (self.salidaFinal * (1 - self.salidaFinal))

        deltaSalidaAux = self.delta_salida.reshape(1,self.delta_salida.shape[0])

        self.delta_oculta = torch.sum(torch.mm(deltaSalidaAux, self.pesos_so),1) * (self.salidaOculta * (1 - self.salidaOculta))

        self.delta_salida = self.delta_salida.reshape(self.delta_salida.shape[0], 1)
        self.delta_oculta = self.delta_oculta.reshape(self.delta_oculta.shape[0],1)

    def evaluarClasificacionesErroneas(self, X, T):

        return 0.5 * ((X - T) ** 2)

    def entrenarRed(self, numIteraciones, X, T):

        for epoch in range(numIteraciones):
            sumaError = 0
            for data in range(len(X)):
                #self.etiqueta = torch.Tensor([T[data].copy()])
                self.etiqueta = torch.Tensor(T[data])
                self.calcularPasadaAdelante(X[data].copy())
                sumaError += self.evaluarClasificacionesErroneas(self.salidaFinal, self.etiqueta)
                self.calcularPasadaAtras()
                self.actualizarPesosSegunDeltas()
            self.errorXIteracion[0].append(epoch)
            self.errorXIteracion[1].append(sumaError)


            #print("Epoch {} \n Pesos Entrada a Oculta {} \n Pesos Oculta a Salida {}".format(epoch, self.pesos_oe, self.pesos_so))


    def evaluarMuesta(self, x):
        self.calcularPasadaAdelante(x)
        return self.salidaFinal

    def actualizarPesosSegunDeltas(self):

        #Actualizacion de pesos de Capa Oculta a Capa de Salida
        aux1 = self.learningRate *( self.delta_salida * self.salidaOculta)
        self.pesos_so -= aux1

        self.bias_oculto -= self.learningRate * self.delta_salida

        # Actualizacion de pesos de Capa de Entrada a Capa de Oculta
        aux2 = (self.learningRate *( self.delta_oculta * self.entrada))
        self.pesos_oe -= aux2

        self.bias_entrada -= self.learningRate * self.delta_oculta

    def graficarError(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.errorXIteracion[0], self.errorXIteracion[1], 'r')
        ax.set_title('Gráfica del error por iteración')
        plt.show()


if __name__ == "__main__":
    i, t = loadlocal_mnist(images_path ='data/mnist/train-images.idx3-ubyte',labels_path='data/mnist/train-labels.idx1-ubyte')
    t = [vectorized_result(number) for number in t]
    red = RedNeural([784, 397, 10], 0.5, 100)
    red.entrenarRed(1000, i, t)


    red.evaluarMuesta(i[0])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    print("Esperado: ",t[0],'\n' )

    red.evaluarMuesta(i[1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    print("Esperado: ",t[1],'\n' )

    red.evaluarMuesta(i[2])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    print("Esperado: ",t[2],'\n' )

    red.evaluarMuesta(i[3])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    print("Esperado: ",t[3],'\n' )

    red.graficarError()