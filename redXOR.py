import torch
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np


class RedNeural:

    def __init__(self, neuronasPorCapa, alpha, maxPesosRand):
        self.D = neuronasPorCapa[0]
        self.M = neuronasPorCapa[1]
        self.K = neuronasPorCapa[2]
        self.delta_oculta = torch.zeros(self.M)
        self.delta_salida = torch.zeros(self.K)
        self.learningRate = alpha
        self.maxPesosRand = maxPesosRand
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

        entrada = torch.Tensor(entrada)
        self.entrada = entrada
        salidaOculta = torch.mm(self.pesos_oe,entrada.reshape(entrada.shape[0],1)) #+1

        salidaOculta += self.bias_entrada

        salidaOculta = salidaOculta.reshape(salidaOculta.shape[0])
        salidaOculta : torch.Tensor = torch.sigmoid(salidaOculta)

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
                self.etiqueta = torch.Tensor(T[data])
                self.calcularPasadaAdelante(X[data].copy())
                sumaError += self.evaluarClasificacionesErroneas(self.salidaFinal, self.etiqueta)
                self.calcularPasadaAtras()
                self.actualizarPesosSegunDeltas()
            self.errorXIteracion[0].append(epoch)
            self.errorXIteracion[1].append(sumaError)

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
        ax.set_title('Gráfica del Error por Iteración')
        plt.show()



if __name__ == "__main__":
    red = RedNeural([2, 5, 1], 0.6, 100)
    i = [[1,1], [1, 0], [0, 1], [0,0]]
    t = [[0],[1],[1],[0]]

    red.entrenarRed(50000,i, t)

    red.evaluarMuesta([1,1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))

    red.evaluarMuesta([1,0])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))

    red.evaluarMuesta([0, 1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))

    red.evaluarMuesta([0, 0])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))

    red.graficarError()