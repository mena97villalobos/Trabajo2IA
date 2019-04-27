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
        # Inicializando los pesos de manera aleatoria
        # Valor adicional del bias
        self.pesos_oe : torch.Tensor = torch.rand(self.M, self.D)
        #self.pesos_oe = torch.Tensor([[-4.3561,  4.3269],[-0.8986,  0.2353],        [-1.7860,  1.5103],        [-3.7588,  3.8514],        [ 5.1884, -5.4514],        [-0.0425, -0.5171],        [ 3.5132, -3.3224],        [-1.7697,  1.4660],        [ 5.4206, -5.7103],        [ 4.4498, -4.5519],        [-4.8896,  4.7710],        [-4.8967,  4.7743],        [-4.9030,  4.7749],        [-1.1180,  0.4828],        [-5.1811,  5.0248]])
        self.pesos_so : torch.Tensor = torch.rand(self.K, self.M)
        #self.pesos_so = torch.Tensor([[ 0.8236, -0.7595, -0.5422,  0.6657,  2.9399, -0.3755,  0.2504, -0.2873,4.4854,  1.4221,  1.7347,  2.1508,  2.0846, -0.3335,  2.7299]])
        self.salidaFinal = None
        self.salidaOculta = None
        self.entrada = None
        self.etiqueta = None
        self.bias_oculto : torch.Tensor = torch.rand(self.K,1)
        #self.bias_oculto = torch.Tensor([[-4.3157]])
        self.bias_entrada : torch.Tensor = torch.rand(self.M,1)
        #self.bias_entrada = torch.Tensor([[-3.0266],[-3.9380],[-3.5800],[-2.9217],[-3.0197],[-3.8357],        [-2.5004],        [-3.5859],        [-3.1173],        [-2.7470],        [-3.1659],        [-3.1658],        [-3.1640],        [-3.8913],        [-3.2581]])
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
                self.etiqueta = torch.Tensor([T[data]])
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
    red = RedNeural([2, 15, 1], 0.8, 100)
    i = [[1,1], [1, 0], [0, 1], [0,0]]
    t = [0,1,1,0]
    # El copy se usa para pasarlo por valor y no referencia y que no se modifique el i
    red.entrenarRed(2000, i, t)
    red.evaluarMuesta([1,1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    # print("pesos_oe ",red.pesos_oe)
    # print("pesos_so ",red.pesos_so)
    # print("bias_entrada", red.bias_entrada)
    # print("bias_oculto", red.bias_oculto)
    red.graficarError()