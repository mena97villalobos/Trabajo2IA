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
        #self.pesos_oe : torch.Tensor = torch.rand(self.M, self.D)
        self.pesos_oe = torch.Tensor([[-2.0669,  1.1724], [-5.3924,  4.4594], [ 4.5612, -5.5052], [-5.3783,  4.4875], [-5.3159,  4.4148], [-5.2931,  4.4065], [ 5.9003,  5.2954], [-5.8603,  4.8858], [-0.7211, -0.5424], [-2.1043,  1.2004], [ 3.8150, -4.5788], [ 5.1162, -6.1825], [ 5.1113, -6.1521], [-5.1674,  4.3163], [-5.4477,  4.5300]])
        #self.pesos_so : torch.Tensor = torch.rand(self.K, self.M)
        self.pesos_so = torch.Tensor([[-0.0726,  1.2745,  1.5667,  1.9381,  1.2215,  1.0115,  0.9610,  2.9072,-0.4534,  0.1668,  0.6210,  3.8220,  3.3079,  0.7912,  1.0864]])
        self.salidaFinal = None
        self.salidaOculta = None
        self.entrada = None
        self.etiqueta = None
        #self.bias_oculto : torch.Tensor = torch.rand(self.K,1)
        self.bias_oculto = torch.Tensor([[-5.2498]])
        #self.bias_entrada : torch.Tensor = torch.rand(self.M,1)
        self.bias_entrada = torch.Tensor([[-3.2510], [-2.7887], [-2.5243], [-2.8151], [-2.7814], [-2.7831], [-2.4752], [-2.9565], [-3.6106], [-3.1965], [-2.2614], [-2.7420], [-2.7473], [-2.7593], [-2.8234]])
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
    #red.entrenarRed(2000, i, t)
    red.evaluarMuesta([0,1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))
    # print("pesos_oe ",red.pesos_oe)
    # print("pesos_so ",red.pesos_so)
    # print("bias_entrada", red.bias_entrada)
    # print("bias_oculto", red.bias_oculto)
    red.graficarError()