import torch


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
        self.pesos_oe = torch.Tensor([[3,2],[2,1]])
        #self.pesos_so : torch.Tensor = torch.rand(self.K, self.M )
        self.pesos_so = torch.Tensor([[4,3]])
        self.salidaFinal = None
        self.salidaOculta = None
        self.entrada = None
        self.etiqueta = None



    def dSigmoid(self, x : torch.Tensor):
        return x * (1 - x)


    def calcularPasadaAdelante(self, entrada):
        # Añadiendo elemento para que se conserve el bias
       # entrada += [[1]]
        #entrada+=[1]
        entrada = torch.Tensor(entrada)
        self.entrada = entrada
        salidaOculta = torch.mm(self.pesos_oe,entrada.reshape(entrada.shape[0],1)) #+1
        #salidaOculta = torch.mm(self.pesos_oe, entrada)
        salidaOculta = salidaOculta.reshape(salidaOculta.shape[0])
        salidaOculta : torch.Tensor = torch.sigmoid(salidaOculta)
        # Añadiendo elemento para que se conserve el bias
        #salidaOculta = torch.cat((salidaOculta, torch.Tensor([1])), 0)
        self.salidaOculta = salidaOculta
        salidaFinal = torch.mm(self.pesos_so, salidaOculta.reshape(salidaOculta.shape[0],1)) #+ 1 #AQUI SA EL BIAS
        salidaFinal = salidaFinal.reshape(salidaFinal.shape[0])
        self.salidaFinal = torch.sigmoid(salidaFinal.reshape(salidaFinal.shape[0]))
        return self.salidaFinal


    def calcularPasadaAtras(self):

        self.delta_salida = (self.salidaFinal - self.etiqueta) * (self.salidaFinal * (1 - self.salidaFinal))

        deltaSalidaAux = self.delta_salida.reshape(1,self.delta_salida.shape[0])

        self.delta_oculta = torch.sum(torch.mm(deltaSalidaAux, self.pesos_so),1) * (self.salidaOculta * (1 - self.salidaOculta))

        self.delta_salida = self.delta_salida.reshape(self.delta_salida.shape[0], 1)
        self.delta_oculta = self.delta_oculta.reshape(self.delta_oculta.shape[0],1)


        # entrada += [[1]]
        # entrada = torch.Tensor(entrada)
        # etiqueta = torch.Tensor(etiqueta)
        # salidaOculta = torch.mm(self.pesos_oe, entrada)
        # salidaOculta: torch.Tensor = torch.sigmoid(salidaOculta)
        # # Añadiendo elemento para que se conserve el bias
        # salidaOculta = torch.cat((salidaOculta, torch.Tensor([[1]])), 0)
        # salidaFinal = torch.mm(self.pesos_so, salidaOculta)
        # salidaFinal = torch.sigmoid(salidaFinal)
        # self.delta_salida = (salidaFinal - etiqueta) * (salidaFinal * (1 - salidaFinal))
        # self.delta_oculta = torch.mm(self.pesos_so.t(), self.delta_salida) * (salidaOculta*(1-salidaOculta))


    def evaluarClasificacionesErroneas(self, X, T):
        pass


    def entrenarRed(self, numIteraciones, X, T):
        for epoch in range(numIteraciones):
            for data in range(len(X)):
                #self.etiqueta = torch.Tensor([T[data].copy()])
                self.etiqueta = torch.Tensor([T[data]])
                self.calcularPasadaAdelante(X[data].copy())
                self.calcularPasadaAtras()
                self.actualizarPesosSegunDeltas()
            self.evaluarClasificacionesErroneas(X, T)
            print("Epoch {} \n Pesos Entrada a Oculta {} \n Pesos Oculta a Salida {}".format(epoch, self.pesos_oe, self.pesos_so))


    def evaluarMuesta(self, x):
        self.calcularPasadaAdelante(x)
        return self.salidaFinal



    def actualizarPesosSegunDeltas(self):

        #Actualizacion de pesos de Capa Oculta a Capa de Salida
        aux1 = self.learningRate *( self.delta_salida * self.salidaOculta)
        self.pesos_so -= aux1

        # Actualizacion de pesos de Capa de Entrada a Capa de Oculta
        aux2 = (self.learningRate *( self.delta_oculta * self.entrada))
        self.pesos_oe -= aux2



if __name__ == "__main__":
    red = RedNeural([2, 2, 1], 0.4, 100)
    i = [[1,1], [1, 0], [0, 1], [0,0]]
    t = [0,1,1,0]
    # El copy se usa para pasarlo por valor y no referencia y que no se modifique el i
    red.entrenarRed(1, i, t)
    red.evaluarMuesta([1, 1])
    salida = red.salidaFinal
    print("Salida de la pasada hacia adelante: {}".format(salida))