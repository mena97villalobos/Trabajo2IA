import torch


class RedNeural:


    def __init__(self, neuronasPorCapa, alpha, maxPesosRand):
        self.D = neuronasPorCapa[0]
        self.M = neuronasPorCapa[1]
        self.K = neuronasPorCapa[2]
        self.delta_oculta = 0
        self.delta_salida = 0
        self.learningRate = alpha
        self.maxPesosRand = maxPesosRand
        # Inicializando los pesos de manera aleatoria
        # Valor adicional del bias
        self.pesos_oe : torch.Tensor = torch.rand(self.M, self.D + 1)
        self.pesos_so : torch.Tensor = torch.rand(self.K, self.M  + 1)


    def dSigmoid(self, x : torch.Tensor):
        return x * (1 - x)


    def calcularPasadaAdelante(self, entrada):
        # Añadiendo elemento para que se conserve el bias
        entrada += [[1]]
        entrada = torch.Tensor(entrada)
        salidaOculta = torch.mm(self.pesos_oe, entrada)
        salidaOculta : torch.Tensor = torch.sigmoid(salidaOculta)
        # Añadiendo elemento para que se conserve el bias
        salidaOculta = torch.cat((salidaOculta, torch.Tensor([[1]])), 0)
        salidaFinal = torch.mm(self.pesos_so, salidaOculta)
        salidaFinal = torch.sigmoid(salidaFinal)
        return salidaFinal


    def calcularPasadaAtras(self, entrada, etiqueta):
        entrada += [[1]]
        entrada = torch.Tensor(entrada)
        etiqueta = torch.Tensor(etiqueta)
        salidaOculta = torch.mm(self.pesos_oe, entrada)
        salidaOculta: torch.Tensor = torch.sigmoid(salidaOculta)
        # Añadiendo elemento para que se conserve el bias
        salidaOculta = torch.cat((salidaOculta, torch.Tensor([[1]])), 0)
        salidaFinal = torch.mm(self.pesos_so, salidaOculta)
        salidaFinal = torch.sigmoid(salidaFinal)
        self.delta_salida = (salidaFinal - etiqueta) * (salidaFinal * (1 - salidaFinal))
        self.delta_oculta = torch.mm(self.pesos_so.t(), self.delta_salida) * (salidaOculta*(1-salidaOculta))


    def evaluarClasificacionesErroneas(self, X, T):
        pass


    def entrenarRed(self, numIteraciones, X, T):
        for epoch in range(numIteraciones):
            for data in range(len(X)):
                self.calcularPasadaAtras(X[data].copy(), T[data].copy())
                self.actualizarPesosSegunDeltas()
            self.evaluarClasificacionesErroneas(X, T)
            print("Epoch {} \n Pesos Entrada a Oculta {} \n Pesos Oculta a Salida {}".format(epoch, self.pesos_oe, self.pesos_so))


    def evaluarMuesta(self, x):
        return self.calcularPasadaAdelante(x)



    def actualizarPesosSegunDeltas(self):
        aux1 = self.learningRate * self.delta_salida
        aux2 = (self.learningRate * self.delta_oculta).t()
        self.pesos_so += aux1
        self.pesos_oe += aux2
        pass


if __name__ == "__main__":
    red = RedNeural([2, 2, 1], 0.4, 100)
    i = [[[1], [0]], [[0], [0]], [[0], [1]], [[1], [1]]]
    t = [[0], [1], [0], [1]]
    # El copy se usa para pasarlo por valor y no referencia y que no se modifique el i
    red.entrenarRed(15, i, t)
    salida = red.calcularPasadaAdelante([[1], [0]])
    print("Salida de la pasada hacia adelante: {}".format(salida))