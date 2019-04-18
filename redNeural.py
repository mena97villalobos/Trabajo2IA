import torch


class RedNeural:

    def __init__(self, neuronasPorCapa, alpha, maxPesosRand):
        self.D = neuronasPorCapa[0]
        self.M = neuronasPorCapa[1]
        self.K = neuronasPorCapa[2]
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
        self.pesos_so.requires_grad_()
        self.pesos_oe.requires_grad_()
        entrada += [[1]]
        entrada = torch.Tensor(entrada)
        salidaOculta = torch.mm(self.pesos_oe, entrada)
        salidaOculta: torch.Tensor = torch.sigmoid(salidaOculta)
        salidaOculta = torch.cat((salidaOculta, torch.Tensor([[1]])), 0)
        salidaFinal = torch.mm(self.pesos_so, salidaOculta)
        salidaFinal = torch.sigmoid(salidaFinal)
        # Errores de la capa de salida a la capa oculta
        errorSalida = etiqueta - salidaFinal
        errores_oculta = torch.mm(self.pesos_so.t(), errorSalida)
        gradiente_errores = self.dSigmoid(errores_oculta)
        gradiente_errores = torch.mm(gradiente_errores, errores_oculta)
        gradiente_errores = self.learningRate * gradiente_errores
        # Errores de la capa oculta a la capa de salida

        pass

if __name__ == "__main__":
    red = RedNeural([2, 2, 1], 0.4, 100)
    i = [[1], [0]]
    # El copy se usa para pasarlo por valor y no referencia y que no se modifique el i
    salida = red.calcularPasadaAdelante(i.copy())
    print("Salida de la pasada hacia adelante: {}".format(salida))