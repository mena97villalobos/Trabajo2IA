import matplotlib.pyplot as plt
import numpy as np
from math import exp
import torch
from torch.distributions import MultivariateNormal


def dataGenerator(parametros1, parametros2):
    mean1 = parametros1['mean']
    cov1 = parametros1['cov']
    numSample1 = parametros1['sampleSize']
    mean2 = parametros2['mean']
    cov2 = parametros2['cov']
    numSample2 = parametros2['sampleSize']
    func1 = MultivariateNormal(mean1, cov1)
    func2 = MultivariateNormal(mean2, cov2)
    return [func1.sample((numSample1, )), func2.sample((numSample2, ))]


def sigmoid(muestra, pesos):
    return torch.sigmoid(torch.dot(muestra,pesos))


def funcionError(predicciones, targets):
    aux0 = 1 - targets
    aux1 = torch.sum(aux0 * torch.log(1 - predicciones))
    aux2 = torch.sum(targets * torch.log(predicciones))
    return - 1 * (aux1 + aux2)

"""def ajustarValores(i):
    if i == torch.Tensor([0]):
        return 0.01
    else:
        return 0.99
"""
def normalizarPesos(pesos):

    for i in range(0,len(pesos)):
        pesos[i] = (pesos[i] - min(pesos)) / (max(pesos) - min(pesos))

    return pesos

def gradientOpt(pesos, muestras, target, threshold, alpha):
    # Construyendo vector de predicciones para la iteración
    predicciones = torch.Tensor([sigmoid(torch.cat((torch.Tensor([1]),muestra)), pesos) for muestra in muestras]).reshape((muestras.shape[0]))
    error = funcionError(predicciones, target)
    while threshold < error:
        aux = target - predicciones
        aux = torch.mul(muestras, aux.reshape(aux.shape[0],1))
        aux = torch.sum(aux,0)
        # FIXME ANTES ESTABA CON sum(aux,0) creo que se deberia quedar asi como esta.
        pesos = pesos + torch.cat((torch.Tensor([1*alpha]),(alpha * aux)))
        pesos = normalizarPesos(pesos)

        predicciones = torch.Tensor([sigmoid(torch.cat((torch.Tensor([1]),muestra)), pesos) for muestra in muestras]).reshape((muestras.shape[0]))#.reshape((muestras.shape[0], 1))
        error = funcionError(predicciones, target)
        print("Error: ",error)
        print(pesos)


if __name__ == "__main__":
    p1 = {'mean': torch.Tensor([0.10, 0.20]), 'cov': torch.Tensor([[0.5, 0.2], [0.2, 0.1]]), 'sampleSize': 200}
    p2 = {'mean': torch.Tensor([0.150, 0.120]), 'cov': torch.Tensor([[0.3, 0.1], [0.1, 0.2]]), 'sampleSize': 200}
    data = dataGenerator(p1, p2)
    X = data[0]
    Y = data[1]
    # Construyendo vector de muestras
    M = torch.cat((X, Y), 0)
    # Construyendo vector de targets
    t = torch.cat((torch.ones(X.shape[0]), -1 * torch.ones(Y.shape[0])), 0) #t = torch.cat((torch.ones(X.shape), torch.zeros(Y.shape)), 0)
    # Construyendo vector de pesos
    w = torch.ones((1, 3))[0] #torch.ones(1,2)[0]#torch.rand(1,2)[0]
    gradientOpt(w, M, t, 0.01, 0.4)



