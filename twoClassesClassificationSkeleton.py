import matplotlib.pyplot as plt
import numpy as np
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


def obtenerMuestras(dataset):
    size = len(dataset)
    sizeMuestra = size * 70 // 100
    return dataset[:sizeMuestra], dataset[sizeMuestra:]


class RegresionLogistica:


    def __init__(self, lr=0.01, num_iter=100000):
        self.learning_rate = lr
        self.num_iter = num_iter
        self.pesos = []
        self.hist_error = []


    @staticmethod
    def annadir_bias(X):
        muestrasBias = np.ones((X.shape[0], 1))
        return np.concatenate((muestrasBias, X), axis=1)


    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def f_perdida(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


    def entrenar(self, X, y):
        X = self.annadir_bias(X)
        self.pesos = np.zeros(X.shape[1])
        cont_perdida = 0
        perdida_anterior = 0
        for i in range(self.num_iter):
            z = np.dot(X, self.pesos)
            h = self.sigmoid(z)
            gradiente = np.dot(X.T, (h - y)) / y.size
            self.pesos -= self.learning_rate * gradiente
            z = np.dot(X, self.pesos)
            h = self.sigmoid(z)
            perdida_actual = self.f_perdida(h, y)
            if perdida_actual == perdida_anterior:
                cont_perdida += 1
                if cont_perdida == 10:
                    print("Convergencia en iteración {}, Pesos Actuales {}".format(i, self.pesos))
                    return self.pesos
            perdida_anterior = perdida_actual
            # Almacenando el error para graficarlo
            self.hist_error += [[i, perdida_actual]]
            if i % 1000 == 0:
                print("Iteración {}, Pesos Actuales {}".format(i, self.pesos))


    def graficar_error(self):
        datos = np.array(self.hist_error)
        plt.scatter(datos[:, 0], datos[:, 1], color='red', marker='o', label="Gráfica de Error")
        plt.legend()
        plt.show()


    def predicciones(self, X):
        X = self.annadir_bias(X)
        return self.sigmoid(np.dot(X, self.pesos))


    def evaluarClasificacioneErroneas(self, X, targets):
        predicciones = self.predicciones(X)
        predicciones = [True if i > 0.5 else False for i in predicciones]
        return  [predicciones[i] == targets[i] for i in range(len(predicciones))]


if __name__ == "__main__":
    # Configuración de los parámetros para conseguir datos
    p1 = {'mean': torch.Tensor([0.5, 0.5]), 'cov': torch.Tensor([[1, 0], [0, 1]]), 'sampleSize': 400}
    p2 = {'mean': torch.Tensor([0.5, 0.5]), 'cov': torch.Tensor([[0.1, 0], [0, 0.1]]), 'sampleSize': 400}
    data = dataGenerator(p1, p2)
    # Datos de Entrenamiento
    entrenamientoClase1, pruebasClase1 = obtenerMuestras(data[0])
    entrenamientoClase0, pruebasClase0 = obtenerMuestras(data[1])
    datosEntrenamiento = np.array(torch.cat((entrenamientoClase1, entrenamientoClase0), 0))
    targets = np.array([1] * len(entrenamientoClase1) + [0] * len(entrenamientoClase0))
    model = RegresionLogistica(lr=0.1, num_iter=300000)
    model.entrenar(datosEntrenamiento, targets)
    model.graficar_error()
    # Probando con entrenamiento realizada
    print("Evaluando muestras de prueba")
    predsClase1 = model.evaluarClasificacioneErroneas(pruebasClase1, [1] * len(pruebasClase1))
    cant_errores_c1 = len(pruebasClase1) - sum(predsClase1)
    predsClase0 = model.evaluarClasificacioneErroneas(pruebasClase0, [0] * len(pruebasClase1))
    cant_errores_c0 = len(pruebasClase0) - sum(predsClase0)
    print("Cantidad de errores en la clasificación de clase 1: {} de {} muestras de prueba".format(cant_errores_c1, len(pruebasClase1)))
    print("Cantidad de errores en la clasificación de clase 0: {} de {} muestras de prueba".format(cant_errores_c0, len(pruebasClase0)))
    print("Parámetros resultantes del entrenamiento: {}".format(model.pesos))
    plt.figure(figsize=(10, 6))
    plt.scatter(entrenamientoClase0[:,0], entrenamientoClase0[:, 1], color='blue', label='Datos de Entrenamiento Clase 0')
    plt.scatter(entrenamientoClase1[:, 0], entrenamientoClase1[:, 1], color='red', label='Datos de Entrenamiento Clase 1')
    plt.scatter(pruebasClase1[:,0], pruebasClase1[:,1], color='green', label='Datos de Pruebas Clase 1')
    plt.scatter(pruebasClase0[:, 0], pruebasClase0[:, 1], color='yellow', label='Datos de Pruebas Clase 0')
    plt.legend()
    x1_min, x1_max = datosEntrenamiento[:,0].min(), datosEntrenamiento[:,0].max(),
    x2_min, x2_max = datosEntrenamiento[:,1].min(), datosEntrenamiento[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predicciones(grid).reshape(xx1.shape)
    # Graficando la línea de regresión resultante
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
    plt.show()
