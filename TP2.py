import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# Primer Cumulo de Datos N = 200 Sigma = [[15, 2], [2, 1]] Mu = [1, 2]
cumulo1 = MultivariateNormal(torch.Tensor([1, 2]),
                             torch.Tensor([[15, 2], [2, 1]]))
datosCumulo1 = cumulo1.sample((200,))
# Segundo Cumulo de Datos N = 200 Sigma = [[3, 1], [1, 2]] Mu = [15, 12]
cumulo2 = MultivariateNormal(torch.Tensor([15, 12]),
                             torch.Tensor([[3, 1], [1, 2]]))
datosCumulo2 = cumulo2.sample((200,))

plt.scatter(datosCumulo1[:, 0], datosCumulo1[:, 1],
            color='red', marker='+', label="Media = [1, 2]")
plt.scatter(datosCumulo2[:, 0], datosCumulo2[:, 1],
            color='blue', marker='o', label="Media = [15, 12]")
plt.legend()
plt.show()


