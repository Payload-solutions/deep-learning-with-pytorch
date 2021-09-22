"""First implementation of neuron network"""


import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    

    def forward(self, x):
        return self.linear(x)


def main():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)


    input_dim = 1
    output_dim = 1
    learning_rate = 0.01
    epochs = 100

    model = LinearRegression(input_dim, output_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



if __name__ == "__main__":
    main()