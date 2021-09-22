"""First implementation of neuron network"""

from torch.utils.data import Dataset
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    

    def forward(self, x):
        return self.linear(x)



class DataRegression(Dataset):...



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


    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

        """Clear gradient buffers because we don't want any
        gradient from previous epoch to carry forward, dont 
        want to cummulate
        """
        optimizer.zero_grad()

        # get the output from the model, given the inputs
        outputs = model(inputs)


        # get the loss for the predicted output
        loss = criterion(outputs, labels)
        # print(loss)
        
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        # print(f'epochs {epoch}, loss {loss.item()}')

    with torch.no_grad(): # we don't need gradients in the testing phase
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        print(predicted)
    
    plt.clf()
    plt.plot(x_train, y_train, "go", label="True data", alpha=0.5)
    plt.plot(x_train, predicted, '--', label="predictions", alpha=0.5)
    plt.legend(loc='best')
    plt.show()
if __name__ == "__main__":
    main()