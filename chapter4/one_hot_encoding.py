"""Generate One hot encoding in the datatensors"""


import torch
import csv
import imageio
import numpy as np
import pandas



def one_hot_encoding():
    wine_path = "../data/winequality-white.csv"
    wine_quality = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                              skiprows=1)
    
    col_list = next(csv.reader(open(wine_path), delimiter=';'))
    wineq = torch.from_numpy(wine_quality)

    data = wineq[:, :-1]
    target = wineq[:, -1].long()

    target_onehot = torch.zeros(target.shape[0], 10)
    
    # print(target_onehot)
    print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))

    """Let's see what scatter_ does. First, we notice that its
    name ends with an underscore. As you learned in the previous
    chapter, this is a conventions in Pytorch that indicates the
    method will not return a new tensor, but will instead modify
    the tensor in place. The arguments for scatter_ are as follows:
    
        >> The dimension along which the following two arguments
            are specified
        >> A column tensor indicating the indeces of the elemens to scatter
        >> A tensor containing the elements to scatter or a single scalar
            to scatter (1, in this case)
    """


def main():
    one_hot_encoding()


if __name__ == "__main__":
    main()