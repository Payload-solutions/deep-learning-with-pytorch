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
    # print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))

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

    data_mean = torch.mean(data, dim=0)    
    data_var = torch.var(data, dim=0)

    """In this case, dim=0 indicates that the reduction
    is preformed along dimension 0. At this point, we can
    normalize the data y substracting the mean and dividing
    by the standar deviation, which helps with the learning
    process
    """

    data_normalized = (data - data_mean)/torch.sqrt(data_var)

    """Next, let's start to look at the data with an eye
    to seeing if there is an easy way to tell good and bad
    wines apart at a glance. First, we're going to determine
    which rows in target correspong to a score less that or 
    equal to 3:
    """

    bad_indexes = target <= 3
    # print(f"shape: \n{bad_indexes.shape} \n dtype: \n{bad_indexes.dtype} \n sum: \n{bad_indexes.sum()}")

    """Note that only 20 of the bad_indexes antries are set to
    True! By using a feature in Pytorch called advances indexing,
    we can use a tensor with data type torch.bool to index data
    tensor. This will essentially filter data to be inly items 
    (or rows) corresponding to True in the indexing tensor.
    The bad_indexes tensor has the same shape as target, with
    vaues of False or True depending on the outcome of the 
    comparison between our threshold abd each element in the
    original target tensor:"""

    bad_data = data[bad_indexes]
    # print(bad_data.shape)

    """Note that the new bad_data tensor has 20 rows, the same
    as the number of rows with True in the bad_indexes tensor.
    It retains all 11 columns. Now we can start to get information
    about wines grouped into good, middling, and bad categories.
    let's take the .mean() of each column
    """

    bad_data = data[target <= 3]
    mid_data = data[(target > 3) & (target < 7)]
    good_data = data[target >= 7]

    bad_mean = torch.mean(bad_data, dim=0)
    mid_mean = torch.mean(mid_data, dim=0)
    good_mean = torch.mean(good_data, dim=0)


    for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
        print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))


def main():
    one_hot_encoding()


if __name__ == "__main__":
    main()