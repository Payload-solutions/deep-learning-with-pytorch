#!/usr/bin/python

import torch
import os
import h5py



def tensors_samples():
    points = torch.ones(3, 4)
    points_np = points.numpy()

    print(points)
    print(points_np)


def saving_tensors():
    # saving_tensors()
    
    # viewing tensors saved
    # tensors = torch.load('tensors.t')
    # print(tensors)
    A = torch.randn(4, 3, 4, 2)
    # torch.save(A, 'tensors.t')

    file = h5py.File("ourpoints.h5py", 'w')
    dataset = file.create_dataset('coords', data=A.numpy())
    file.close()



def challenge():
    list_1 = torch.tensor(list(range(9)))
    print(list_1)

    print(list_1.stride())
    print(list_1.t())
    # print(list_1.offset())

    # creating view samples
    tensor_a = torch.tensor(list(range(9)))
    print(tensor_a)
    tensor_b = tensor_a.view(3, 3)
    print(tensor_b)


    tensor_c = tensor_b[1:, 1:]
    print(tensor_c)


    print(torch.sqrt(tensor_a))
    print(torch.cos(tensor_a))




def main():
    """saving_tensors()

    file =  h5py.File("ourpoints.h5py", "r")
    dataset = file["coords"]
    last_points = dataset[-2:]
    print(dataset)
    print(last_points)"""

    challenge()


if __name__ == "__main__":
    main()
