"""First model implementation"""

import numpy as np
import torch
from typing import Tuple
from random import randint


def stride_sample(tensor: torch.tensor, 
    position: Tuple[int]):
    
    r, c = position[0], position[1]
    target = None
    try:
        r_count = 0
        for x in tensor:
            c_count = 0
            for y in x:
                if r_count == r and c_count == r:
                    target = y.item()
                    raise StopIteration
                c_count += 1
            r_count += 1
    except StopIteration:
        pass
    
    print(target)
    stride_1 = None
    while True:
        stride_1 = randint(0, tensor.shape[0])
        
        if stride_1*position[0] + position[1] == target:
            break
    
    return tuple(stride_1, 1)
    
    
def firsts_tensors():

    # tensor_img = torch.randn(3, 5, 5) # shape [channels rows, columns]
    # tensor_zeros = torch.zeros(3, 5, 5)
    # # print(tensor_zeros)
    # # print(tensor_img)

    # weights = torch.tensor([0.2126, 0.7152, 0.0722])
    # # print(weights)

    # batch_t = torch.randn(2, 3, 5, 5)
    # print(batch_t)
    # img_gray_naive = tensor_img.mean(-3)
    # batch_gray_naive = batch_t.mean(-3)
    # print("\n", img_gray_naive.shape )
    # print("\n", batch_gray_naive.shape)

    # tensor_stride = torch.tensor([1, 2, 3, 4])
    # print(tensor_stride.stride())

    a = torch.rand(5, 6)
    print(a)
    print(a.stride())
    print(stride_sample(a, (2, 1)))

def main():
    firsts_tensors()




if __name__ == "__main__":
    main()
