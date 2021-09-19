"""First model implementation"""

import numpy as np
import torch




def firsts_tensors():

    tensor_img = torch.randn(3, 5, 5) # shape [channels rows, columns]
    tensor_zeros = torch.zeros(3, 5, 5)
    # print(tensor_zeros)
    # print(tensor_img)

    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    # print(weights)

    batch_t = torch.randn(2, 3, 5, 5)
    print(batch_t)
    img_gray_naive = tensor_img.mean(-3)
    batch_gray_naive = batch_t.mean(-3)
    print("\n", img_gray_naive.shape )
    print("\n", batch_gray_naive.shape)

def main():
    firsts_tensors()




if __name__ == "__main__":
    main()
