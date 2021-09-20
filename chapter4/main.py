"""chapter 4
representation of real world data
in tensors"""

import torch
import imageio

PATH = "images"




def main():
    image_arr = imageio.imread(PATH+"/dog.jpg")
    print(image_arr)
    print(image_arr.shape)


if __name__ == "__main__":
    main()





