"""chapter 4
representation of real world data
in tensors"""

import torch
import imageio
import os


def working_data_():

    batch_size = 3
    batch = torch.zeros(batch_size, 3, 256, 256)
    data_dir = "../data/p1ch4/image-cats/"

    filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']


    for i, filename in enumerate(filenames):
        image_arr = imageio.imread(os.path.join(data_dir, filenames))
        img_t = torch.from_numpy(image_arr)
        img_t = img_t.permute(2, 0, 1)
        img_t = img_t[:3]
        batch[i] = img_t
def main():
    image_arr = imageio.imread(PATH+"/dog.jpg")
    # print(image_arr)
    # print(image_arr.shape)

    """Chaning the layaout: We can use the tensor
    permute method with the old dimensions for eac
    new dimesion to get to an appropiate layaout. Given
    an input tensor HxWxC as obtained previously, we
    get a proper layaout by having channel 2 first then 
    channels 0 an 1"""

    # img = torch.from_numpy(image_arr)
    # out = img.permute(2, 0, 1)

    # print(img)
    # print(out)

    batch_size = 3
    batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
    print(batch)

if __name__ == "__main__":
    main()





