"""chapter 4
representation of real world data
in tensors"""

import torch
import imageio
import os


def working_data():

    batch_size = 3
    batch = torch.zeros(batch_size, 3, 256, 256)
    data_dir = '../data/p1ch4/image-cats/'

    filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']


    for i, filename in enumerate(filenames):
        image_arr = imageio.imread(os.path.join(data_dir, filename))
        img_t = torch.from_numpy(image_arr)
        img_t = img_t.permute(2, 0, 1)
        img_t = img_t[:3]
        batch[i] = img_t
    

    """We mentionaed earlier htat neural networks usually work with
    floating-points tensors as their input. Neural networks exibit
    the best training performance when the input data ranges roughly
    from 0 to 1, or from -1 to 1 (this is an effect of how their
    building blocks are defined).
    
        So tipical thing we'll wanto to do is cast a tensor to floating-
        pong ans normalize the values of the pixels. Casting to floating
        point is easy, but normalizations is trickier, as ir depends on 
        what range of input we decide should lie between 0 and 1 or -1 to 1
        . One posibility is to just divide the values of the pixels by
        255 (the maximun representable number in 8-bit unisged)
    """
    batch = batch.float()
    batch /= 255.0

    n_channels = batch.shape[1]
    # image manipulation available
    print(f" before touch the dataset \n\n{batch}")
    for c in range(n_channels):
        mean = torch.mean(batch[:, c])
        std = torch.std(batch[:, c])
        batch[:, c] = (batch[:, c] - mean) /std
    print(f"after touch the dataset \n\n{batch}")
def main():
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

    # batch_size = 3
    # batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
    # print(batch)

    working_data()

if __name__ == "__main__":
    main()





