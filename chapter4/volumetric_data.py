"""Working with volumetric data"""


import imageio
import torch



DIR_PATH = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"

def volumetric_data_tensors():
    vol_arr = imageio.volread(DIR_PATH, 'DICOM')
    print(vol_arr.shape)

    vol = torch.from_numpy(vol_arr).float()
    vol = torch.unsqueeze(vol, 0)

    print(f"volumen shape => {vol.shape}")


def main():
    volumetric_data_tensors()



if __name__ == "__main__":
    main()