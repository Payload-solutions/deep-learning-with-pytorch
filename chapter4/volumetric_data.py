"""Working with volumetric data"""

from typing import Any
import numpy as np
import imageio
import torch
import pandas as pd
import csv

DIR_PATH = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"


def parsing_wine_dataset() -> None:
    wine_path = "../data/winequality-white.csv"
    wine_quality = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                              skiprows=1)

    """here we just prescribe what the type of the 2D array should be
    (32-bit floating-point), the delimiter used to separated values en 
    each row, and the fact that the first line should not be read since 
    it contains the columns names. Let's check that all the data has been read"""

    col_list = next(csv.reader(open(wine_path), delimiter=';'))
    # print(wine_quality.shape, col_list)

    # and process to convert the NumPy array to Pytorch tensor:
    wineq = torch.from_numpy(wine_quality)

    # print("wineq: \n\n", wineq, "\n\n")

    # representing scores
    # selects all rows and all columns except the last
    data = wineq[:, :-1]
    # print("data: \n\n", data)
    # print(data.shape)

    # selects all rows ans the last column
    target = wineq[:, -1].long()
    # print(target)

    """The other approach is to build a one-hot encoding of the scores:
        that is, encode each of the 10 scores in a vector of 10 elements,
        with all elements set to 0 but one, at a different index for each score.
        This way, a score of 1 could be mapped onto the vector
        (1, 0, 0, 0, 0, 0, 0, 0, 0 ,0), a score of 5 onto (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        and so on. Noe that the fact that the score corresponds to the index of the nonzero
        element is purely incidental: we could be shuffle the assignment, and
        nothing would change from a classifications standpoint.
    112
    """


def volumetric_data_tensors():
    vol_arr = imageio.volread(DIR_PATH, 'DICOM')
    print(vol_arr.shape)

    vol = torch.from_numpy(vol_arr).float()
    vol = torch.unsqueeze(vol, 0)

    print(f"volume shape => {vol.shape}")


def main():
    # volumetric_data_tensors()
    # dataset = pd.read_csv("../data/winequality-white.csv")
    # print(dataset.shape)
    parsing_wine_dataset()


if __name__ == "__main__":
    main()
