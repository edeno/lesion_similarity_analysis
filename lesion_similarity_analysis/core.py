import time
from itertools import combinations

import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim

animals = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                    111, 112, 113, 114, 115,
                    200, 201, 202, 203, 204, 205, 206, 207, 208])

control = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])

output_size = {
    ("L", "Dorsal"): 363,
    ("L", "Intermediate"): 498,
    ("R", "Dorsal"): 349,
    ("R", "Intermediate"): 500,
}


def evaluateAllSlice(LorR, hippoPart, sigma):
    """

    Parameters
    ----------
    LorR : ("L", "R")
        Brain hemisphere
    hippoPart : ("Dorsal", "Intermediate")
        Sub area of the hippocampus
    sigma : float
        Gaussian standard deviation

    """
    duration = 0
    n_slices = output_size[(LorR, hippoPart)]
    output = np.zeros((animals.size, animals.size, n_slices))

    for i in range(n_slices):
        if i < 10:
            which = f"000{i}"
        elif i < 100:
            which = f"00{i}"
        elif i < 1000:
            which = f"0{i}"
        else:
            which = str(i)
        start = time.time()
        sim = getAllSim(which, LorR, hippoPart, sigma)
        fullSim = completeArray(sim)
        output[:, :, i] = fullSim[:, :]
        end = time.time()

        duration *= i
        duration += (end - start)
        duration /= (i + 1)
        print(duration)
        print(((n_slices - i - 1) * duration) / 60)

    output_filename = f"Desktop/sim_{LorR}_{hippoPart}.txt"
    with open(output_filename, "w+") as file:
        file.write(np.array2string(output))


def getAllSim(which, LorR, hippoPart, sigma):
    sim = np.full((animals.size, animals.size), np.nan)
    for i, j in combinations(range(animals.size), 2):
        sim[i][j] = getSim(str(animals[i]), str(
            animals[j]), which, LorR, hippoPart, sigma)
    return sim


def getSim(an1, an2, which, LorR, hippoPart, sigma):
    im1 = loadImage(an1, which, LorR, hippoPart)
    im2 = loadImage(an2, which, LorR, hippoPart)
    return ssim(im1, im2, gaussian_weights=True, sigma=sigma)


def loadImage(nm, which, LorR, hippoPart):
    imageStr = f"Desktop/FXScoh4/Cropped/{hippoPart}/{LorR}/{nm}/{nm}/{nm}{LorR}{which}.tiff"
    image = io.imread(imageStr)
    return image


def getNN(input):
    NNinfo = np.zeros((input.shape[0], 2))
    for i in range(control.size):
        for j in range(control.size):
            if control[j]:
                if(NNinfo[i][0] < input[i][j]):
                    NNinfo[i][0] = input[i][j]
                    NNinfo[i][1] = control[j]
    return NNinfo


def completeArray(input):
    fullSim = np.zeros((animals.size, animals.size))
    for i in range(animals.size):
        for j in range(animals.size):
            if(i < j):
                fullSim[i][j] = input[j][i]
            else:
                fullSim[i][j] = input[i][j]
    return fullSim
