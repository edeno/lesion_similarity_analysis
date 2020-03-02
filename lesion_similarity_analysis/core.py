import time

import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim

anKey = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                  111, 112, 113, 114, 115,
                  200, 201, 202, 203, 204, 205, 206, 207, 208])

control = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])


def evaluateAllSlice(LorR, hippoPart, sig):
    fName = "Desktop/sim_" + LorR + "_" + hippoPart + ".txt"
    duration = 0
    if(LorR == "L"):
        if(hippoPart == "Dorsal"):
            output = np.zeros((anKey.size, anKey.size, 363))
        elif(hippoPart == "Intermediate"):
            output = np.zeros((anKey.size, anKey.size, 498))
    elif(LorR == "R"):
        if(hippoPart == "Dorsal"):
            output = np.zeros((anKey.size, anKey.size, 349))
        elif(hippoPart == "Intermediate"):
            output = np.zeros((anKey.size, anKey.size, 500))
    howMany = output.shape[2]
    for i in range(howMany):
        if(i < 10):
            which = "000" + str(i)
        elif(i < 100):
            which = "00" + str(i)
        elif(i < 1000):
            which = "0" + str(i)
        else:
            which = str(i)
        start = time.time()
        sim = getAllSim(which, LorR, hippoPart, sig)
        fullSim = completeArray(sim)
        output[:, :, i] = fullSim[:, :]
        end = time.time()
        duration *= i
        duration += (end - start)
        duration /= (i + 1)
        print(duration)
        print(((howMany - i - 1) * duration) / 60)
    f = open(fName, "w+")
    f.write(np.array2string(output))
    f.close


def getAllSim(which, LorR, hippoPart, sig):
    sim = np.zeros((anKey.size, anKey.size))
    for i in range(anKey.size):
        for j in range(anKey.size):
            if(i > j):
                sim[i][j] = getSim(str(anKey[i]), str(
                    anKey[j]), which, LorR, hippoPart, sig)
            else:
                sim[i][j] = np.nan
    return sim


def getSim(an1, an2, which, LorR, hippoPart, sig):
    im1 = loadImage(an1, which, LorR, hippoPart)
    im2 = loadImage(an2, which, LorR, hippoPart)
    return ssim(im1, im2, gaussian_weights=True, sigma=sig)


def loadImage(nm, which, LorR, hippoPart):
    imageStr = "Desktop/FXScoh4/Cropped/" + hippoPart + "/" + \
        LorR + "/" + nm + "/" + nm + LorR + which + ".tif"
    image = io.imread(imageStr)
    return image


def getNN(input):
    NNinfo = np.zeros((input.shape[0], 2))
    for i in range(control.size):
        for j in range(control.size):
            if(control[j]):
                if(NNinfo[i][0] < input[i][j]):
                    NNinfo[i][0] = input[i][j]
                    NNinfo[i][1] = control[j]
    return NNinfo


def completeArray(input):
    fullSim = np.zeros((anKey.size, anKey.size))
    for i in range(anKey.size):
        for j in range(anKey.size):
            if(i < j):
                fullSim[i][j] = input[j][i]
            else:
                fullSim[i][j] = input[i][j]
    return fullSim
