import os.path
from glob import glob
from itertools import combinations

import dask
import dask.array as da
import numpy as np
from dask import delayed
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

animals = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                    111, 112, 113, 114, 115,
                    200, 201, 202, 203, 204, 205, 206, 207, 208])

control = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])

SLICE_SIZE = {
    ("L", "Dorsal"): 363,
    ("L", "Intermediate"): 498,
    ("R", "Dorsal"): 349,
    ("R", "Intermediate"): 500,
}


def get_animal_names(filenames, LorR):
    return [os.path.basename(fn).split(LorR)[0] for fn in filenames]


def get_slice_filenames(data_path, hippoPart, LorR, slice_id):
    path = f"{data_path}/{hippoPart}/{LorR}/**/*{LorR}{slice_id:04d}.tif"
    filenames = glob(path)
    return sorted(filenames, key=alphanumeric_key)


def load_images(filenames):
    sample = imread(filenames[0])

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    return da.stack(dask_arrays, axis=0)


def compute_structural_similarity(images, sigma):
    n_animals = images.shape[0]
    result = [
        dask.delayed(ssim)(images[i], images[j],
                           gaussian_weights=True, sigma=sigma)
        for i, j in combinations(range(n_animals), 2)
    ]
    return np.asarray(dask.compute(*result, scheduler="processes"))


def mirror_diagonal(similarity, n_animals):
    output = np.full((n_animals, n_animals), np.nan)
    i, j = zip(*list(combinations(range(n_animals), 2)))
    output[(i, j)] = similarity
    output[(j, i)] = similarity
    return output


def evaluate_slices(data_path, hippoPart, LorR, sigma, output_path=""):
    """

    Parameters
    ----------
    data_path : str
        File path to data directory
    hippoPart : ("Dorsal", "Intermediate")
        Sub area of the hippocampus
    LorR : ("L", "R")
        Brain hemisphere
    sigma : float
        Gaussian standard deviation
    output_path : str
        Where to save the output

    """
    n_slices = SLICE_SIZE[LorR, hippoPart]
    output = []

    for slice_ind in tqdm(range(n_slices), desc="slices"):
        filenames = get_slice_filenames(data_path, hippoPart, LorR, slice_ind)
        animal_names = get_animal_names(filenames, LorR)
        n_animals = len(animal_names)
        images = load_images(filenames)
        similarity = compute_structural_similarity(images, sigma)
        similarity = mirror_diagonal(similarity, n_animals)
        output.append(similarity)

    output = np.stack(output, axis=-1)
    output_filename = os.path.abspath(
        os.path.join(output_path, f"sim_{LorR}_{hippoPart}.txt"))
    with open(output_filename, "w+") as file:
        file.write(np.array2string(output))
