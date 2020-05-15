import os.path
import re
from glob import glob
from itertools import combinations

import dask
import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from tqdm.auto import tqdm

from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from skimage.metrics import structural_similarity as ssim

SLICE_SIZE = {
    ("dorsal"): 602,
    ("intermediate"): 755,
}


def get_animal_name(filename, pattern_matcher):
    animal, hemisphere, _ = pattern_matcher.findall(
        os.path.basename(filename))[0]
    return f"{animal}-{hemisphere}"


def get_animal_names(filenames):
    pattern_matcher = re.compile(r"([\w]+)(L|R)([\d]+).tif")
    return [get_animal_name(fn, pattern_matcher) for fn in filenames]


def get_slice_filenames(data_path, hippoPart, slice_id):
    path = os.path.join(data_path, hippoPart, "**", f"*{slice_id:04d}.tif")
    filenames = glob(path, recursive=True)
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


def evaluate_slices(data_path, hippoPart, sigma, output_path=""):
    """

    Parameters
    ----------
    data_path : str
        File path to data directory
    hippoPart : ("Dorsal", "Intermediate")
        Sub area of the hippocampus
    sigma : float
        Gaussian standard deviation
    output_path : str
        Where to save the output

    """
    n_slices = SLICE_SIZE[hippoPart]
    output = []

    for slice_ind in tqdm(range(n_slices), desc="slices"):
        # Get same slice for each animal
        filenames = get_slice_filenames(data_path, hippoPart, slice_ind)
        animal_names = get_animal_names(filenames)
        n_animals = len(animal_names)

        # Load images
        images = load_images(filenames)

        # Find similarity between images across animals
        similarity = compute_structural_similarity(images, sigma)
        similarity = mirror_diagonal(similarity, n_animals)
        output.append(similarity)

    output = xr.DataArray(
        np.stack(output, axis=-1),
        dims=["animal1", "animal2", "slice"],
        coords={"animal1": animal_names,
                "animal2": animal_names,
                "slice": range(output.shape[-1])
                },
        name="similarity")
    output_filename = os.path.abspath(
        os.path.join(output_path, f"sim_{hippoPart}.csv"))
    output.to_dataframe().to_csv(output_filename)
