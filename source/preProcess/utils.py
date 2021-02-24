from os import path, makedirs
from PIL import Image
import numpy as np

def make_dir(directory:str):
    """
    Check if the directory exists, if don't, create the directory

    Parameters:
        path: str - Path to the folder
    """

    if not path.exists(directory):
        makedirs(directory)

def image_to_array(img_path:str):
    """
    Convert image to numpy array

    Paramenter:
        img_path: str - Path to the image

    Returns:
        array: numpy.ndarray - The array converted from the image
    """

    image = Image.open(str(img_path))

    return np.array(image)

def save_array(fileName, np_object):
    """
    Save the array (or numpy data object) to a file.

    Parameters:
        fileName: str - The file name with the .npy extension.
        np_object: numpy.ndarray - Numpy array to be saved.
    """
    np.save(fileName, np_object)