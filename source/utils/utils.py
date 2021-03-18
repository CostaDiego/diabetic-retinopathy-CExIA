from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def split_data(dataFrameY):
    pass