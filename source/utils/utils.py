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

def split_data(dataFrameY, validation_size: float, test_size:float = None, reset_index = True):
    """
    This method split the dataframe in train and validation dataframes,
    if test_size is specified it will also provide a test dataframe.

    If test_size is specified, first split in train and test dataframe
    and then split train into train and validation dataframe. If test_size
    is not specified split just into train and validation dataframe.

    Parameters:
        dataFrameY: The dataframe intended to be splited.

        validation_size: The portion of the original dataframe to be splited
            into the validation dataframe.

        test_size: The portion of the original dataframe to be splited
            into the test dataframe. Optional.

        reset_index: Whether to reset the index or not. Default True. Optional.
    """
    if test_size and isinstance(test_size,float):
        if test_size >= 0.0 and test_size <= 1.0:
            trainY, testY = train_test_split(dataFrameY,test_size = test_size)
            trainY, valY = train_test_split(trainY,test_size = validation_size)

            if reset_index:
                trainY = trainY.reset_index(drop=True)
                valY = valY.reset_index(drop=True)
                testY = testY.reset_index(drop=True)


            return trainY, valY, testY

        else:
            print('Test Size must bem a float: 0.0 <= test_size <= 1.0')
    else:
        trainY, valY = train_test_split(dataFrameY,test_size = validation_size)

        if reset_index:
            trainY = trainY.reset_index(drop=True)
            valY = valY.reset_index(drop=True)

        return trainY, valY

def append_ext(dataframe, column, extension = '.jpeg'):
    dataframe[column]=dataframe[column].apply(lambda x: (x+extension))

    return dataframe