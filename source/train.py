import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import normalize, np_utils, multi_gpu_model, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.utils import class_weight


def train():
    pass