from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import multi_gpu_model

def cnn_model(
    input_shape: tuple = (256,256,3),
    kernel_size:tuple = (8,8),
    nb_filters: int = 32,
    nb_classes: int = 5,
    nb_gpus = None,
    summary = False
    ):
    """
    Define the Convolutional Neural Network.

    Parameters:
        input_shape: (img_row: int, img_col: int, channel: int) - Input shape of the images
        kernel_size: (filt_row: int, filt_col: int) - Initial size of kernel
        nb_filters: int - Initial number of filters
        nb_classes: int - Number of classes for classification
        nb_gpus: int - To use multi-gpus models, set the number os GPUS
        summary: bool - To print or not the model summary

    Returns:
        CNN model
    """

    model = Sequential()

    model.add(
        Conv2D(
            nb_filters,
            kernel_size,
            padding='valid',
            strides=1,
            input_shape=input_shape,
            activation="relu"
            )
        )

    model.add(
        Conv2D(
            nb_filters,
            kernel_size,
            activation="relu"
            )
        )

    model.add(
        Conv2D(
            nb_filters,
            kernel_size,
            activation="relu"
            )
        )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
            )
        )
    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if nb_gpus:
        model = multi_gpu_model(model, gpus=nb_gpus)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

    if summary:
        model.summary()

    return model