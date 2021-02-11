from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import multi_gpu_model

LOG_DIR_FOLDER = './Graph'
EARLY_STOP_PATIENCE = 4

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

def callbacks(
    earlyStop_patience = EARLY_STOP_PATIENCE,
    tensorBoard_logDir = LOG_DIR_FOLDER
    ):
    """
    Define the callbacks to be used on the train.

    Parameters:
        earlyStop_patience: int - Number of epoch to wait before finish training process.
        tensorBoard_logDir: str - String path to the TensorBoard log directory

    Returns:
        callbacks: dict - Dictionary containing the callbacks
    """

    stop = EarlyStopping(
        monitor = 'val_acc',
        min_delta = 0.001,
        patience = earlyStop_patience,
        verbose = 0,
        mode = 'auto'
    )

    tensor_board = TensorBoard(
        log_dir = tensorBoard_logDir,
        histogram_freq = 0,
        write_graph = True,
        write_images = True
    )

    return {
        'EarlyStopping': stop,
        'TensorBoard': tensor_board
    }