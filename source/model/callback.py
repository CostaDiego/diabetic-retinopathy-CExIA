from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

def earlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
    ):
    """
    Define the  Early Stopping callback to be used on the train.

    Parameters:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.

    Returns:
        Early Stopping callback: The callback to stop when train reaches a plateou
    """

    early_stop = EarlyStopping(
        patience = patience,
        monitor = monitor,
        min_delta = min_delta,
        verbose = verbose,
        mode = mode,
        baseline = baseline,
        restore_best_weights = restore_best_weights
    )

    return early_stop

def tensorBoard(
        log_dir='tensorBoard_logs',
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
        **kwargs
    ):
    """
    Define the TensorBoard callback to be used in training.

    Parameters:
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    Returns:
        TensorBoard callback: The callback to use TensorBoard.
    """

    tensor_board = TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_freq,
        write_graph=write_graph,
        write_images=write_images,
        update_freq=update_freq,
        profile_batch=profile_batch,
        embeddings_freq=embeddings_freq,
        embeddings_metadata=embeddings_metadata,
        **kwargs
    )

    return tensor_board

def callbacks(early_stopping = True, tensor_board = True):
    """
    Return all the modifieds callbacks

    Parameters:
        early_stopping: If includes the EarlyStopping callback. Optional.

        tensor_board: If includes the Tensorboard callback. Optional.

    Returns:
        callbacks: List of callbacks
    """

    callbacks = []

    if early_stopping:
        callbacks.append(earlyStopping())

    if tensor_board:
        callbacks.append(tensorBoard())

    return callbacks