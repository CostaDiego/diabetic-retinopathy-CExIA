from keras.preprocessing.image import ImageDataGenerator

class CustomImageDataGenerator(ImageDataGenerator):
    """Generate batches of tensor image data with real-time data augmentation.

   The data will be looped over (in batches).

  Arguments:
      featurewise_center: Boolean.
          Set input mean to 0 over the dataset, feature-wise.
      samplewise_center: Boolean. Set each sample mean to 0.
      featurewise_std_normalization: Boolean.
          Divide inputs by std of the dataset, feature-wise.
      samplewise_std_normalization: Boolean. Divide each input by its std.
      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
      zca_whitening: Boolean. Apply ZCA whitening.
      rotation_range: Int. Degree range for random rotations.
      width_shift_range: Float, 1-D array-like or int
          - float: fraction of total width, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-width_shift_range, +width_shift_range)`
          - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `width_shift_range=[-1, 0, +1]`,
              while with `width_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      height_shift_range: Float, 1-D array-like or int
          - float: fraction of total height, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-height_shift_range, +height_shift_range)`
          - With `height_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `height_shift_range=[-1, 0, +1]`,
              while with `height_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      brightness_range: Tuple or list of two floats. Range for picking
          a brightness shift value from.
      shear_range: Float. Shear Intensity
          (Shear angle in counter-clockwise direction in degrees)
      zoom_range: Float or [lower, upper]. Range for random zoom.
          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
      channel_shift_range: Float. Range for random channel shifts.
      fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
          Default is 'nearest'.
          Points outside the boundaries of the input are filled
          according to the given mode:
          - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          - 'nearest':  aaaaaaaa|abcd|dddddddd
          - 'reflect':  abcddcba|abcd|dcbaabcd
          - 'wrap':  abcdabcd|abcd|abcdabcd
      cval: Float or Int.
          Value used for points outside the boundaries
          when `fill_mode = "constant"`.
      horizontal_flip: Boolean. Randomly flip inputs horizontally.
      vertical_flip: Boolean. Randomly flip inputs vertically.
      rescale: rescaling factor. Defaults to None.
          If None or 0, no rescaling is applied,
          otherwise we multiply the data by the value provided
          (after applying all other transformations).
      preprocessing_function: function that will be applied on each input.
          The function will run after the image is resized and augmented.
          The function should take one argument:
          one image (Numpy tensor with rank 3),
          and should output a Numpy tensor with the same shape.
      data_format: Image data format,
          either "channels_first" or "channels_last".
          "channels_last" mode means that the images should have shape
          `(samples, height, width, channels)`,
          "channels_first" mode means that the images should have shape
          `(samples, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
      dtype: Dtype to use for the generated arrays.
  """
  
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 dtype='float32'):

        super().__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            data_format=data_format,
            validation_split=validation_split,
            dtype=dtype)