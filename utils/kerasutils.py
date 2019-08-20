import sys
import numpy as np
from keras import backend as K


"""Different Keras-specific utils."""


def get_image(image, idx):
    """Gets an image from a ND tensor according to the image_data_format. """
    if K.image_data_format() == 'channels_last':
        return image[idx, ..., 0]
    else:
        return image[idx, 0]


def set_image(image, idx, img):
    """Sets an image to a ND tensor according to the image_data_format.
    :param image: ND tensor
    :param idx: index to image
    :param img: replacing image
    """
    if K.image_data_format() == 'channels_last':
        image[idx, ..., 0] = img
    else:
        image[idx, 0] = img


def get_channel_axis():
    """Gets the channel axis."""
    if K.image_data_format() == 'channels_first':
        return 1
    else:
        return -1



def correct_data_format(data):
    """Corrects data format according to K.image_data_format().
    :param numpy.array data: a ND array of image or label data.
    :return: corrected data. No copy is made.
    """
    if K.image_data_format() == 'channels_last' and np.argmin(data.shape[1:]) == 0:
        axes = range(1, data.ndim)
        axes = [0] + axes[1:] + axes[:1]
        data = data.transpose(axes)
    elif K.image_data_format() == 'channels_first' and np.argmin(data.shape[1:]) == data.ndim - 2:
        axes = range(1, data.ndim)
        axes = [0] + axes[-1:] + axes[:-1]
        data = data.transpose(axes)
    return data


def save_model_summary(model, path):
    """Saves model summary to a text file.
    :param model: the model.
    :param path: text file.
    """
    with open(path, 'w') as f:
        current_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = current_stdout
