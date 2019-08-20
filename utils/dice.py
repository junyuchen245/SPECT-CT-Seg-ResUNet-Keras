import numpy as np

def dice_coef(y_pred, y_true, num_labels):
    dice = []
    #print y_true.shape
    for i in range(num_labels):
        y_true_tmp = y_true[:,:,i].ravel()
        y_pred_tmp = y_pred[:,:,i].ravel()
        intersection = np.sum(y_true_tmp * y_pred_tmp)
        sum1 = np.sum(y_true_tmp)
        sum2 = np.sum(y_pred_tmp)
        try:
            dice.append((2. * intersection) / (sum1 + sum2))
        except:
            dice.append(1)  # label does not exist in both means perfect match
    return np.array(dice)

def portion_wrong_image(y_true, y_pred):
    """In each image, computes the portion of pixels with labels not in the ground truth.
    Returns the average from all images.
    :param list/numpy.array y_true: ground truth. Can be bhw or bdhw with or without last channel = 1.
    :param list/numpy.array y_pred: prediction. Can be bhw or bdhw with or without last channel = 1.
    :return numpy.array: the average portion of pixels with labels not in the ground truth.
    """

    # Standardize input formats
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Reshape to ravel each image, (number of images, number of pixels)
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)
    assert y_true.shape == y_pred.shape

    # Computes the portion of pixels with labels not in the ground truth.
    unique_true = [np.unique(y) for y in y_true]  # Unique labels in each y_true image
    wrong_pred = np.array([~np.in1d(y, u) for u, y in izip(unique_true, y_pred)]).mean()

    return wrong_pred


def mean_square_error(y_pred, y_true):
    diff = y_pred - y_true
    mse = np.mean(np.sum(np.power(diff, 2),axis=1))
    return mse