from keras import backend as K


def dice_coef(y_true, y_pred):
    """Computes Dice coefficients with additive smoothing.
    :param y_true: one-hot tensor multiplied by label weights (batch size, number of pixels, number of labels).
    :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
    :return: Dice coefficients (batch size, number of labels).
    """
    smooth = 1.0
    y_true = K.cast(K.not_equal(y_true, 0), K.floatx())  # Change to binary
    intersection = K.sum(y_true * y_pred, axis=1)  # (batch size, number of labels)
    union = K.sum(y_true + y_pred, axis=1)  # (batch size, number of labels)
    return (2. * intersection + smooth) / (union + smooth)  # (batch size, number of labels)


def generalized_dice(y_true, y_pred, exp):
    GD =[]
    print(K.shape(y_pred))
    smooth = 1.0
    for i in range(y_pred.shape[2]):
        y_true_per_label = K.cast(K.not_equal(y_true[:, :, i], 0), K.floatx())  # Change to binary
        y_pred_per_label = y_pred[:, :, i]  # K.cast(K.not_equal(y_pred[:, :, i], 0), K.floatx())  # Change to binary
        weight = K.pow(1/K.sum(y_true_per_label, axis=1), exp)
        intersection = K.sum(y_true_per_label * y_pred_per_label, axis=1)  # (batch size, number of labels)
        union = K.sum(y_true_per_label + y_pred_per_label, axis=1)  # (batch size, number of labels)
        GD.append(weight * (2. * intersection + smooth) / (union + smooth))
    GD_out = K.stack(GD)
    return GD_out



def exp_dice_loss(exp=1.0):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """

    def inner(y_true, y_pred):
        """Computes the average exponential log Dice coefficients as the loss function.
        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average exponential log Dice coefficient.
        """

        dice = dice_coef(y_true, y_pred)
        #dice = generalized_dice(y_true, y_pred, exp)
        dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
        dice = K.pow(-K.log(dice), exp)
        if K.ndim(dice) == 2:
            dice = K.mean(dice, axis=-1)
        return dice

    return inner
