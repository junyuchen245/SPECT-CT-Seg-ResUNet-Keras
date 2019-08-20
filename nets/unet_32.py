from keras.models import Model
#from keras.layers import Input, Reshape, Dropout, Activation, Permute, Concatenate, GaussianNoise, Add
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *
import numpy as np
#from utils.kerasutils import get_channel_axis
from nets.custom_losses import exp_dice_loss

"""Building D-Net."""


def Unet(pretrained_weights = None, input_size = (192,192,1)):

    """ first encoder for spect image """
    input_seg = Input(input_size)
    conv1_seg = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_seg)
    conv1_seg = BatchNormalization()(conv1_seg)
    conv1_seg = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_seg)
    conv1_seg = BatchNormalization(name='conv_spect_32')(conv1_seg)
    pool1_seg = MaxPool2D(pool_size=(2, 2))(conv1_seg)

    conv2_seg = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_seg)
    conv2_seg = BatchNormalization()(conv2_seg)
    conv2_seg = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_seg)
    conv2_seg = BatchNormalization(name='conv_spect_64')(conv2_seg)
    pool2_seg = MaxPool2D(pool_size=(2, 2))(conv2_seg)

    conv3_seg = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_seg)
    conv3_seg = BatchNormalization()(conv3_seg)
    conv3_seg = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_seg)
    conv3_seg = BatchNormalization(name='conv_spect_128')(conv3_seg)
    pool3_seg = MaxPool2D(pool_size=(2, 2))(conv3_seg)

    conv4_seg = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_seg)
    conv4_seg = BatchNormalization()(conv4_seg)
    conv4_seg = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_seg)
    conv4_seg = BatchNormalization(name='conv_spect_256')(conv4_seg)
    drop4_seg = Dropout(0.5)(conv4_seg)
    pool4_seg = MaxPool2D(pool_size=(2, 2))(drop4_seg)

    """ second encoder for ct image """
    input_ct = Input(input_size)
    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_ct)
    conv1_ct = BatchNormalization()(conv1_ct)
    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_ct)
    conv1_ct = BatchNormalization(name='conv_ct_32')(conv1_ct)
    pool1_ct = MaxPool2D(pool_size=(2, 2))(conv1_ct) #192x192

    conv2_ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ct)
    conv2_ct = BatchNormalization()(conv2_ct)
    conv2_ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ct)
    conv2_ct = BatchNormalization(name='conv_ct_64')(conv2_ct)
    pool2_ct = MaxPool2D(pool_size=(2, 2))(conv2_ct) #96x96

    conv3_ct = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ct)
    conv3_ct = BatchNormalization()(conv3_ct)
    conv3_ct = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ct)
    conv3_ct = BatchNormalization(name='conv_ct_128')(conv3_ct)
    pool3_ct = MaxPool2D(pool_size=(2, 2))(conv3_ct) #48x48

    conv4_ct = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_ct)
    conv4_ct = BatchNormalization()(conv4_ct)
    conv4_ct = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_ct)
    conv4_ct = BatchNormalization(name='conv_ct_256')(conv4_ct)
    drop4_ct = Dropout(0.5)(conv4_ct)
    pool4_ct = MaxPool2D(pool_size=(2, 2))(drop4_ct) #24x24

    conv5_ct = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_ct)
    conv5_ct = BatchNormalization()(conv5_ct)
    conv5_ct = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_ct)
    conv5_ct = BatchNormalization(name='conv_ct_512')(conv5_ct)
    conv5_ct = Dropout(0.5)(conv5_ct)
    #pool5_ct = MaxPool2D(pool_size=(2, 2))(conv5_ct) #12x12

    conv5_seg = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_seg)
    conv5_seg = BatchNormalization()(conv5_seg)
    conv5_seg = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_seg)
    conv5_seg = BatchNormalization(name='conv_spect_512')(conv5_seg)
    conv5_seg = Dropout(0.5)(conv5_seg)
    #pool5_spect = MaxPool2D(pool_size=(2, 2))(conv5_spect)

    merge5_cm = concatenate([conv5_seg, conv5_ct], axis=3) #12x12

    up7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(merge5_cm)) #24x24
    up7_cm = BatchNormalization()(up7_cm)
    merge7_cm = concatenate([drop4_ct, drop4_seg, up7_cm], axis=3)  # cm: cross modality
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_cm)
    conv7_cm = BatchNormalization()(conv7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_cm)
    conv7_cm = BatchNormalization(name='decoder_conv_256')(conv7_cm)

    up8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_cm))
    up8_cm = BatchNormalization()(up8_cm)
    merge8_cm = concatenate([conv3_ct, conv3_seg, up8_cm], axis=3)  # cm: cross modality
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_cm)
    conv8_cm = BatchNormalization()(conv8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_cm)
    conv8_cm = BatchNormalization(name='decoder_conv_128')(conv8_cm)

    up9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_cm))
    up9_cm = BatchNormalization()(up9_cm)
    merge9_cm = concatenate([conv2_ct, conv2_seg, up9_cm], axis=3)  # cm: cross modality
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_cm)
    conv9_cm = BatchNormalization()(conv9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_cm)
    conv9_cm = BatchNormalization(name='decoder_conv_64')(conv9_cm)

    up10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9_cm))
    up10_cm = BatchNormalization()(up10_cm)
    merge10_cm = concatenate([conv1_ct, conv1_seg, up10_cm], axis=3)  # cm: cross modality
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_cm)
    conv10_cm = BatchNormalization()(conv10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_cm)
    conv10_cm = BatchNormalization(name='decoder_conv_32')(conv10_cm)

    conv11_cm = Conv2D(filters=6, kernel_size=3, activation='relu', padding='same')(conv10_cm)
    out = Conv2D(filters=3, kernel_size=1, activation='softmax', padding='same', name='segmentation')(conv11_cm)
    # if channels_first:
    #     new_shape = tuple(range(1, K.ndim(x)))
    #     new_shape = new_shape[1:] + new_shape[:1]
    #     x = Permute(new_shape)(x)

    image_size = tuple((192, 192))

    x = Reshape((np.prod(image_size), 3))(out)
    #x = Activation('softmax')(x)

    model = Model(inputs=[input_ct, input_seg], outputs=x)
    model.compile(optimizer=Adam(lr=1e-3), loss=exp_dice_loss(exp=1.0))

    return model
