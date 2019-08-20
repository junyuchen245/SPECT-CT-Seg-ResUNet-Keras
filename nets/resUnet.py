from keras.models import Model
#from keras.layers import Input, Reshape, Dropout, Activation, Permute, Concatenate, GaussianNoise, Add
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *
import numpy as np
from nets.custom_losses import exp_dice_loss

"""Building Res-U-Net."""


def ResUnet(pretrained_weights = None, input_size = (192,192,1)):

    """ first encoder for spect image """
    input_seg = Input(input_size)
    input_segBN = BatchNormalization()(input_seg)

    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_segBN)
    conv1_spect = BatchNormalization()(conv1_spect)
    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_spect)
    conv1_spect = BatchNormalization(name='conv_spect_32')(conv1_spect)
    conv1_spect = Add()([conv1_spect, input_segBN])
    pool1_spect = MaxPool2D(pool_size=(2, 2))(conv1_spect)


    conv2_spect_in = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_spect)
    conv2_spect_in = BatchNormalization()(conv2_spect_in)
    conv2_spect = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_spect_in)
    conv2_spect = BatchNormalization(name='conv_spect_64')(conv2_spect)
    conv2_spect = Add()([conv2_spect, conv2_spect_in])
    pool2_spect = MaxPool2D(pool_size=(2, 2))(conv2_spect)

    conv3_spect_in = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_spect)
    conv3_spect_in = BatchNormalization()(conv3_spect_in)
    conv3_spect = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_spect_in)
    conv3_spect = BatchNormalization(name='conv_spect_128')(conv3_spect)
    conv3_spect = Add()([conv3_spect, conv3_spect_in])
    pool3_spect = MaxPool2D(pool_size=(2, 2))(conv3_spect)

    conv4_spect_in = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_spect)
    conv4_spect_in = BatchNormalization()(conv4_spect_in)
    conv4_spect = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_spect_in)
    conv4_spect = BatchNormalization(name='conv_spect_256')(conv4_spect)
    conv4_spect = Add()([conv4_spect, conv4_spect_in])
    drop4_spect = Dropout(0.5)(conv4_spect)
    pool4_spect = MaxPool2D(pool_size=(2, 2))(drop4_spect)

    """ second encoder for ct image """
    input_ct = Input(input_size)
    input_ctBN = BatchNormalization()(input_ct)

    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_ctBN)
    conv1_ct = BatchNormalization()(conv1_ct)
    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_ct)
    conv1_ct = BatchNormalization(name='conv_ct_32')(conv1_ct)
    conv1_ct = Add()([conv1_ct, input_ctBN])
    pool1_ct = MaxPool2D(pool_size=(2, 2))(conv1_ct) #192x192

    conv2_ct_in = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ct)
    conv2_ct_in = BatchNormalization()(conv2_ct_in)
    conv2_ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ct_in)
    conv2_ct = BatchNormalization(name='conv_ct_64')(conv2_ct)
    conv2_ct = Add()([conv2_ct, conv2_ct_in])
    pool2_ct = MaxPool2D(pool_size=(2, 2))(conv2_ct) #96x96

    conv3_ct_in = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ct)
    conv3_ct_in = BatchNormalization()(conv3_ct_in)
    conv3_ct = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ct_in)
    conv3_ct = BatchNormalization(name='conv_ct_128')(conv3_ct)
    conv3_ct = Add()([conv3_ct, conv3_ct_in])
    pool3_ct = MaxPool2D(pool_size=(2, 2))(conv3_ct) #48x48

    conv4_ct_in = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_ct)
    conv4_ct_in = BatchNormalization()(conv4_ct_in)
    conv4_ct = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_ct_in)
    conv4_ct = BatchNormalization(name='conv_ct_256')(conv4_ct)
    conv4_ct = Add()([conv4_ct, conv4_ct_in])
    drop4_ct = Dropout(0.5)(conv4_ct)
    pool4_ct = MaxPool2D(pool_size=(2, 2))(drop4_ct) #24x24 

    conv5_ct_in = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_ct)
    conv5_ct_in = BatchNormalization()(conv5_ct_in)
    conv5_ct = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_ct_in)
    conv5_ct = BatchNormalization(name='conv_ct_512')(conv5_ct)
    conv5_ct = Add()([conv5_ct, conv5_ct_in])
    conv5_ct = Dropout(0.5)(conv5_ct)
    #pool5_ct = MaxPool2D(pool_size=(2, 2))(conv5_ct) #12x12

    conv5_spect_in = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_spect)
    conv5_spect_in = BatchNormalization()(conv5_spect_in)
    conv5_spect = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_spect_in)
    conv5_spect = BatchNormalization(name='conv_spect_512')(conv5_spect)
    conv5_spect = Add()([conv5_spect, conv5_spect_in])
    conv5_spect = Dropout(0.5)(conv5_spect)
    #pool5_spect = MaxPool2D(pool_size=(2, 2))(conv5_spect)

    merge5_cm = concatenate([conv5_spect, conv5_ct], axis=3) #12x12

    up7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(merge5_cm)) #24x24
    up7_cm = BatchNormalization()(up7_cm)
    merge7_cm = concatenate([drop4_ct, drop4_spect, up7_cm], axis=3)  # cm: cross modality
    merge7_cm = BatchNormalization()(merge7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_cm)
    conv7_cm_in = BatchNormalization()(conv7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_cm_in)
    conv7_cm = BatchNormalization(name='decoder_conv_256')(conv7_cm)
    conv7_cm = Add()([conv7_cm, conv7_cm_in])

    up8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_cm))
    up8_cm = BatchNormalization()(up8_cm)
    merge8_cm = concatenate([conv3_ct, conv3_spect, up8_cm], axis=3)  # cm: cross modality
    merge8_cm = BatchNormalization()(merge8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_cm)
    conv8_cm_in = BatchNormalization()(conv8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_cm_in)
    conv8_cm = BatchNormalization(name='decoder_conv_128')(conv8_cm)
    conv8_cm = Add()([conv8_cm, conv8_cm_in])

    up9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_cm))
    up9_cm = BatchNormalization()(up9_cm)
    merge9_cm = concatenate([conv2_ct, conv2_spect, up9_cm], axis=3)  # cm: cross modality
    merge9_cm = BatchNormalization()(merge9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_cm)
    conv9_cm_in = BatchNormalization()(conv9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_cm_in)
    conv9_cm = BatchNormalization(name='decoder_conv_64')(conv9_cm)
    conv9_cm = Add()([conv9_cm, conv9_cm_in])

    up10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9_cm))
    up10_cm = BatchNormalization()(up10_cm)
    merge10_cm = concatenate([conv1_ct, conv1_spect, up10_cm], axis=3)  # cm: cross modality
    merge10_cm = BatchNormalization()(merge10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_cm)
    conv10_cm_in = BatchNormalization()(conv10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_cm_in)
    conv10_cm = BatchNormalization(name='decoder_conv_32')(conv10_cm)
    conv10_cm = Add()([conv10_cm, conv10_cm_in])

    conv11_cm = Conv2D(filters=6, kernel_size=3, activation='relu', padding='same')(conv10_cm)
    conv11_cm = BatchNormalization()(conv11_cm)
    out = Conv2D(filters=3, kernel_size=1, activation='softmax', padding='same', name='segmentation')(conv11_cm)
    # if channels_first:
    #     new_shape = tuple(range(1, K.ndim(x)))
    #     new_shape = new_shape[1:] + new_shape[:1]
    #     x = Permute(new_shape)(x)

    image_size = tuple((192, 192))

    x = Reshape((np.prod(image_size), 3))(out)

    model = Model(inputs=[input_ct, input_seg], outputs=x)
    model.compile(optimizer=Adam(lr=1e-3), loss=exp_dice_loss(exp=1.0))

    return model
