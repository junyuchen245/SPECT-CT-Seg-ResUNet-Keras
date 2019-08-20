import os
import sys
sys.path.append('/data/jchen/anaconda3/lib/python3.7/site-packages')
sys.path.append('/netscratch/jchen/boneSegResUnet/')
import numpy as np
import keras
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import shutil
import time
from utils.image_reading import load_image_from_folder, load_test_from_folder
from nets.resUnet import ResUnet
from nets.custom_losses import exp_dice_loss
from utils.image_augmentation import ImageDataGenerator
from utils.dice import dice_coef
import gc
from keras.utils import to_categorical
from scipy.misc import imsave, imread
import math
import matplotlib.pyplot as plt

#print('backend')
#print(K.backend())


if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    print('GPU Setup done')

input_path = '/netscratch/jchen/SPECTData_sub/'
input_path_test = '/netscratch/jchen/patient_test/'
output_path = ('/netscratch/jchen/boneSegResUnet/outputs/')
output_image_path = output_path + 'images/'
output_feats_path = output_path + 'features/'
output_test_path  = output_path + 'patient_test/'
output_model_path = output_path + 'model/'

if output_path is not None and not os.path.exists(output_path):
    os.makedirs(output_path)

# ---------------------
#  Load Image Data
# ---------------------

train_portion = 0.8
valid_portion = 0.2

image_array, label_array = load_image_from_folder(input_path, (192, 192), HE=False, Truc=False, Aug=False)
image_test               = load_test_from_folder(input_path_test, (192, 192), HE=False, Truc=False, Aug=False)
#image_train, label_train = load_image_from_folder((input_path+'train/'), (192, 192), HE=False, Truc=False, Aug=False)
#image_valid, label_valid = load_image_from_folder((input_path+'valid/'), (192, 192), HE=False, Truc=False, Aug=False)
print("image_array, label_array generation done")

image_train = image_array[0:int(train_portion*len(image_array)),:,:]
label_train = label_array[0:int(train_portion*len(image_array)),:,:]
image_valid = image_array[int(train_portion*len(image_array)):len(image_array),:,:]
label_valid = label_array[int(train_portion*len(image_array)):len(image_array),:,:]

# This gives all the class label values present in the train label data
unique_labels = np.unique(label_valid)
print('unique_labels: ')
print(unique_labels)

# Correct data format
image_train = np.expand_dims(image_train, axis=3)
label_train = np.expand_dims(label_train, axis=3)
image_valid = np.expand_dims(image_valid, axis=3)
label_valid = np.expand_dims(label_valid, axis=3)
image_test  = np.expand_dims(image_test, axis=3)

if K.image_data_format() == 'channels_last':
    image_height = label_train.shape[1]
    image_width = label_train.shape[2]
else:
    image_height = label_train.shape[2]
    image_width = label_train.shape[3]
pixels_per_image = image_height * image_width

# Training arguments
num_labels = 3
batch_size = 35
n_batches_per_epoch = 200
n_epochs = 500

if output_path is not None and not os.path.exists(output_path):
    os.makedirs(output_path)

# ---------------------
#  Initialize Networks
# ---------------------
net = ResUnet()
print(net.summary())
segmentation_model = Model(inputs=net.input, outputs=net.get_layer('segmentation').output)
#sys.exit(0)

activation_model = Model(inputs=net.input, outputs=[net.get_layer('conv_spect_32').output, net.get_layer('conv_spect_64').output,
                                                    net.get_layer('conv_spect_128').output, net.get_layer('conv_spect_256').output,
                                                    net.get_layer('conv_spect_512').output, net.get_layer('conv_ct_32').output,
                                                    net.get_layer('conv_ct_64').output, net.get_layer('conv_ct_128').output,
                                                    net.get_layer('conv_ct_256').output, net.get_layer('conv_ct_512').output,
                                                    net.get_layer('decoder_conv_256').output, net.get_layer('decoder_conv_128').output,
                                                    net.get_layer('decoder_conv_64').output, net.get_layer('decoder_conv_32').output])

# ---------------------
#  Display Activation
# ---------------------

def display_activation(activation_map, filter_num, layer_name):
    # 16 : 192 x 192
    # 32 : 96 x 96
    # 64 : 48 x 48
    # 128: 24 x 24
    # 256: 12 x 12
    col_size = math.ceil(math.sqrt(filter_num))
    row_size = col_size
    fig_ind = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 2.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation_map[0, :, :, fig_ind], cmap='gray')
            fig_ind += 1
            if fig_ind >= filter_num:
                break
        if fig_ind >= filter_num:
            break
    plt.savefig(output_feats_path + layer_name + '.png')
    plt.close()

def save_act_figs(act_model, img1, img2):
    activation_maps = act_model.predict([img1, img2])
    display_activation(activation_maps[0], 32, 'conv_spect_32')
    display_activation(activation_maps[1], 64, 'conv_spect_64')
    display_activation(activation_maps[2], 128, 'conv_spect_128')
    display_activation(activation_maps[3], 256, 'conv_spect_256')
    display_activation(activation_maps[4], 512, 'conv_spect_512')
    display_activation(activation_maps[5], 32, 'conv_ct_32')
    display_activation(activation_maps[6], 64, 'conv_ct_64')
    display_activation(activation_maps[7], 128, 'conv_ct_128')
    display_activation(activation_maps[8], 256, 'conv_ct_256')
    display_activation(activation_maps[9], 512, 'conv_ct_512')
    display_activation(activation_maps[10], 256, 'decoder_conv_256')
    display_activation(activation_maps[11], 128, 'decoder_conv_128')
    display_activation(activation_maps[12], 64, 'decoder_conv_64')
    display_activation(activation_maps[13], 32, 'decoder_conv_32')


# Method to call to get the output of the segmentation layer (that's your segmentation)
def get_segmentation(img1,img2):
    output = segmentation_model.predict([img1,img2])
    output[output>=0.5]=1
    output[output<0.5]=0
    return output

# Get weights from network
def get_class_weights(class_weights_exp):
    class_frequencies = np.array([np.sum(label_train == f) for f in range(num_labels)])
    class_weights = class_frequencies.sum() / (class_frequencies.astype(np.float32)+1e-6)
    return class_weights ** class_weights_exp

# Saving Samples
def save_samples(imgCT, imgSPECT, label_out, seg_out, output_image_path, idx):
    plt.figure(num=None, figsize=(15, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.subplot(1, 6, 1);plt.axis('off');plt.imshow(imgCT[idx, :, :, 0], cmap='gray');plt.title('CT image')
    plt.subplot(1, 6, 2);plt.axis('off');plt.imshow(imgSPECT[idx, :, :, 0], cmap='gray');plt.title('SPECT image')
    plt.subplot(1, 6, 3);plt.axis('off');plt.imshow(label_out[idx, :, :, 1], cmap='gray');plt.title('Lesion Label')
    plt.subplot(1, 6, 4);plt.axis('off');plt.imshow(label_out[idx, :, :, 2], cmap='gray');plt.title('Bone Label')
    plt.subplot(1, 6, 5);plt.axis('off');plt.imshow(seg_out[idx, :, :, 1], cmap='gray');plt.title('Lesion Seg.')
    plt.subplot(1, 6, 6);plt.axis('off');plt.imshow(seg_out[idx, :, :, 2], cmap='gray');plt.title('Bone Seg.')
    output_name = 'seg.' + str(epoch) + '.'+str(idx)+'.png'
    plt.savefig(output_image_path + output_name)
    plt.close()

# ---------------------
#  Initialize Generator
# ---------------------
train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()
test_generator  = ImageDataGenerator()

class_weights = get_class_weights(1)
print('Class weights: ' + str(class_weights))

print('\nExperiment started...')

# Create empty log file
with open(output_path + '/stdout.txt', 'w') as f:
    pass

# Some initializations
startTime = time.time()
train_loss = []
val_loss = []
dice = []
dice_all = []
max_dice = -sys.float_info.max
best_epoch = 0
best_weights = None

for epoch in range(n_epochs):

    # ---------------------
    #  Training Phase
    # ---------------------
    # select a random subset of images
    seed = np.random.randint(1e5)
    image_gen = train_generator.flow(image_train, batch_size=batch_size, shuffle=True, seed=seed)
    label_gen = train_generator.flow(label_train, batch_size=batch_size, shuffle=True, seed=seed)
    train_loss_epoch = []
    n_batches = 0
    print('Epoch :' + str(epoch)+'/'+str(n_epochs) + ' training start')

    for img, label in zip(image_gen, label_gen):
        label = label.astype(int) #cast label to int
        # load images
        imgCT    = img[:, :, 0:192, :]
        imgSPECT = img[:, :, 192:192 * 2, :]
        # reshape labels to match network's output
        weights = class_weights[label].reshape(len(imgSPECT), pixels_per_image, 1)
        # expand label image
        label = to_categorical(label, num_labels).reshape(len(imgSPECT), pixels_per_image, num_labels)
        # train network
        loss = net.train_on_batch([imgCT, imgSPECT], label)
        train_loss_epoch.append(loss)
        n_batches += 1
        print('training batch num: '+str(n_batches))
        if n_batches == n_batches_per_epoch:
            break

    train_loss_mean = np.mean(train_loss_epoch)
    with open(output_path + '/stdout.txt', 'a') as f:
        print('Epoch: ' + str(epoch) + '\ntrain loss: ' + str(train_loss_mean),file = f)

    train_loss.append(train_loss_mean)

    # ---------------------
    #  Validation Phase
    # ---------------------

    # some initializations
    val_loss_epoch = []
    dice_epoch = []
    n_batches = 0
    save_tiff = True
    print('Epoch :' + str(epoch)+'/'+str(n_epochs) + ' validation start')
    for img, label in valid_generator.flow(image_valid, label_valid, batch_size=batch_size, shuffle=True):
        # load images
        imgCT    = img[:, :, 0:192, :]
        imgSPECT = img[:, :, 192:192 * 2, :]
        orig_label = label.astype(int)
        label = to_categorical(orig_label, 3).reshape(len(imgSPECT), pixels_per_image, 3)
        # Run test images on network and get loss
        loss = net.test_on_batch([imgCT, imgSPECT], label)
        val_loss_epoch.append(loss)
        # Get segmentation output on test images
        orig_seg = get_segmentation(imgCT, imgSPECT)
        #print(orig_seg.shape)
        seg = orig_seg.reshape(len(imgSPECT), pixels_per_image, 3)
        if save_tiff:
            save_tiff = False
            seg_out = seg.reshape(len(imgSPECT), 192, 192, 3)
            label_out = label.reshape(len(label), 192, 192, 3)
            #print(orig_label.shape)
            #print(seg_out.shape)
            save_samples(imgCT, imgSPECT, label_out, seg_out, output_image_path, 1)
            save_samples(imgCT, imgSPECT, label_out, seg_out, output_image_path, 10)

        # Calculate dice coefficient for batch test images
        dice_epoch.append(dice_coef(seg, label, 3))
        n_batches += 1
        print('Validation batch num: ' + str(n_batches))
        if n_batches == 100: #int(len(image_valid) / batch_size) + int(len(image_valid) % batch_size > 0):
            break

    # ---------------------
    #  Testing Phase
    # ---------------------
    print('Epoch :' + str(epoch) + '/' + str(n_epochs) + ' testing start')
    n_batches = 0
    for img in test_generator.flow(image_test, batch_size=batch_size, shuffle=False):
        #load images
        imgCT    = img[:, :, 0:192, :]
        imgSPECT = img[:, :, 192:192 * 2, :]
        orig_seg = get_segmentation(imgCT, imgSPECT)
        if n_batches == 2:
            img1 = imgSPECT[10, :, :, 0].reshape(1, 192, 192, 1)
            img2 = imgCT[10, :, :, 0].reshape(1, 192, 192, 1)
            # show activations
            save_act_figs(activation_model, img2, img1)
            seg_out = orig_seg.reshape(len(imgSPECT), 192, 192, 3)
            for test_i in range(batch_size):
                plt.figure(num=None, figsize=(15, 6), dpi=200, facecolor='w', edgecolor='k')
                plt.subplot(1, 4, 1);plt.axis('off');plt.imshow(imgCT[test_i, :, :, 0], cmap='gray');plt.title('CT image')
                plt.subplot(1, 4, 2);plt.axis('off');plt.imshow(imgSPECT[test_i, :, :, 0], cmap='gray');plt.title('SPECT image')
                plt.subplot(1, 4, 3);plt.axis('off');plt.imshow(seg_out[test_i, :, :, 1], cmap='gray');plt.title('Lesion Seg.')
                plt.subplot(1, 4, 4);plt.axis('off');plt.imshow(seg_out[test_i, :, :, 2], cmap='gray');plt.title('Bone Seg.')
                output_name = 'seg.' + str(epoch) + '.' + str(test_i) + '.png'
                plt.savefig(output_test_path + output_name)
                plt.close()
        n_batches += 1
        print('testing batch num: ' + str(n_batches))
        if n_batches == 6:  # int(len(image_valid) / batch_size) + int(len(image_valid) % batch_size > 0):
            break

    val_loss_mean = np.mean(val_loss_epoch)
    print('\nEpoch: ' + str(epoch) +'/'+str(n_epochs)+ ' ->  Train loss: ' + str(train_loss_mean) + ' Validation loss: <-' + str(val_loss_mean))

    print('val Dice score: ', np.mean(dice_epoch, 0))
    with open(output_path + '/stdout.txt', 'a') as f:
        print('val loss: ' + str(val_loss_mean), file=f)
        print('val Dice score: '+str(np.mean(dice_epoch, 0)), file=f)

    val_loss.append(val_loss_mean)

    dice_cur_epoch = np.mean(dice_epoch, 0)
    dice_les_cur = dice_cur_epoch[1]
    dice_bone_cur = dice_cur_epoch[2]


    dice_epoch_mean = np.mean(dice_epoch)
    dice.append(dice_epoch_mean)
    # Make sure to pick the best model from a few epochs later
    dice_les_max = 0.75
    dice_bone_max = 0.75
    if (dice_cur_epoch[1] >= dice_les_max and dice_cur_epoch[2] > dice_bone_max):
        if dice_cur_epoch[1] >= dice_les_max:
            dice_les_max = dice_les_max + 0.01
        if dice_cur_epoch[2] >= dice_bone_max:
            dice_bone_max = dice_bone_max + 0.01
        max_dice = dice_epoch_mean
        best_epoch = epoch
        best_weights = net.get_weights()
        net.set_weights(best_weights)
        net.save(output_path + 'model.h5')
    # The image generator has some memory issues
    collected = gc.collect()

endTime = time.time()

print('Time used: ' + str(endTime - startTime) + ' seconds.')
print('Best epoch: ' + str(best_epoch))
with open(output_path + '/stdout.txt', 'a') as f:
    print('Time used: '+str(endTime - startTime)+' seconds.', file=f)
    print('Best epoch: ' + str(best_epoch), file=f)
