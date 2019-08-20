import os
import sys
sys.path.append('/netscratch/garyli2/anaconda2/lib/python2.7/site-packages')
sys.path.append('/netscratch/garyli2/muscleFiberSegmentation/')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.image_reading import load_image_from_folder

input_path = '/netscratch/garyli2/histoImages/train_leftout_G32M1R/'

image_array, label_array = load_image_from_folder(input_path, (256,256), HE=False,Truc=False)

#image_array_HE, label_array = load_image_from_folder(input_path, (256,256), HE=False,Truc=True)

ax = plt.subplot(2, 4, 1)
ax.set_title('Original image')
plt.imshow(image_array[0,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4, 2)
ax.set_title('Augmented image 1')
plt.imshow(image_array[1,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,3 )
ax.set_title('Augmented image 2')
plt.imshow(image_array[2,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,4 )
ax.set_title('Augmented image 3')
plt.imshow(image_array[3,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,5 )
ax.set_title('Augmented image 4')
plt.imshow(image_array[4,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,6 )
ax.set_title('Augmented image 5')
plt.imshow(image_array[5,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,7 )
ax.set_title('Augmented image 6')
plt.imshow(image_array[6,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

ax = plt.subplot(2, 4,8 )
ax.set_title('Augmented image 7')
plt.imshow(image_array[7,:,:], interpolation='none',cmap=cm.Greys_r)
#ax.axis('off')

plt.show()