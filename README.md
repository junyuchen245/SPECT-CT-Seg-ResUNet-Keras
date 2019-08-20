# Bone & Bone Lesion Segmentation in SPECT/CT Using Res-U-Net
This is almost the same as the U-Net segmentation project (<a href="https://github.com/junyuchen245/SPECT-CT-Seg-UNet">SPECT-CT-Seg-UNet</a>), but the U-Net was modified to the Residual U-Net discribed in:

<a href="https://ieeexplore.ieee.org/abstract/document/8309343">Zhange, Zhengxin, et al. "Road extraction by deep residual u-net." IEEE Geoscience and Remote Sensing Letters, vol. 15, no. 5, pp. 749-753, May 2018.</a>

## Example Segmentation Results
### Validation Set (SPECT/CT simulations)
<img src="https://github.com/junyuchen245/SPECT-CT-Seg-ResUNet-Keras/blob/master/sample_img/validation.png" width="400"/>
Dice similarity coefficient: 0.977 for lesion, 0.979 for bone

### Testing Set (PET/CT)
<img src="https://github.com/junyuchen245/SPECT-CT-Seg-ResUNet-Keras/blob/master/sample_img/test.png" width="300"/>
Resulting an undesired result with the unseen patient PET/CT images.
