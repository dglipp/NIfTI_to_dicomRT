import cv2 as cv
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


os.chdir(os.getcwd() + "/data")
for file in ['deep_GGO.nii.gz']:
  img = nib.load(file)
pixels = img.get_fdata()

nonvoid = list()
for i in range(pixels.shape[2]):
  if(np.max(pixels[:,:,i]) == 255.0):
    nonvoid.append(i)

fig, ax = plt.subplots(1,2)
idx = 10
ax[0].imshow(pixels[:,:,nonvoid[idx]])

imgray = pixels[:,:,nonvoid[idx]].astype(np.uint8)

contours, _ = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

mask = np.zeros(imgray.shape, np.uint8)
im2 = cv.UMat(imgray)
cv.drawContours(mask, contours, -1, 255, 1)

ax[1].imshow(mask)
plt.show()
