# -*- coding:utf-8 -*-

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

img_name = os.path.join("imgs","liu_1.png")
img = cv2.imread(img_name, cv2.IMREAD_COLOR)
img_blue = img[:,:,0]
img_green = img[:,:,1]
img_red= img[:,:,2]
cv2.imshow("img_blue",img_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = img[...,[1,0,2]]
print(f"imgage shape:{img.shape},img_red shape:{img_red.shape}")
plt.figure("beauty_full")
plt.subplot(2,2,1)
plt.imshow(img_blue)
plt.subplot(2,2,2)
plt.imshow(img_green)
plt.subplot(2,2,3)
plt.imshow(img_red)
plt.subplot(2,2,4)
plt.imshow(img)
plt.show()
