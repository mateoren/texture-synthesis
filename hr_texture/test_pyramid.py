import numpy as np
import scipy
import tensorflow as tf
import skimage
import vgg19
import utils
import time
import os
import sys
import shutil

import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_gaussian


img_name = str(sys.argv[1])
if os.path.exists('./result/'):
    shutil.rmtree('./result/')
    os.makedirs('./result/')
else :
    os.makedirs('./result/')
## Process the origin image to be an tensorflow tensor


#image  = skimage.io.imread("./test_data/"+img_name)
image = utils.load_image("./test_data/"+img_name)   # [0,1)

#image = data.astronaut()
rows, cols, dim = image.shape
pyramid = tuple(pyramid_gaussian(image, max_layer = 4, downscale=2))




composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()
