from scipy.misc import face
import numpy as np
import matplotlib.pyplot as plt

img = face()
print(img.shape)
plt.imshow(img)
plt.axis('off')
