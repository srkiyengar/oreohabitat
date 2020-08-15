import numpy as np
import quaternion
from scipy import interpolate
import matplotlib.pyplot as plt


rx = -1.0
ry = -1.0
rz = 1.0

r = np.sqrt(rx*rx+ry*ry+rz*rz)
rx = rx/r
ry = ry/r
rz = rz/r


x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)

xx, yy = np.meshgrid(x,y)
z = np.sin(xx**2+yy**2)
f = interpolate.interp2d(x,y,z, kind='cubic')

xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 5.01, 1e-2)

znew = f(xnew, ynew)

plt.plot(x, z[0,:],'r^', xnew, znew[0,:], 'b-')
plt.show()
pass