import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

with open('eye_scan_data.pkl', 'rb') as file:
    data = pickle.load(file)

left_eye_scan_data = data[0]

my_lefteye_table = np.array(left_eye_scan_data)
my_lefteye_table = my_lefteye_table[::6,:]
x = my_lefteye_table[:, 2]  # all yaw
y = my_lefteye_table[:, 3]  # all pitch

z_left = my_lefteye_table[:,0]  # all left_actuator positions
z_right = my_lefteye_table[:,1]  # all right_actuator positions

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z_left, '.')
plt.xlabel('x')
plt.ylabel('y')

print(my_lefteye_table.shape)

points = np.array((x, y)).T
left_eye_interpolator_left = interpolate.LinearNDInterpolator(points, z_left)

new_x = np.arange(-28, 47, 2)
new_y = np.arange(42, 130, 2)
xx, yy = np.meshgrid(new_x, new_y)
new_z = left_eye_interpolator_left(xx.flatten(), yy.flatten())

ax.plot(xx.flatten(), yy.flatten(), new_z.flatten(), 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('scan data', 'interpolated'))
plt.savefig('left-actuator-interpolation.png')
plt.show()