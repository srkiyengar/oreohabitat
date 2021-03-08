import pickle
import numpy as np
import skimage.measure
import math

import matplotlib.pyplot as plt

#fname = "./saliency_map/apartment1"
fname = "./saliency_map/skokloster-castle.glb.saliency"
scene_filename = "./saliency_map/skokloster-castle.glb"
my_block = 16
eye_hov = 120
total_points = 10

def scale_image(some_image, new_max):
    vmin = some_image.min()
    vmax = some_image.max()
    some_image = (some_image - vmin) * new_max / (vmax - vmin)
    return some_image

def find_max_and_index(array_2d):
    #result = np.where(array_2d == np.amax(array_2d))
    result = np.nonzero(array_2d == np.amax(array_2d))
    r = result[0][0]
    c = result[1][0]
    return np.amax(array_2d), r, c


class saliency(object):

    def __init__(self, scene_filename, salmap_file, block_dim, camera_hov, num_points):
        self.block_dim = block_dim
        try:
            with open(salmap_file, "rb") as f:
                self.salmap = pickle.load(f)
                try:
                    with open(scene_filename, "rb") as f:
                        scene = pickle.load(f)
                except IOError as e:
                    print("Failure: Opening image file {}".format(scene_filename))
                    exit(1)
        except IOError as e:
            print("Failure: Opening saliency file {}".format(fname))
            try:
                with open(scene_filename, "rb") as f:
                    scene = pickle.load(f)
            except IOError as e:
                print("Failure: Opening image file {}".format(scene_filename))
            finally:
                exit(1)

        self.scene = scene[:, :, ::-1]
        self.reduced_salmap = skimage.measure.block_reduce(self.salmap, block_size=(block_dim, block_dim), \
                                                           func=np.amax)
        self.reduced_salmap = scale_image(self.reduced_salmap,150)
        self.reduced_sal_points = []
        # Macular angle is 18 out of HFOV (120); d should be 5
        self.d = math.ceil((18.0 * self.salmap.shape[0]) / (camera_hov * block_dim))
        print(f"The kernel dimension d = {self.d}")
        # compute salient points list
        for i in range(num_points):
            val, r, c = find_max_and_index(self.reduced_salmap)
            self.reduced_sal_points.append((val, r, c))
            self.zero_around_macular_center(r, c, i)
        self.num_points = num_points
        for i in range(num_points):
            self.reduced_salmap[self.reduced_sal_points[i][1], self.reduced_sal_points[i][2]] = 255-i*10
        
        self.recreated_salmap = scale_image(np.copy(self.salmap),150)
        self.center_points = []
        self.show_sal_point_for_full_image()


    def zero_around_macular_center(self, r, c, i):
        f = int((self.d - 1) / 2)
        self.reduced_salmap[r-f:r+f+1,c-f:c+f+1] = 0


    def show_sal_point_for_full_image(self):
        count = 0
        for _, r, c in self.reduced_sal_points:
            start_r = r * self.block_dim
            start_c = c * self.block_dim
            '''
            recreated_image[start_r:start_r+block_size,start_c:start_c+block_size] = \
                recreated_image[start_r:start_r + block_size, start_c:start_c + block_size]/2
            '''
            val = 255 - 10 * count
            self.recreated_salmap[start_r:start_r + self.block_dim, start_c:start_c + self.block_dim] = val
            self.center_points.append((val, start_r + 16, start_c + 16))
            count += 1

        return


if __name__ == "__main__":

    sal_object = saliency(scene_filename, fname, my_block, eye_hov, total_points)
    fig = plt.figure(figsize=(8, 8))
    r1c1 = fig.add_subplot(2, 2, 1)
    r1c2 = fig.add_subplot(2, 2, 2)
    r2c1 = fig.add_subplot(2, 2, 3)
    r2c2 = fig.add_subplot(2, 2, 4)

    r1c1.imshow(sal_object.scene)
    r1c2.imshow(sal_object.salmap)
    r2c1.imshow(sal_object.reduced_salmap)
    r2c2.imshow(sal_object.recreated_salmap)
    print(f"job completed")