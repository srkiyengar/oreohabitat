import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from scipy.misc import face
import matplotlib.pyplot as plt
import tensorflow as tf






if __name__ == "__main__":

    img = face()
    plt.imshow(img)
    plt.axis('off')
    # load precomputed log density over a 1024x1024 image
    centerbias_template = np.load('centerbias.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template, (img.shape[0] / 1024, img.shape[1] / 1024), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

    tf.get_default_graph()
    tf.reset_default_graph()
    check_point = 'DeepGazeII.ckpt'  # DeepGaze II
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
    input_tensor = tf.get_collection('input_tensor')[0]
    centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
    log_density = tf.get_collection('log_density')[0]
    log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

    with tf.Session() as sess:
        new_saver.restore(sess, check_point)

        log_density_prediction = sess.run(log_density, {
            input_tensor: image_data,
            centerbias_tensor: centerbias_data,
        })
    print(log_density_prediction.shape)

    plt.gca().imshow(img, alpha=0.2)
    m = plt.gca().matshow((log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
    plt.colorbar(m)
    plt.title('log density prediction')
    plt.axis('off')

    plt.gca().imshow(img, alpha=0.2)
    m = plt.gca().matshow(np.exp(log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
    plt.colorbar(m)
    plt.title('density prediction')
    plt.axis('off');
