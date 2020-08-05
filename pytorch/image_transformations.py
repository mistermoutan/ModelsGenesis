from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    n: number of points
    i: the specific point {0, n_points}
    t: [0,t], given as linspace (0,1,number of steps) - number of steps: higher gives you closer proximity points aling the curve  
    """

    a = comb(n, i) * (t ** (n - i)) * (1 - t) ** i
    return a
    # return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)  # x

    # n poits = 4
    polynomial_array = np.array([bernstein_poly(i=i, n=nPoints - 1, t=t) for i in range(0, nPoints)])

    # the sum term of the polinomila is handle in this mattrix multipliation
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    # project array values to curve
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5, two_dim=False):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    if not two_dim:
        _, img_rows, img_cols, img_deps = x.shape
    else:
        _, img_rows, img_cols = x.shape

    num_block = 10000

    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        if not two_dim:
            block_noise_size_z = random.randint(1, img_deps // 10)
            noise_z = random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[
                0, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z,
            ]
        else:
            window = orig_image[0, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        if not two_dim:
            window = window.reshape((block_noise_size_x, block_noise_size_y, block_noise_size_z))
            image_temp[
                0, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z
            ] = window
        else:
            window = window.reshape((block_noise_size_x, block_noise_size_y))
            image_temp[
                0, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y
            ] = window

    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x, two_dim=False):
    if not two_dim:
        _, img_rows, img_cols, img_deps = x.shape
    else:
        _, img_rows, img_cols = x.shape
    
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        if two_dim:
            block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)    
            x[:, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z] = (
                np.random.rand(block_noise_size_x, block_noise_size_y, block_noise_size_z,) * 1.0
            )
        else:
            x[:, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y] = (
            np.random.rand(block_noise_size_x, block_noise_size_y) * 1.0)
        cnt -= 1
    return x


def image_out_painting(x, two_dim=False):
    if not two_dim:
        _, img_rows, img_cols, img_deps = x.shape
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3],) * 1.0

    else:
        _, img_rows, img_cols = x.shape
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2],) * 1.0

    # print(x.shape)
    image_temp = copy.deepcopy(x)
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    if not two_dim:
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        if (block_noise_size_z + 3) >= img_deps:
            print(img_deps, block_noise_size_z)
            dif = (block_noise_size_z + 3) - img_deps
            block_noise_size_z -= random.randint(dif, dif + 4)
                print(block_noise_size_z)
        if img_deps - block_noise_size_z - 3 <= 3:
            noise_z = random.randint(3, img_deps - 3)
        else:
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)

    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    if not two_dim:
        x[
            :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z
        ] = image_temp[
            :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z
        ]
    else:
        x[:, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y] = image_temp[
            :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        if not two_dim:
            block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            if img_deps - block_noise_size_z - 3 <= 3:
                noise_z = random.randint(3, img_deps - 3)
            else:
                noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
                
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        if not two_dim:
            x[
                :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z
            ] = image_temp[
                :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y, noise_z : noise_z + block_noise_size_z
            ]
        else:
            x[
                :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y
            ] = image_temp[
                :, noise_x : noise_x + block_noise_size_x, noise_y : noise_y + block_noise_size_y
            ]

        cnt -= 1

    return x


def generate_pair(img, batch_size, config, status="test", make_tensors=False, two_dim=False):
    
    # IMG is (N,1,x,y,z) numpy array

    index = [i for i in range(img.shape[0])]
    random.shuffle(index)
    y = img[index[:batch_size]]  # (BAtCH_SIZE, 1, 64, 64, 32)
    x = copy.deepcopy(y)
    for n in range(img.shape[0]):  # range(batch_size):

        # image as (1,64,64,32) here

        # Autoencoder
        x[n] = copy.deepcopy(y[n])

        # Flip
        x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

        # Local Shuffle Pixel
        x[n] = local_pixel_shuffling(x[n], prob=config.local_rate,two_dim=two_dim)

        # Apply non-Linear transformation with an assigned probability
        x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)

        # Inpainting & Outpainting
        if random.random() < config.paint_rate:
            if random.random() < config.inpaint_rate:
                # Inpainting
                x[n] = image_in_painting(x[n],two_dim=two_dim)
            else:
                # Outpainting
                x[n] = image_out_painting(x[n],two_dim=two_dim)

    # Save sample images module
    """     if config.save_samples is not None and status == "train" and random.random() < 0.01:
        n_sample = random.choice([i for i in range(config.batch_size)])
        sample_1 = np.concatenate((x[n_sample, 0, :, :, 2 * img_deps // 6], y[n_sample, 0, :, :, 2 * img_deps // 6]), axis=1)
        sample_2 = np.concatenate((x[n_sample, 0, :, :, 3 * img_deps // 6], y[n_sample, 0, :, :, 3 * img_deps // 6]), axis=1)
        sample_3 = np.concatenate((x[n_sample, 0, :, :, 4 * img_deps // 6], y[n_sample, 0, :, :, 4 * img_deps // 6]), axis=1)
        sample_4 = np.concatenate((x[n_sample, 0, :, :, 5 * img_deps // 6], y[n_sample, 0, :, :, 5 * img_deps // 6]), axis=1)
        final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = "".join([random.choice(string.ascii_letters + string.digits) for n in range(10)]) + "." + config.save_samples
        imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample) """

    if make_tensors:
        from torch import Tensor

        x = Tensor(x)
        y = Tensor(y)

    return (x, y)


if __name__ == "__main__":

    # from config import models_genesis_config

    # config = models_genesis_config(add_model_to_task=False)
    # x = np.zeros((6, 1, 30, 30, 30))
    # generate_pair(x, 6, config)

    A = np.random.randn(1, 2, 3, 3, 3)
    # degree = random.choice([0, 1, 2])
    x = np.flip(A, axis=0)
    print(x)
    print(A)
