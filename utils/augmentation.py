import tensorflow as tf
import numpy as np
import cv2

import scipy

def random_flip_left_right(image):
    ''' image.shape=[img_h, img_w, img_s, img_t, channels]
    '''
    if np.random.randint(2) == 1:
        img_h, img_w, img_s, img_t, channels = image.shape
        for i in range(img_s):
            for j in range(img_t):
                image[:,:,i,j,:] = cv2.flip(image[:,:,i,j,:], 1).reshape(img_h, img_w, channels)
                
    return image

def random_flip_upside_down(image):
    ''' image.shape=[img_h, img_w, img_s, img_t, channels]
    '''
    if np.random.randint(2) == 1:
        img_h, img_w, img_s, img_t, channels = image.shape
        for i in range(img_s):
            for j in range(img_t):
                image[:,:,i,j,:] = cv2.flip(image[:,:,i,j,:], 0).reshape(img_h, img_w, channels)
                
    return image

def random_flip_upside_down_left_right(image):
    ''' image.shape=[img_h, img_w, img_s, img_t, channels]
    '''
    if np.random.randint(2) == 1:
        img_h, img_w, img_s, img_t, channels = image.shape
        for i in range(img_s):
            for j in range(img_t):
                image[:,:,i,j,:] = cv2.flip(image[:,:,i,j,:], -1).reshape(img_h, img_w, channels)
                
    return image

def random_brightness(inputs, alpha=1.6):
    ''' image.shape=[img_h, img_w, img_s, img_t, channel]
    '''
    image = inputs.copy()
    gamma = np.random.rand() * alpha
    gf = [[np.int8(255 * pow(i/255., 1/gamma))] for i in range(256)]
    table = np.reshape(gf, (256, -1))
    img_h, img_w, img_s, img_t, channel = image.shape
    for i in range(img_s):
        for j in range(img_t):
            # print image.shape
            img = np.int8(Normal(image[:,:,i,j,:])*255.)
            image[:,:,i,j,:] = cv2.LUT(img, table).reshape(img_h,img_w,-1)
    return image

def random_crop_and_zoom(image, alpha=0.5):
    ''' image.shape=[img_h, img_w, img_s, img_t, channel]
    '''
    img_h, img_w, img_s, img_t, channels = image.shape
    r = np.random.uniform(0, alpha)
    v1 = np.random.randint(0, int(r*img_h)) if (int(r*img_h) != 0) else 0
    v2 = np.random.randint(0, int(r*img_w)) if (int(r*img_w) != 0) else 0
    
    for i in range(img_s):
        for j in range(img_t):
            image_ = image[v1:(v1+int((1-r)*img_h)), v2:(v2+int((1-r)*img_w)), :, :, :]
            image[:,:,i,j,:] = cv2.resize(np.squeeze(image_[:,:,i,j,:]), (img_w, img_h)).reshape(img_h,img_w,channels)
    return image

def normalize(image):
    # image = image / 127.5 - 1
    img_h, img_w, img_s, img_t, channels = image.shape
    for i in range(img_s):
        for j in range(img_t):
            image[:,:,i,j,:] = Normal(image[:,:,i,j,:])
    return image

def Normal(_data):
    ''' This function return the normalization of image data

        input :  image:               _data

        output:  normalized_image:    data
    '''
    data = _data.copy()
    data = data.astype(np.float32)
    data = (data-np.min(data)) / (np.max(data) - np.min(data))
    return data

def _augment(image):
    # image = cv2.resize(image, (128, 128))
    image = random_flip_left_right(image)
    image = random_flip_upside_down(image)
    image = random_flip_upside_down_left_right(image)
    image = random_crop_and_zoom(image)
    image = random_brightness(image)
    # image = Normal(image)
    return image

def augment(images):
    return np.array([_augment(image) for image in images])