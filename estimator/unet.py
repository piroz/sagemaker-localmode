import os
import glob
import argparse

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import numpy as np

from model.keras_unet import unet_model

# Geforce RTX CUDNN_STATUS_INTERNAL_ERROR work arround
# https://github.com/tensorflow/tensorflow/issues/29632
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
parser.add_argument('--model_version', type=str, default='1')
args = parser.parse_args()

HEIGHT = 256
WIDTH = 256
OUTPUT_CHANNELS = 1

def load_images(path, color_mode='rgb', ext='jpg'):
    pattern = os.path.join(path, '*.{}'.format(ext))
    return np.array(list(map(lambda file: image.img_to_array(image.load_img(file, color_mode=color_mode)), glob.glob(pattern))))

def augmentted_save_path(base_path):
    path = {
        'frames': '{}/generated/frames'.format(base_path),
        'masks': '{}/generated/masks'.format(base_path)
    }

    for k, v in path.items():
        os.makedirs(v, exist_ok=True)

    return path

def create_generator(image_data_generator_args, image_path, ):
    image_data_generator = ImageDataGenerator(**generator_args)

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == '__main__':

    save_path = augmentted_save_path(args.output_dir)

    data_gen_args = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2)

    frame_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    frames_path = os.path.join(args.train_dir, 'train_frames')
    masks_path =os.path.join(args.train_dir, 'train_masks')

    frames = load_images(frames_path + '/img')
    frame_datagen.fit(frames, augment=True, seed=seed)

    masks = load_images(masks_path + '/img', color_mode='grayscale')
    mask_datagen.fit(masks, augment=True, seed=seed)
    
    frame_generator =frame_datagen.flow_from_directory(frames_path,
        class_mode=None, seed=seed, save_to_dir=save_path["frames"])

    mask_generator = mask_datagen.flow_from_directory(masks_path,
        class_mode=None, seed=seed, save_to_dir=save_path["masks"], color_mode='grayscale')

    train_generator = (pair for pair in zip(frame_generator, mask_generator))

    model = unet_model(OUTPUT_CHANNELS)

    model.compile(optimizer='adam',
        loss=dice_coef_loss,
        metrics=[dice_coef])

    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=1)


    model_save_path = os.path.join('/opt/ml/model', unet_model.__name__, args.model_version)
    model.save(model_save_path, save_format='tf')



