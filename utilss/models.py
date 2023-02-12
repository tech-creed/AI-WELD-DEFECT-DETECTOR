import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from efficientnet.tfkeras import EfficientNetB4
import cv2

SIZE = 256

def detectDefect(imagePath,id):
    model = tf.keras.models.load_model(f"models\WeldDetectorModel.h5", compile=False)

    X = np.zeros((1, SIZE, SIZE, 1))
    image = Image.open(imagePath)
    image = np.array(image.resize((SIZE, SIZE)))
    image = image.reshape((SIZE, SIZE, 1))
    X[0,] = np.array(image) / 255.
    pred = model.predict(X) 

    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.imshow(pred[0].reshape(SIZE,SIZE), cmap='coolwarm', alpha=0.4)
    plt.savefig(f"static/uploads/{id}/mask.png", bbox_inches='tight', pad_inches=0)

#----------Augumenttaion-------------#
def zoom_center(img, zoom_factor=1.5):
    y_size = img.shape[0]
    x_size = img.shape[1]
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
    
def zoom_img(image_path,patient_id):
    img = cv2.imread(image_path)
    img_zoomed_and_cropped = zoom_center(img)
    cv2.imwrite(f'static/uploads/{patient_id}/zoom_crop.png', img_zoomed_and_cropped)

def rotate(image_path,patient_id):
    img = cv2.imread(image_path)
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_180 = cv2.rotate(img_90, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f'static/uploads/{patient_id}/rotate_90.png', img_90)
    cv2.imwrite(f'static/uploads/{patient_id}/rotate_180.png', img_180)

def increase_brightness(image_path,id):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'static/uploads/{id}/brightness.png', img)

def color_spaces(image_path, id):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'static/uploads/{id}/hsv.png', hsv)
    cv2.imwrite(f'static/uploads/{id}/YCrCb.png', YCrCb)
    cv2.imwrite(f'static/uploads/{id}/gray.png', gray)
    cv2.imwrite(f'static/uploads/{id}/rgb.png', rgb)


def make_augumentation(image_path,id):
    zoom_img(image_path,id)
    rotate(image_path,id)
    increase_brightness(image_path,id)
    color_spaces(image_path,id)