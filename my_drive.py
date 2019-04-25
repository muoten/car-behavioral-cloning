import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import utils

from config import *
import cv2
from skimage.feature import *


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 100
MIN_SPEED = 10

speed_limit = MAX_SPEED
TRACE = False
iteration = 0
score = 0


def preprocess(image):

    img = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # crop image:
    img = img[Y_CROP:Y_CROP2, :, :]
    # resize image:
    img = cv2.resize(img, (X_PIX, Y_PIX), interpolation=cv2.INTER_CUBIC)
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize again
    img_gray = Image.fromarray(np.uint8(256 * img_gray))
    img_gray = img_gray.resize([int(0.5 * s) for s in img_gray.size], Image.ANTIALIAS)

    # hog features
    img_gray = hog(img_gray, pixels_per_cell=(8, 8))

    return img_gray



@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"].replace(",", "."))
        # The current throttle of the car
        throttle = float(data["throttle"].replace(",", "."))
        # The current speed of the car
        speed = float(data["speed"].replace(",", "."))
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = preprocess(image)  # apply the preprocessing
            image = np.asarray(image).reshape(1, -1) # the model expects this shape

            # predict the steering angle for the image
            steering_angle = model.predict(image)[0]

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            if TRACE == True:
                print('{} {} {}'.format(steering_angle, throttle, speed))

            global score
            score += speed
            global iteration
            iteration += 1
            if (iteration % 100) == 0:
                print('Avg speed: {}, score: {}'.format(score / iteration, score))

            steering_angle = steering_angle.__str__().replace(".", ",")
            throttle = throttle.__str__().replace(".", ",")

            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    import pickle

    PATH_TO_MODEL = args.model
    model = pickle.load(open(PATH_TO_MODEL, 'rb'))

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
