import cv2
import dlib
import tensorflow as tf

import numpy as np
from imutils import face_utils
import imutils

# Import Pretrained model
model = tf.keras.models.load_model('./trained_model.h5')


def getFace(image):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        image = image[y:y+h, x:x+w]
    return image


# import predictor and detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('predictor.dat')

# Initialise Camera
cam = cv2.VideoCapture(0)


if __name__ == "__main__":

    # Get original image
    image = cam.read()[1]

    # Convert image to model format
    frame = getFace(image)
    frame = cv2.resize(image, (48, 48))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.uint8(frame)
    frame = frame.reshape((1, 48, 48, 1))
    frame = frame.astype('float32') / 255

    # Labels as per ./fer2013.csv
    labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
              3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    # Print Emotion
    y = model.predict_classes([frame])
    print(labels[y[0]])
