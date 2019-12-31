# Bot DogBot that sends a random dog picture when receives /bop from user.

from telegram.ext import Updater, InlineQueryHandler, CommandHandler, MessageHandler, Filters
import requests
import re
import cv2
from PIL import Image

# Baseline MLP for MNIST dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import h5py
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
from scipy import ndimage
import math


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def preprocess_image(img):

    # All images are size normalized to fit in a 20x20 pixel box and they are centered in a 28x28 image using the center of mass.

    gray = cv2.resize(255-np.asarray(img), (28, 28), cv2.INTER_LANCZOS4)
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # fit the images into this 20x20 pixel box. Therefore we need to remove every row and column at the sides of the image which are completely black.
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    # resize our outer box to fit it into a 20x20 box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    # we need a 28x28 pixel image so we add the missing black rows and columns using the np.lib.pad function which adds 0s to the sides.
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    img = gray
    return img


def predict(bot, chat_id):
    # global model
    model = load_model("new_digit_model.h5")
    # load the image
    img = load_img('image.png', color_mode = "grayscale")
    width, height = img.size
    # print("width: ",width)
    # print("height: ",height)
    if width>1000 or height>1000:                                                           # some of the smaller images did not respond well to the image preprocessing
        print("Preprocessing image...")
        img = preprocess_image(img)
    else:
        print("Not preprocessing image...")
        # img = np.invert(load_img('image.png', color_mode = "grayscale", target_size = (28,28)))
        img = img.resize((28,28), Image.LANCZOS)                                            # Image.NEAREST is used above
        img = np.invert(img)

    cv2.imwrite("processed.png", img)
    img_array = img_to_array(img)
    img_array /= 255                                                                        # nn scales down pixels between 0 and 1
    pred = model.predict(img_array.reshape(1,28,28,1))
    best_pred = str(pred.argmax())
    best_percent = str(round(pred.max()*100,2))
    print(best_pred)
    print(best_percent)
    sent = best_pred + " (with " + best_percent + "% confidence)."
    bot.send_message(chat_id=chat_id, text=sent)


def receive_image(bot, update):
    chat_id = update.message.chat_id
    file = bot.getFile(update.message.photo[-1].file_id)
    bot.send_message(chat_id=chat_id, text="Processing image...")
    file.download('image.png')
    predict(bot, chat_id)



def main():
    updater = Updater('1031604196:AAFVlKDWmeXM8TJ3p6ZIoMGvf0nfJTMy-C4')
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo, receive_image))
    updater.start_polling()
    updater.idle()

# print("broken here")
# model = load_model('new_digit_model.h5')
# print("here")
# # load the image
# img = load_img('image.png', color_mode = "grayscale")
# width, height = img.size
# # print("width: ",width)
# # print("height: ",height)
# if width>1000 or height>1000:                                                           # some of the smaller images did not respond well to the image preprocessing
#     print("Preprocessing image...")
#     img = preprocess_image(img)
# else:
#     print("Not preprocessing image...")
#     # img = np.invert(load_img('image.png', color_mode = "grayscale", target_size = (28,28)))
#     img = img.resize((28,28), Image.LANCZOS)                                            # Image.NEAREST is used above
#     img = np.invert(img)
#
# cv2.imwrite("processed.png", img)
# print("Image written")
# img_array = img_to_array(img)
# img_array /= 255                                                                        # nn scales down pixels between 0 and 1
# print("pre-predicted")
# pred = model.predict(img_array.reshape(1,28,28,1))
# print("predicted")
# best_pred = str(pred.argmax())
# best_percent = str(round(pred.max()*100,2))
# print(best_pred)
# print(best_percent)
# # sent = best_pred + " (with " + best_percent + "% confidence)."
# # bot.send_message(chat_id=chat_id, text=sent)

if __name__ == '__main__':
    main()
