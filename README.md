# Telegram_Digit_Recognition_Bot
A Telegram Bot that, when sent an image of a handwritten digit, replies with the digit in the image and a confidence score (using a Keras neural net).

The neural net was trained on the classic MNIST handwritten-digits dataset, saved and then used within the operations of the Telegram bot.

Here is a brief outline of the process of the code:

1. The telegram bot receives an image of a handwritten digit.
2. The image is then dowloaded and saved locally.
3. The downloaded image is then preprocessed in order to best resemble the images used in training by: (a) reducing the size of the handwritten digit to 20x20 pixels by removing all the pixel rows and columns of the image that do not contain any handwriting. (b) Adding blank padding around the handwriting in order to meet the criteria of a 28x28 pixel image. (c) Performing a shift to ensure that the hand-writing is centered in the image.
4. The 28x28 pixel image is then converted to a numpy array and passed into the model (neural net).
5. The predicted value and the probability associated with the digit is sent by the bot as the prediction for the handwritten digit and the confidence score respectively.

The following screenshots demonstrates the bot:

![alt text](https://github.com/ANDRYA005/Telegram_Digit_Recognition_Bot/blob/master/Screenshot_for_GitHub.PNG)

![alt text](https://github.com/ANDRYA005/Telegram_Digit_Recognition_Bot/blob/master/Screenshot_for_GitHub(2).PNG)
