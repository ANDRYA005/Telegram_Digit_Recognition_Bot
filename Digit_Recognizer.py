# Baseline MLP for MNIST dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import h5py
from keras.models import load_model
from keras.preprocessing.image import save_img

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def display_image():
	image_index = 771 # You may select anything up to 60,000
	print(y_train[image_index]) # The label is 8
	plt.axis('off')
	plt.imshow(x_train[image_index], cmap='Greys')
	print(x_train[image_index])
	plt.show()

def reshape_and_norm():
	global x_train, x_test
	# Reshaping the array to 4-dims so that it can work with the Keras API
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	# input_shape = (28, 28, 1)
	# Making sure that the values are float so that we can get decimal points after division
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# Normalizing the RGB codes by dividing it to the max RGB value.
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print('Number of images in x_train', x_train.shape[0])
	print('Number of images in x_test', x_test.shape[0])


def create_model():
	# Creating a Sequential Model and adding the layers
	model = Sequential()
	model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation='softmax'))
	return model

def compile_and_fit(model):
	model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
	model.fit(x=x_train,y=y_train, epochs=10)
	model.save('final_digit_model.h5')

def evaluate(model):
	model.evaluate(x_test, y_test, verbose=2)


def predict(model):
	image_index = 7777
	plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
	plt.show()
	pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
	print(pred.argmax())

def main():
	# display_image()
	reshape_and_norm()
	mod = create_model()
	compile_and_fit(mod)
	evaluate(mod)
	# predict(mod)



if __name__ == '__main__':
    main()



#
#
# # flatten 28*28 images to a 784 vector for each image
# num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
# X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# # normalize inputs from 0-255 to 0-1
# X_train = X_train / 255
# X_test = X_test / 255
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
# # define baseline model
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
# 	# Compile model
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model
# # build the model
# model = baseline_model()
# # Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
