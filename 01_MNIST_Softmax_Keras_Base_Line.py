# MNIST 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist

# In Python3, use urllib instead of urllib2
import ssl
import urllib

# Prevent https access from SLL certificate verification
# Close certificate verification when importing SLL
ssl._create_default_https_context = ssl._create_unverified_context

# Hyperparameters
NB_EPOCHS = 100
BATCH_SIZE = 128
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2

# Load Data
# X_train is 60000 rows of 28*28 values, X_test is 10000 rows of 28*28
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# flatten
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Normalize
X_train /= 255
X_test /= 255


# Convert class into one-hot encoding
Y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

# Build the model
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(784,)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer=SGD(), metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCHS, verbose = 1, 
                   validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
