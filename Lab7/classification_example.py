import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers

# Get details of the first GPU device
tf.config.list_physical_devices('GPU')

#import labeled dataset (MNIST handwritten digit database) distributed with Keras.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# fig = plt.figure()
# plt.imshow(x_train[0], cmap = "Greys") #Use matplotlib to plot an image
# plt.show()

# ( number of x pixels ) by ( number of y pixels ) by ( number of color channels )
input_shape = (28, 28, 1)
# we will prefer to " flatten " this matrix into a vector :
input_shape_flat = (input_shape[0] * input_shape[1] * input_shape[2],)
input_shape_flat

# Make sure images have shape (784 ,)
# i.e . x_train is a (# data points x 784) matrix
x_train = np.reshape(x_train, (-1, input_shape_flat[0]))
x_test = np.reshape(x_test, (-1, input_shape_flat[0]))

# Scale images to the [0 , 1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Model / data parameters
num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# L1 = layers.Dense(20, activation='relu') # 20 neurons -> 20 outputs ; ReLU , i.e. max {x ,0} , activation
# L2 = layers.Dense(24, activation='relu') # 24 neurons -> 24 outputs ; ReLU , i.e. max {x ,0} , activation
# L3 = layers.Dense(10, activation= "softmax") # can specify as no activation i.e. linear activation

# inLayer = layers.Input(shape=(10,)) # 10 inputs , i.e. input a vector of length 10
# inLayer = layers.Input(shape=input_shape_flat) 
# # print(inLayer)

# nnOutput = L3(L2(L1(inLayer)))
# # print(nnOutput)
# nn = keras.models.Model(inputs = inLayer , outputs = nnOutput)
# # print(nn)
# nn.predict(x_train[0].reshape(1, -1))  # In this case , predict expects a (# of images ) x (784) matrix , hence the " reshape " operation
# # print(test)

# " MSE " = " mean - squared error "
# " adam " = ADAM optimizer ( improved stochastic gradient descent )
# nn.compile(loss = "MSE", optimizer = "adam")
# y_hat = nn.predict (x_train)
# mse = np.sum((y_train[0] - y_hat[0]) ** 2)
# print("MSE: ", mse)



nn = keras.Sequential([
    keras.Input(shape=input_shape_flat),
    keras.layers.Reshape(input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.fit(x_train , y_train , batch_size =128 , epochs =5 , validation_data =(x_test , y_test))