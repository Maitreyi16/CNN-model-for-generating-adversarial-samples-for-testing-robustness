# CNN-model-for-generating-adversarial-samples-for-testing-robustness
import keras
from keras.datasets import mnist
from keras import backend as k
#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() #everytime loading data won't be so easy :)

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig

img_rows = 28  # Replace '28' with the actual number of rows in your images
img_cols = 28  # Replace '28' with the actual number of columns in your images
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else :
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

img_rows = 28  # Replace '28' with the actual number of rows in your images
img_cols = 28  # Replace '28' with the actual number of columns in your images
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else :
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

import keras
#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :)
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(num_category, activation='softmax'))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


              batch_size = 128
num_epoch = 10
#model training
model_log = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904


import os
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig


#Save the model
# serialize model to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save_weights("model_digit.h5")
print("Saved model to disk")


import numpy as np
import tensorflow as tf

# Define the Fast Gradient Sign Method (FGSM) attack function in Keras
def fgsm_attack(model, x, y, epsilon=1):
    x_adv = tf.identity(x)  # Create a copy of the input using TensorFlow identity function

    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        logits = model(x_adv)
        loss = tf.keras.losses.categorical_crossentropy(y, logits)

    # Get the gradients of the loss with respect to the input
    grad = tape.gradient(loss, x_adv)

    # Compute the perturbation as the sign of the gradients
    perturbation = epsilon * tf.sign(grad)
    x_adv = x_adv + perturbation

    return x_adv.numpy()

# Generate adversarial examples using FGSM attack for a few test samples
x_test_adv_fgsm = np.zeros_like(X_test[:100])

for i in range(5):
    x_adv = fgsm_attack(model, X_test[i:i+1], y_test[i:i+1], epsilon=0.1)
    x_test_adv_fgsm[i] = x_adv[0]

# Evaluate the model on the adversarial examples
adv_loss_fgsm, adv_accuracy_fgsm = model.evaluate(x_test_adv_fgsm, y_test[:100], verbose=0)
print(f"Adversarial Test Accuracy (FGSM Attack): {adv_accuracy_fgsm}")



import matplotlib.pyplot as plt

# Display the original and adversarial digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('Original vs Adversarial Digits (FGSM Attack)', fontsize=16)

for i in range(5):
    # Original digit
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f"Original: {y_test[i].argmax()}")

    # Adversarial digit
    axes[1, i].imshow(x_test_adv_fgsm[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f"Adversarial: {model.predict(x_test_adv_fgsm[i:i+1]).argmax()}")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

from keras import backend as k
import tensorflow as tf
# Define the DeepFool attack function in Keras
def deepfool_attack(model, x, num_classes=10, max_iter=100, epsilon=1):
    x_adv = tf.identity(x)  # Create a copy of the input using TensorFlow identity function

    for _ in range(max_iter):
        x_adv = tf.Variable(x_adv, dtype=tf.float32)  # Convert to TensorFlow variable
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv)
            target_label = tf.argmax(logits, axis=-1)

            target_onehot = tf.one_hot(target_label, num_classes)
            loss = tf.reduce_max(logits * (1 - target_onehot))

        grad = tape.gradient(loss, x_adv)
        grad_norm = tf.reduce_sum(tf.abs(grad))

        perturbation = epsilon * grad / grad_norm
        x_adv = x_adv + perturbation

    return x_adv.numpy()

# Generate adversarial examples using DeepFool attack for a few test samples
x_test_adv_deepfool = np.zeros_like(X_test[:100])

for i in range(5):
    x_adv = deepfool_attack(model, X_test[i:i+1],epsilon=0.3)
    x_test_adv_deepfool[i] = x_adv[0]

# Evaluate the model on the adversarial examples
adv_loss_deepfool, adv_accuracy_deepfool = model.evaluate(x_test_adv_deepfool, y_test[:100], verbose=0)
print(f"Adversarial Test Accuracy (DeepFool Attack): {adv_accuracy_deepfool}")


import matplotlib.pyplot as plt

# Display the original and adversarial digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('Original vs Adversarial Digits (DeepFool Attack)', fontsize=16)

for i in range(5):
    # Original digit
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f"Original: {y_test[i].argmax()}")

    # Adversarial digit
    axes[1, i].imshow(x_test_adv_deepfool[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f"Adversarial: {model.predict(x_test_adv_deepfool[i:i+1]).argmax()}")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()


epsilon = 1
num_steps = 100
step_size = 1  # Adjusted the step size proportionally
# Define the Basic Iterative Method (BIM) attack function in Keras

def bim_attack(model, x, y, epsilon=1, num_steps=10, step_size=0.01):
    x_adv = tf.identity(x)  # Create a copy of the input using TensorFlow identity function



    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv)
            loss = tf.keras.losses.categorical_crossentropy(y, logits)

        # Get the gradients of the loss with respect to the input
        grad = tape.gradient(loss, x_adv)

        # Compute the perturbation as the sign of the gradients
        perturbation = step_size * tf.sign(grad)
        x_adv = x_adv + perturbation

        # Clip the adversarial example to ensure it stays within the epsilon neighborhood of the original input
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)

    return x_adv.numpy()
# Generate adversarial examples using BIM attack for a few test samples
x_test_adv_bim = np.zeros_like(X_test[:100])

for i in range(5):
    x_adv = bim_attack(model, X_test[i:i+1], y_test[i:i+1], epsilon=0.1, num_steps=10, step_size=0.01)
    x_test_adv_bim[i] = x_adv[0]

# Evaluate the model on the adversarial examples
adv_loss_bim, adv_accuracy_bim = model.evaluate(x_test_adv_bim, y_test[:100], verbose=0)
print(f"Adversarial Test Accuracy (BIM Attack): {adv_accuracy_bim}")



import matplotlib.pyplot as plt

# Display the original and adversarial digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('Original vs Adversarial Digits (BIM Attack)', fontsize=16)

for i in range(5):
    # Original digit
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f"Original: {y_test[i].argmax()}")

    # Adversarial digit
    axes[1, i].imshow(x_test_adv_bim[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f"Adversarial: {model.predict(x_test_adv_bim[i:i+1]).argmax()}")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
          
