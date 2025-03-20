import tensorflow as tf
import keras
import time

# Load and preprocess MNIST data:
#   - Flatten 28x28 images into 784-length vectors.
#   - Normalize pixel values to [0, 1]
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Build the model as a Sequential Keras model:
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(784,)),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer=tf.keras.initializers.RandomNormal(
                              mean=0.0, stddev=0.1),
                              bias_initializer="zeros"),
    tf.keras.layers.Dense(10, activation="softmax",
                          kernel_initializer=tf.keras.initializers.RandomNormal(
                              mean=0.0, stddev=0.1),
                              bias_initializer="zeros")
])

# Compile the model:
#   - Use sparse categorical crossentropy since labels are integer-encoded.
#   - The SGD optimizer mimics the update rule in arn_generic.py.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model for seven epochs:
start = time.time()
history = model.fit(x_train, y_train, epochs=7, batch_size=128, validation_data=(x_test, y_test))
elapsed = time.time() - start

# Evaluate on test data:
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy (TensorFlow):", test_acc)
print("Elapsed time:", elapsed)