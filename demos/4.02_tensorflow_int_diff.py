"""
See section 4.2: Problems with Integer Tensor Differentiation
This demonstrates how TensorFlow doesn't support 
differentiation for integer tensors.
"""
import tensorflow as tf

x = tf.Variable(1)
y = tf.Variable(2)

with tf.GradientTape() as tape:
    z = x + y

grad = tape.gradient(z, [x, y])
print("Gradients w.r.t. x and y:", grad)

# Output:
# Gradients w.r.t. x and y: [None, None]