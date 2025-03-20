import tensorflow as tf
import xlns as xl

# Create an xlns object
xlns_data = xl.xlnsnp([2.0, 4.0, 5.0])
print("xlns data:", xlns_data)

# Try to create a TensorFlow constant directly from an xlns object
try:
    tf_data = tf.constant(xlns_data)
except ValueError as e:
    # Display only the first line of the error message
    error_str = str(e).splitlines()[0]
    print("\nTensorFlow error when passing xlns object:", error_str)

# The correct approach is to convert to floating point first:
fp_data = [float(x) for x in xlns_data.xlns()]
tf_data = tf.constant(fp_data)
print("\nConverted TensorFlow constant:", tf_data)