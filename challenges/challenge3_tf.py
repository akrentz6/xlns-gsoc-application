import tensorflow as tf
import xlns as xl

# Constants for the base and zero internal representation
xlnsB_tf = tf.constant(xl.xlnsB, dtype=tf.float64)
xlns0_tf = tf.constant(-0x7fffffffffffffff, dtype=tf.int64)

def xlns_to_tf_const(xlns_array):
    return tf.constant(xlns_array.nd, dtype=tf.int64)

def tf_const_to_xlns(tf_tensor):
    
    internal_data = tf_tensor.numpy()
    fp_values = []
    
    # Set the internal data of each xlns object
    for value in internal_data:
        x_obj = xl.xlns(0)
        x_obj.x = value >> 1
        x_obj.s = value & 1
        fp_values.append(x_obj)
    
    return xl.xlnsnp(fp_values)

# Logarithm term of addition equation
@tf.function
def tf_sbdb_ideal(d, s, B=None):

    d_f = tf.cast(d, tf.float64)
    s_f = tf.cast(s, tf.float64)

    base = xlnsB_tf if B is None else tf.cast(B, tf.float64)
    term = tf.pow(base, d_f)

    v = tf.abs(1.0 - 2.0 * s_f + term)
    log_val = tf.math.log(v) / tf.math.log(base)

    result = tf.bitwise.left_shift(tf.cast(tf.round(log_val), tf.int64), 1)
    return result

@tf.function
def tf_logadd(x, y):

    # Our formulae suppose the log part of x is greater than that of y.
    # We are able to take the maximum of the full internal representation
    # since the sign bit is the least significant bit so doesn't matter. 
    part1 = tf.math.maximum(x, y)

    d = -tf.abs(tf.bitwise.right_shift(x, 1) - tf.bitwise.right_shift(y, 1))
    s = tf.bitwise.bitwise_and(tf.bitwise.bitwise_xor(x, y), 1)
    
    part2 = tf_sbdb_ideal(d, s)

    return part1 + part2

@tf.function
def tf_myadd(x, y):

    # Handles logic for when their sum equals 0. In this case, the 
    # internal representations differ only in their LSB, i.e. x^y=1.
    sum_to_zero_cond = tf.where(tf.equal(tf.bitwise.bitwise_xor(x, y), 1),
                               xlns0_tf, tf_logadd(x, y))
    
    # Handles logic for when either term equals 0.
    y_equals_zero_cond = tf.where(tf.equal(tf.bitwise.bitwise_or(y, 1),
                                           xlns0_tf),
                                 x, sum_to_zero_cond)
    x_equals_zero_cond = tf.where(tf.equal(tf.bitwise.bitwise_or(x, 1),
                                           xlns0_tf),
                                 y, y_equals_zero_cond)
    
    return x_equals_zero_cond

if __name__ == "__main__":

    # Test data including edge cases (negatives and numbers that sum to 0)
    # x = tf.Tensor([16777216 26591258 -9223372036854775808 0])
    # y = tf.Tensor([0 1 16777216 1])
    x = xlns_to_tf_const(xl.xlnsnp([2,  3,  0,  1]))
    y = xlns_to_tf_const(xl.xlnsnp([1, -1,  2, -1]))

    result = tf_myadd(x, y)
    print("TensorFlow myadd result:", tf_const_to_xlns(result))