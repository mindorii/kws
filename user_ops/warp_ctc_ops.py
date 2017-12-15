from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

import os
wso = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'warp_ctc.so')
warp_ctc = tf.load_op_library(wso)

def warp_ctc_loss(inputs, input_lengths, labels, label_lengths):
  """Computes the CTC (Connectionist Temporal Classification) Loss.

  Args:
    inputs: 3-D `float` `Tensor` sized
      `[max_time x batch_size x num_classes]`.  The logits.
    labels:

  Returns:
    A 1-D `float` `Tensor`, size `[batch]`, containing losses.

  """
  loss, _ = warp_ctc.warp_ctc(inputs, input_lengths,
                              labels, label_lengths)
  return loss


@ops.RegisterGradient("WarpCtc")
def _WarpCtcGrad(op, grad_loss, _):
  # Outputs are: loss, grad
  grad = op.outputs[1]
  # Return gradient for inputs and None for
  # input_lengths, labels and label_lenghts
  grad_loss = tf.reshape(grad_loss, (1, -1, 1))
  return [tf.multiply(grad_loss, grad),
          None, None, None]


@ops.RegisterShape("WarpCtc")
def _WarpCtcShape(op):

    inputs_shape = op.inputs[0].get_shape().with_rank(3)
    batch_size = inputs_shape[1]
    # loss, gradient
    return [tensor_shape.vector(batch_size), inputs_shape]

