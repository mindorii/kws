from __future__ import division

import math
import numpy as np
import tensorflow as tf
from user_ops import warp_ctc_ops

class SpeechModel(object):

    def __init__(self):
        self._init_inference = False
        self._init_cost = False
        self._init_train = False

    def init_inference(self, config):
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        batch_size = config['batch_size']
        self._input_dim = input_dim = config['input_dim']
        self._output_dim = output_dim = config['alphabet_size'] + 1

        # 1st layer convolution params
        num_filters = config['num_filters']
        self._temporal_context = temporal_context = config['temporal_context']
        frequency_context = config.get('frequency_context') or input_dim
        padding = config.get('padding') or 'VALID'
        self._stride = stride = config['stride']

        # TODO, awni, move these into a config
        init_scale = 1e-2
        initializer = tf.random_uniform_initializer(minval=-init_scale,
                        maxval=init_scale)

        self._inputs = tf.placeholder(tf.float32, [batch_size, None, input_dim])
        self._seq_lens = tf.placeholder(tf.int32, shape=batch_size)

        # TODO, awni, for now on the client to remember to initialize these.
        self._mean = tf.get_variable("mean",
                        shape=input_dim, trainable=False)
        self._std = tf.get_variable("std",
                        shape=input_dim, trainable=False)

        std_inputs = (self._inputs - self._mean) / self._std

        with tf.variable_scope("conv1", initializer=initializer):
            filt_shape = (temporal_context, frequency_context,
                          1, num_filters)
            filters = tf.get_variable("filters", shape=filt_shape)
            bias = tf.get_variable("bias", shape=num_filters)

        std_inputs = tf.reshape(std_inputs, (batch_size, -1, input_dim, 1))
        conv = tf.nn.conv2d(std_inputs, filters,
                            strides=[1, stride, stride, 1],
                            padding=padding)

        # Compute sequence lengths after conv application.
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias))
        out_lens = _conv_out_size_op(self._seq_lens,
                        temporal_context, stride, padding=padding)
        self._out_lens = out_lens

        num_responses = _conv_out_size(input_dim, frequency_context,
                                       stride, padding=padding)
        conv_out = tf.reshape(conv_out,
                    [batch_size, -1, num_filters * num_responses])

        # Convert to time-major for RNN and CTC
        rnn_in = tf.transpose(conv_out, [1, 0, 2])

        with tf.variable_scope("rnn", initializer=initializer):
            layers = [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers)
            self._initial_state = cells.zero_state(batch_size, tf.float32)

        (rnn_out, state) = tf.nn.dynamic_rnn(cells, rnn_in, out_lens,
                            time_major=True, dtype=tf.float32,
                            initial_state=self._initial_state)
        self._rnn_state = state

        # Collapse time and batch dims pre softmax.
        rnn_out = tf.reshape(rnn_out, (-1, hidden_size))
        logits, probas = _add_softmax_linear(rnn_out, hidden_size,
                                             output_dim, initializer)
        # Reshape to time-major.
        self._logits = tf.reshape(logits, (-1, batch_size, output_dim))
        self._probas = tf.reshape(probas, (-1, batch_size, output_dim))

        self._init_inference = True

    def init_cost(self):
        assert self._init_inference, "Must init inference before cost."

        self._labels = tf.placeholder(tf.int32)
        self._label_lens = tf.placeholder(tf.int32)

        losses = warp_ctc_ops.warp_ctc_loss(self.logits, self._out_lens,
                                            self._labels, self._label_lens)
        self._cost = tf.reduce_mean(losses)

        self._init_cost = True

    def init_train(self, config):
        assert self._init_inference, "Must init inference before train."
        assert self._init_cost, "Must init cost before train."

        learning_rate = config['learning_rate']
        self._momentum_val = config['momentum']
        max_grad_norm = config['max_grad_norm']
        decay_steps = config['lr_decay_steps']
        decay_rate = config['lr_decay_rate']

        self._momentum = tf.Variable(0.5, trainable=False)
        self._global_step = step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate, step,
                    decay_steps, decay_rate, staircase=True)

        ema = tf.train.ExponentialMovingAverage(0.99, name="avg")
        avg_cost_op = ema.apply([self.cost])
        self._avg_cost = ema.average(self.cost)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        scaled_grads, norm = tf.clip_by_global_norm(grads, max_grad_norm)

        optimizer = tf.train.MomentumOptimizer(self.lr, self._momentum)
        with tf.control_dependencies([avg_cost_op]):
            self._train_op = optimizer.apply_gradients(zip(scaled_grads, tvars),
                                 global_step=step)

        self._grad_norm = norm
        self._init_train = True

    def feed_dict(self, inputs, labels=None, rnn_state=None):
        """
        Constructs the feed dictionary from given inputs necessary to run
        an operations for the model.

        Args:
            inputs : List of 2D numpy array input spectrograms. Should be
                of shape [input_dim x time]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If label=None does not feed the
                label placeholder (for e.g. inference only).
            rnn_state : State arrays to feed the RNN state.
                (Size and shape depends on configuration.)

        Returns:
            A dictionary of placeholder keys and feed values.
        """
        sequence_lengths = [d.shape[1] for d in inputs]
        feed_dict = { self._inputs : _batch_major(inputs),
                      self._seq_lens : sequence_lengths}
        if labels:
            values = [l for label in labels for l in label]
            label_lens = [len(label) for label in labels]
            label_dict = { self._labels : values,
                           self._label_lens : label_lens }
            feed_dict.update(label_dict)
        if rnn_state is not None:
            for i, r in zip(self._initial_state, rnn_state):
                feed_dict[i] = r

        return feed_dict

    def start_momentum(self, session):
        m = self._momentum.assign(self._momentum_val)
        session.run([m])

    def set_mean_std(self, mean, std, session):
        m = self._mean.assign(mean)
        s = self._std.assign(std)
        session.run([m, s])

    @property
    def cost(self):
        assert self._init_cost, "Must init cost."
        return self._cost

    @property
    def avg_cost(self):
        assert self._init_train, "Must init train."
        return self._avg_cost

    @property
    def grad_norm(self):
        assert self._init_train, "Must init train."
        return self._grad_norm

    @property
    def global_step(self):
        assert self._init_train, "Must init train."
        return self._global_step

    @property
    def initial_state(self):
        assert self._init_inference, "Must init inference."
        return self._initial_state

    @property
    def input_dim(self):
        assert self._init_inference, "Must init inference."
        return self._input_dim

    @property
    def logits(self):
        assert self._init_inference, "Must init inference."
        return self._logits

    @property
    def output_dim(self):
        assert self._init_inference, "Must init inference."
        return self._output_dim

    @property
    def output_lens(self):
        assert self._init_inference, "Must init inference."
        return self._out_lens

    @property
    def probabilities(self):
        assert self._init_inference, "Must init inference."
        return self._probas

    @property
    def state(self):
        assert self._init_inference, "Must init inference."
        return self._rnn_state

    @property
    def stride(self):
        assert self._init_inference, "Must init inference."
        return self._stride

    @property
    def temporal_context(self):
        assert self._init_inference, "Must init inference."
        return self._temporal_context

    @property
    def train_op(self):
        assert self._init_train, "Must init train."
        return self._train_op

def _add_softmax_linear(inputs, input_dim, output_dim, initializer):
    with tf.variable_scope("softmax", initializer=initializer):
        W_softmax = tf.get_variable("softmax_W",
                        shape=(input_dim, output_dim))
        b_softmax = tf.get_variable("softmax_b", shape=(output_dim),
                        initializer=tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(inputs, W_softmax), b_softmax)
    probas = tf.nn.softmax(logits)
    return logits, probas


def _conv_out_size(input_size, filter_size, stride,
                   padding="VALID"):
    """
    Computes the output size for a given input size after
    applying a 1D convolution with stride.

    Args:
        input_size : integer input size.
        filter_size : size of 1D filter.
        stride : step size between filter applications.
        padding : type of zero-padding used in the convolution.

    Returns:
        Integer output size.
    """
    if padding == "VALID":
        sizes = (input_size - filter_size + 1)
    elif padding == "SAME":
        sizes = input_size
    else:
        raise ValueError("Invalid padding: {}".format(padding))
    return int(math.ceil(sizes / stride))

def _conv_out_size_op(input_sizes, filter_size, stride,
                      padding="VALID"):
    """
    **Same as _conv_out_size but operates on tensors.**

    Computes the output sizes on a list of input sizes after
    applying a 1D convolution with stride.

    Args:
        input_sizes : 1D tensor of input sizes.
        filter_size : size of 1D filter.
        stride : step size between filter applications.
        padding : type of zero-padding used in the convolution.

    Returns:
        A tensor of type int32 with the output sizes.
    """
    input_sizes = tf.cast(input_sizes, tf.float32)
    if padding == "VALID":
        sizes = (input_sizes - filter_size + 1)
    elif padding == "SAME":
        sizes = input_sizes
    else:
        raise ValueError("Invalid padding: {}".format(padding))
    return tf.cast(tf.ceil(sizes / stride), tf.int32)


def _batch_major(data):
    """
    Reshapes a batch of spectrogram arrays into
    a single tensor [batch_size x max_time x input_dim].

    Args :
        data : list of 2D numpy arrays of [input_dim x time]
    Returns :
        A 3d tensor with shape
        [batch_size x max_time x input_dim]
        and zero pads as necessary for data items which have
        fewer time steps than max_time.
    """
    max_time = max(d.shape[1] for d in data)
    batch_size = len(data)
    input_dim = data[0].shape[0]
    all_data = np.zeros((batch_size, max_time, input_dim),
                        dtype=np.float32)
    for e, d in enumerate(data):
        all_data[e, :d.shape[1], :] = d.T
    return all_data

