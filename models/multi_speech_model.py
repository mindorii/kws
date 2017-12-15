from __future__ import division

import tensorflow as tf

import speech_model

class MultiSpeechModel(object):

    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self._init_inference = False
        self._init_cost = False
        self._init_train = False

    def init_inference(self, config):
        batch_size = config['batch_size']
        assert batch_size % self.num_gpus == 0, \
                "Batch size must be divisible by the number of GPUs."
        batch_per_gpu = batch_size // self.num_gpus

        self._models = []
        for i in range(self.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                model = speech_model.SpeechModel()
                config['batch_size'] = batch_per_gpu
                model.init_inference(config)
                tf.get_variable_scope().reuse_variables()
                self._models.append(model)

        self._init_inference = True

    def init_cost(self):
        assert self._init_inference, "Must init inference before cost."
        for i in range(self.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                self._models[i].init_cost()
                tf.get_variable_scope().reuse_variables()

        costs = [model.cost for model in self._models]
        zero = tf.constant(0.0)
        finite_costs = [tf.select(tf.is_finite(c), c, zero) for c in costs]
        self._cost =  tf.div(tf.add_n(finite_costs),
                             self.num_gpus,
                             # TODO, give the cost a name so we
                             # can change the denominator betwen on restore
                             name="truediv_8")
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

        grads = []
        for i in range(self.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                tvars = tf.trainable_variables()
                grads.append(tf.gradients(self._models[i].cost, tvars))
        average_grads = _average_gradients(grads)
        scaled_grads, norm = tf.clip_by_global_norm(average_grads, max_grad_norm)
        self._grad_norm = norm

        optimizer = tf.train.MomentumOptimizer(self.lr, self._momentum)
        with tf.control_dependencies([avg_cost_op]):
            self._train_op = optimizer.apply_gradients(zip(scaled_grads, tvars),
                                 global_step=step)

        self._init_train = True

    def feed_dict(self, inputs, labels=None):
        """
        Constructs the feed dictionary from given inputs necessary to run
        an operations for the model.

        Args:
            inputs : List of 2D numpy array input spectrograms. Should be
                of shape [input_dim x time]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If label=None does not feed the
                label placeholder (for e.g. inference only).

        Returns:
            A dictionary of placeholder keys and feed values.
        """
        feed_dict = {}
        batches = _split_batch(self.num_gpus, inputs, labels)
        for model, (i, l) in zip(self._models, batches):
            feed_dict.update(model.feed_dict(i, labels=l))
        return feed_dict

    def start_momentum(self, session):
        m = self._momentum.assign(self._momentum_val)
        session.run([m])

    def set_mean_std(self, mean, std, session):
        self._models[0].set_mean_std(mean, std, session)

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
    def input_dim(self):
        assert self._init_inference, "Must init inference."
        return self._models[0].input_dim

    @property
    def output_dim(self):
        assert self._init_inference, "Must init inference."
        return self._models[0].output_dim

    @property
    def train_op(self):
        assert self._init_train, "Must init train."
        return self._train_op

def _average_gradients(model_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of list of gradients for each model.
    Returns:
        List of gradients where each gradient has been averaged
        across all models.
    """
    average_grads = []
    for grads in zip(*model_grads):
        grads = [tf.expand_dims(g, 0) for g in grads]

        # Average over the 'model' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        average_grads.append(grad)

    return average_grads

def _split_batch(num_gpus, data, labels=None):
    """
        Split a set of data into batch_size // num_gpus batches.

        Args:
            inputs : List of 2D numpy array input spectrograms. Should be
                of shape [input_dim x time]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If labels=None the corresponding
                labels item for each batch will also be None.

        Returns:
            A num_gpus length list of (inputs, labels) of
            the same types as above but with batch_size // num_gpus
            entries in each.
    """
    batch_size = len(data)
    n = batch_size // num_gpus
    batches = []
    for i in range(0, batch_size, n):
        batch = [data[i:i + n], None]
        if labels:
            batch[1] = labels[i:i + n]
        batches.append(batch)
    return batches

