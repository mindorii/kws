from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import time

import warp_ctc_ops

class CTC(object):
    def __init__(self):
        self.logits = tf.placeholder(tf.float32)

        self.indices = tf.placeholder(tf.int64)
        self.values = tf.placeholder(tf.int32)
        self.shape = tf.placeholder(tf.int64)
        labels = tf.SparseTensor(self.indices, self.values, self.shape)

        self.seq_lens = tf.placeholder(tf.int32)

        self.ctc = tf.nn.ctc_loss(labels, self.logits, self.seq_lens)

    def feed_dict(self, logits, labels, seq_lens):
        indices = [[b, t] for b, label in enumerate(labels)
                            for t in range(len(label))]
        values = [l for label in labels for l in label]
        max_len = max(len(label) for label in labels)
        shape = [len(labels), max_len]
        feed_dict = { self.logits : logits,
                      self.seq_lens : seq_lens,
                      self.indices : indices,
                      self.values : values,
                      self.shape : shape }
        return feed_dict

class WarpCTC(object):
    def __init__(self):
        self.logits = tf.placeholder(tf.float32)

        self.labels = tf.placeholder(tf.int32)
        self.label_lens = tf.placeholder(tf.int32)
        self.seq_lens = tf.placeholder(tf.int32)

        self.ctc = warp_ctc_ops.warp_ctc_loss(self.logits, self.seq_lens,
                                              self.labels, self.label_lens)

    def feed_dict(self, logits, labels, seq_lens):
        values = [l + 1 for label in labels for l in label]
        label_lens = [len(label) for label in labels]
        feed_dict = { self.logits : logits,
                      self.seq_lens : seq_lens,
                      self.labels : values,
                      self.label_lens : label_lens}
        return feed_dict


def fake_data_batch(batch_size, max_time, output_dim, max_label_steps):

    data = np.random.randn(max_time, batch_size, output_dim)
    labels = [list(np.random.randint(0, output_dim - 1, max_label_steps))
              for _ in range(batch_size)]
    seq_lens = [max_time] * batch_size
    return data, labels, seq_lens

def benchmark(ctc, batch):
    logits, labels, seq_lens = batch

    with tf.Session() as session:
        start = time.time()
        for _ in range(num_its):
            feed_dict = ctc.feed_dict(logits, labels, seq_lens)
            costs, = session.run([ctc.ctc], feed_dict)
        end = time.time()
        total_time = end - start
        avg_batch_time = total_time / num_its
        print("Total time", total_time)
        print("Avg time per batch", avg_batch_time)

def check_ctc():
    gt_cost = 0.085191089113
    gt_shape = (2, 1, 5)

    with tf.Session():
        inputs = tf.constant([[[0.1, 0.6, 0.1, 0.1, 0.1]],
                              [[0.1, 0.1, 0.6, 0.1, 0.1]]])
        input_lengths = tf.constant([2], dtype=tf.int32)
        labels = tf.constant([1, 2], dtype=tf.int32)
        label_lengths = tf.constant([2], dtype=tf.int32)
        loss, grad = warp_ctc_ops.warp_ctc.warp_ctc(inputs, input_lengths,
                                                    labels, label_lengths)
        p_cost = math.exp(-loss.eval()[0])

        assert abs((p_cost - gt_cost) / gt_cost) < 1e-8, \
                "Cost incorrect."
        assert grad.eval().shape == gt_shape, "Shape mismatch."

if __name__ == "__main__":

    np.random.seed(10)

    # Correctness test
    with tf.device('/cpu:0'):
        check_ctc()
    with tf.device('/gpu:0'):
        check_ctc()

    # Performance test
    output_dim = 29
    max_time = 500
    batch_size = 256
    max_label_steps = 100
    num_its = 10

    batch = fake_data_batch(batch_size, max_time,
                            output_dim, max_label_steps)

    ctc = CTC()
    print("Benchmarking TF CTC")
    benchmark(ctc, batch)

    with tf.device('/cpu:0'):
        warp_ctc = WarpCTC()
        print("Benchmarking Warp CTC CPU")
        benchmark(warp_ctc, batch)

    with tf.device('/gpu:0'):
        warp_ctc = WarpCTC()
        print("Benchmarking Warp CTC GPU")
        benchmark(warp_ctc, batch)
