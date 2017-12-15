
import tensorflow as tf

from speech.user_ops import warp_ctc_ops

class CTC(object):
    """
    Simple class to do ctc scoring on a single example.
    """

    def __init__(self, session):
        self.session = session

        self.logits = tf.placeholder(tf.float32)
        self.seq_lens = tf.placeholder(tf.int32)
        self.labels = tf.placeholder(tf.int32)
        self.label_lens = tf.placeholder(tf.int32)

        self.ctc = warp_ctc_ops.warp_ctc_loss(self.logits, self.seq_lens,
                                              self.labels, self.label_lens)

    def score(self, logits, labels):
        """
        CTC score the labels against the logits.
        *NB* the logits should be unnormalized.

        Args:
            logits : 2D numpy array of shape [time x outputs]
            labels : 1D list of integer labels
        Returns:
            The negative log-likelihood CTC score.
        """
        logits = logits.reshape([logits.shape[0], 1, logits.shape[1]])
        feed_dict = { self.logits : logits,
                      self.seq_lens : [logits.shape[0]],
                      self.labels : labels,
                      self.label_lens : [len(labels)] }
        res = self.session.run([self.ctc], feed_dict)
        return res[0][0]


