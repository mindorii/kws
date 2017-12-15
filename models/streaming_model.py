from __future__ import division

import numpy as np

import audio_producer

class SpeechModelState(object):

    def __init__(self, rnn_state):
        self.rnn_state = rnn_state
        self.audio = None
        self.frames = None

class StreamingSpeechModel(object):

    def __init__(self, model, session, sample_rate):
        self.model = model
        self.session = session

        self.sample_rate = sample_rate

        self.context = model.temporal_context
        self.stride = model.stride

        self.spec_window = 25
        self.spec_hop = 10
        self.ops = [model.probabilities, model.state]

    def initial_state(self):
        rnn_state = (i.eval(session=self.session)
                        for i in self.model.initial_state)
        return SpeechModelState(rnn_state)

    def propagate_packet(self, packet, sample_rate, state):
        model = self.model
        if state.audio is not None:
            packet = np.hstack([state.audio, packet])

        feats = audio_producer.compute_features_raw(packet,
                                                self.sample_rate,
                                                sample_rate)
        if state.frames is not None:
            feats = np.hstack([state.frames, feats])

        if feats.shape[1] >= self.context:
            feed_dict = model.feed_dict([feats], rnn_state=state.rnn_state)
            res = self.session.run(self.ops, feed_dict)
            probs, rnn_state = res
            probs = probs.squeeze(axis=1)
        else:
            probs = np.empty((0, model.output_dim), dtype=np.float32)
            rnn_state = None

        ## Compute the audio overlap pre specgram
        n_window = int(self.spec_window * sample_rate / 1000)
        n_hop = int(self.spec_hop * sample_rate / 1000)
        skip = n_hop - (packet.shape[0] - n_window) % n_hop
        n_overlap = n_window - skip
        state.audio = packet[-n_overlap:]

        ## Compute the features overlap pre convolution
        skip = self.stride - (feats.shape[1] - self.context) % self.stride
        frame_overlap = self.context - skip
        state.frames = feats[:, -frame_overlap:]

        state.rnn_state = rnn_state

        return probs, state

