from __future__ import division
import argparse
import json
import numpy as np
import os
import tensorflow as tf

from speech import audio_producer, decoder
import speech_model
import streaming_model
from speech.utils import wave_utils

class KWS(object):

    def __init__(self, save_path, keyword,
                 window_size=800, step_size=100):
        """
        Stateful class for Key-word spotting.
        Args:
            save_path : Path to saved SpeechModel.
            keyword : String keyword to spot.
            window_size : Size of search window in milliseconds.
            step_size : Size of step between searches in milliseconds.
        """

        char_map = audio_producer.TOKEN_DICT
        self.keyword = [char_map[k] for k in keyword]

        config_path = os.path.join(save_path, "model.json")
        with open(config_path, 'r') as fid:
            config = json.load(fid)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            self.model = speech_model.SpeechModel()
            config['inference']['batch_size'] = 1
            self.model.init_inference(config['inference'])

            self.sample_rate = config['data']['sample_rate']

            saver = tf.train.Saver()

            model_path = os.path.join(save_path, "model.ckpt")
            saver.restore(self.session, model_path)
            self.streamer = streaming_model.StreamingSpeechModel(self.model,
                        self.session, config['data']['sample_rate'])
        self.reset()

        # We want the number of frames output from a 'same' convolution.
        self.window_size = window_size
        self.window_frames = self.output_size_from_ms(window_size)

        self.step_size = step_size

    def reset(self):
        """
        Reset KWS state. This should only be called when
        streaming non-sequential audio.
        """
        self.probs = np.empty((0, self.model.output_dim),
                               dtype=np.float32)
        self.state = self.streamer.initial_state()

    def output_size_from_ms(self, duration):
        """
        Computes the number of output frames for a given
        input duration in milliseconds.
        """
        # TODO, awni, this computation should be checked.
        seconds = duration / 1000
        in_frames = audio_producer._spec_time_steps(seconds)
        out_frames = np.ceil(in_frames / self.model.stride)
        return int(out_frames)

    def evaluate(self, data, sample_rate, vad=False):
        """
        Args:
            data : A 1D numpy array of wave data.
            sample_rate : The sample rate (Hz) of the data.
        """
        assert data.shape[0] == int(sample_rate * self.step_size / 1000), \
                "Can only update by step size - %d != %d * %d / 1000" % (data.shape[0], sample_rate, self.step_size)

        probs, self.state = self.streamer.propagate_packet(data,
                                sample_rate, self.state)

        self.probs = np.vstack([self.probs, probs])
        if self.probs.shape[0] < self.window_frames:
            if vad:
                return 0
            else:
                return float('inf')

        self.probs = self.probs[-self.window_frames:, :]
        if vad:
            ctc_score = -np.sum(np.log(self.probs[:, 0]))
        else:
            ctc_score = decoder.score_kws(self.probs, self.keyword,
                                          audio_producer.BLANK)
        return ctc_score

    def evaluate_wave(self, file_name, vad=False):
        data, sample_rate = wave_utils.read_wave_array(file_name)
        sample_step = int(self.step_size * sample_rate / 1000)
        sample_window = int(self.window_size * sample_rate / 1000)
        scores = []
        for i in range(0, data.shape[0] - sample_step + 1, sample_step):
            packet = data[i:i+sample_step]
            scores.append(self.evaluate(packet, sample_rate, vad=vad))
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True,
            help="Path where model is saved.")
    parser.add_argument("--wave_file", required=True,
            help="Audio wave file to decode.")
    args = parser.parse_args()

    kws = KWS(args.save_path, "olivia")
    print(kws.evaluate_wave(args.wave_file))
