
from __future__ import division
from __future__ import print_function

import json
import math
import multiprocessing as mp
import numpy as np
import os
import random

from utils import wave_utils
from utils import cspecgram

# TODO, awni fix this. Baking the tokens in for now
BLANK = 0
TOKENS = list("abcdefghijklmnopqrstuvwxyz \'")
TOKEN_DICT = dict(zip(TOKENS, range(1, len(TOKENS) + 1)))

class AudioProducer(object):

    def __init__(self, data_jsons, batch_size, sample_rate=8000,
                 min_duration=.3, max_duration=16.0):
        """
        Args:
            data_json : List of paths to files with speech data in json
                format. Each line should be a new example and the required
                fields for each example are 'duration' (seconds) is the
                length of the audio, 'key' is the path to the wave file
                and 'text' is the transcription.
            batch_size : Size of the batches for training.
            sample_rate : Rate to resample audio prior to feature computation.
            min_duration : Minimum length of allowed audio in seconds.
            max_duration : Maximum length of allowed audio in seconds.
        """
        self.tokens = TOKEN_DICT
        self.batch_size = batch_size
        self.sample_rate = sample_rate

        data = []
        for dj in data_jsons:
            data.extend(_read_data_json(dj))
        data = sorted(data, key=lambda x : x['duration'])

        def bad_data_fn(d):
            if d['duration'] > max_duration or d['duration'] < min_duration:
                return False
            # For CTC number of input steps has to be greater
            # than number of labels.
            # TODO, awni, better way to do this.
            if len(d['text']) >= _spec_time_steps(d['duration']) / 3:
                return False
            return True

        data = filter(bad_data_fn, data)

        # *NB* this cuts off the longest data items in the last segment
        # if len(data) is not a multiple of batch_size
        batches = [data[i:i+batch_size]
                   for i in range(0, len(data) - batch_size + 1, batch_size)]
        random.shuffle(batches)
        self.batches = batches

    def estimate_mean_std(self, sample_size=2048):
        keys =  [random.choice(random.choice(self.batches))['key']
                 for _ in range(sample_size)]
        feats = np.hstack([compute_features(k, self.sample_rate)
                           for k in keys])
        mean = np.mean(feats, axis=1)
        std = np.std(feats, axis=1)
        return mean, std

    def iterator(self, sort=False, max_size=64,
                 num_workers=8, max_examples=None):
        if sort:
            batches = sorted(self.batches,
                             key=lambda d : d[-1]['duration'])
        else:
            batches = self.batches

        if max_examples is not None:
            batches = batches[:int(max_examples / self.batch_size)]

        consumer = mp.Queue()
        producer = mp.Queue(max_size)
        for b in batches:
            consumer.put(b)

        procs = [mp.Process(target=queue_featurize_example,
                            args=(consumer, producer,
                                  self.sample_rate, self.tokens))
                 for _ in range(num_workers)]
        [p.start() for p in procs]

        for _ in batches:
            yield producer.get()

    @property
    def alphabet_size(self):
        return len(self.tokens)

    @property
    def input_dim(self):
        return _spec_freq_dim(self.sample_rate)

def to_ints(text):
    return [TOKEN_DICT[t] for t in text]

def to_text(inds):
    return ''.join([TOKENS[i-1] for i in inds])

def queue_featurize_example(consumer, producer, sample_rate, tokens):
    while True:
        try:
            batch = consumer.get(block=True, timeout=5)
        except mp.queues.Empty as e:
            return
        labels = [[tokens[t] for t in b['text']] for b in batch]
        inputs = [compute_features(b['key'], sample_rate) for b in batch]
        producer.put((inputs, labels))

def compute_features(audio_file, sample_rate):
    data, d_sample_rate = wave_utils.read_wave_array(audio_file)
    return compute_features_raw(data, sample_rate, d_sample_rate)

def compute_features_raw(data, output_sample_rate,
                         input_sample_rate):
    if input_sample_rate != output_sample_rate:
        data = wave_utils.resample(data, input_sample_rate,
                                   output_sample_rate)
    epsilon = 1e-7
    features = _compute_specgram(data, output_sample_rate)
    features = np.log(features + epsilon)
    return features

def _read_data_json(file_name):
    with open(file_name, 'r') as fid:
        return [json.loads(l) for l in fid]

def _spec_time_steps(duration, window_size=25, hop_size=10):
    """
    Compute the number of time steps of a spectrogram.

    Args:
        duration : Length of audio in seconds.
        window_size : Size of specgram window in milliseconds.
        hop_size : Size of steps between ffts in
            specgram in milliseconds.
    Returns:
        The number of time-steps in the
        output of the spectrogram.
    """
    duration_ms = duration * 1000
    return math.ceil((duration_ms - window_size) / hop_size)

def _spec_freq_dim(sample_rate, window_size=25):
    """
    Compute the number of frequency bins of a spectrogram.

    Args:
        sample_rate : Hz of the audio.
        window_size : Size of specgram window in milliseconds.
    Returns:
        An integer representing the number of dimensions in
        the spectrogram.
    """
    return int(((sample_rate / 1000) * window_size) / 2) + 1

def _compute_specgram(data, sample_rate,
                      window_size=25, hop_size=10):
    """
    Compute the spectrogram of the given audio file.

    Args:
        data : 1D numpy array with wave data.
        sample_rate : rate to resample audio to before specgram.
        window_size : size in milliseconds of the stft window.
        hop_size : size in milliseconds of the step between windows.

    Returns:
        spectrum : M x N numpy array where M is the number of
            spectral bins and N is the number of time-steps.

    """

    if data.dtype != np.int16:
        data = data.astype(np.int16)

    NFFT = int(window_size * (sample_rate / 1000))
    noverlap = int((window_size - hop_size) * (sample_rate / 1000))
    n_freqs = int(NFFT / 2 + 1)
    time_steps = int((data.size - noverlap) / (NFFT - noverlap))

    spectrum = np.empty((n_freqs, time_steps),
                        dtype=np.float32, order='F')
    cspecgram.specgram(data, NFFT, sample_rate,
                       noverlap, spectrum)
    return spectrum

