from __future__ import division

import numpy as np
import wave
import scikits.samplerate as ssr

def write_wave_array(data, file_name, sample_rate):
    wv = wave.open(file_name, 'w')
    wv.setframerate(sample_rate)
    wv.setnchannels(1)
    wv.setsampwidth(2)
    data = data.astype(np.int16)
    wv.writeframes(data.tostring())
    wv.close()

def read_wave_array(audio_file):
    """
    Reads wave file into numpy array.
    Audio must be single-channel with 2-byte samples.
    """
    wv = wave.open(audio_file)
    assert wv.getsampwidth() == 2, "Bad sample width."
    frames = wv.readframes(-1)
    data = np.fromstring(frames, dtype=np.int16)
    sample_rate = wv.getframerate()
    if wv.getnchannels() == 2:
        data = data[::2]
    wv.close()
    return data, sample_rate

def resample(data, in_rate, out_rate):
    assert in_rate >= out_rate, \
            "Shouldn't use this resampler unless downsampling."
    return ssr.resample(data, out_rate / in_rate, 'linear')
