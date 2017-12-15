from models.kws import KWS
import numpy as np
import subprocess
import os
import gevent

THRESHOLD = 2.0
kws = None
prior_scores = None

def to_pcm_s16le_8khz(filepath):
    name = filepath + "_raw"
    subprocess.call(['avconv', '-y', '-i', filepath, '-ac', '1', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '8k', name])
    return name

def denoise(filepath, media_playback_filepath):
    filepath_denoised = "/tmp/denoised.raw"
    filepath_denoised_wav = "/tmp/denoised" + ".wav"

    greenlets = [gevent.spawn(to_pcm_s16le_8khz, path) for path in filepath, media_playback_filepath]
    gevent.joinall(greenlets)

    filepath_raw, media_playback_filepath_raw = [greenlets[0].value, greenlets[1].value]

    subprocess.call(['/vagrant/core/third_party/rel643_aec_x64_demo/aecdemo', '-mic', filepath_raw,
                     '-spk', media_playback_filepath_raw, '-o', filepath_denoised])

    subprocess.call(['avconv', '-y', '-f', 's16le', '-ar', '8k', '-ac', '1', '-i', filepath_denoised, filepath_denoised_wav])

    return filepath_denoised_wav

# Use globals trick to prevent TensorFlow executor hang when executing on multiple threads.
def recognize(filepath, media_playback_filepath=None, keyword="olivia"):
    global kws
    global prior_scores

    if not kws:
        kws = KWS(
            save_path=(os.path.join(os.environ["MODEL_PATH"], "kws_noise_2")),
            keyword=keyword,
            window_size=800,
            step_size=92.88) # (8192 / 44100) / 2

    if media_playback_filepath:
        filepath = denoise(filepath, media_playback_filepath)

    with open(filepath) as f:
        this_turn_scores = kws.evaluate_wave(f)

    if prior_scores is not None:
       scores = prior_scores + this_turn_scores
    else:
        scores = this_turn_scores
    print scores

    result_dict = {}

    if any(s < THRESHOLD for s in scores):
        result_dict['text'] = "yes"
        result_dict['confidence'] = 1.
        result_dict['offset_ms'] = offset_ms(kws, scores)
        kws.reset()
        prior_scores = None
    else:
        result_dict['text'] = "no"
        result_dict['confidence'] = 1.
        prior_scores = this_turn_scores

    #result_dict['scores'] = [float(score) for score in scores]

    return result_dict['text'], result_dict['confidence'], result_dict

def offset_ms(kws, scores):
    """
    Return the offset of the end of the keyword utterance
    from the beginning of the audio file in millis.
    """
    # jump back half the window size from the end of the
    # best scoring packet id.
    idx = np.argmin(scores)
    offset_ms = kws.step_size * (idx + 1) - 0.5 * kws.window_size
    return offset_ms

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the keyword spotter")
    parser.add_argument('file_path', type=str,
        help='file path to wave file to recognize', default=None)
    parser.add_argument('--media_file_path', type=str, help='Media playback reference file', default=None)
    args = parser.parse_args()
    print recognize(args.file_path, args.media_file_path)
