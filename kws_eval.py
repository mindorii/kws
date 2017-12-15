from __future__ import print_function

import argparse
import numpy as np

import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True,
            help="Path where model is saved.")
    parser.add_argument("--file_list", required=True,
            help="Path to list of wave files.")
    args = parser.parse_args()

    kws = models.KWS(args.save_path, "olivia",
                      window_size=800, step_size=200)

    with open(args.file_list) as fid:
        wave_files = [l.strip() for l in fid]

    min_scores = []
    for e, wv in enumerate(wave_files):
        scores = kws.evaluate_wave(wv)
        kws.reset()
        min_scores.append(min(scores))
        print(e, min(scores))
    print("Average", np.mean(min_scores))
    print("Std", np.std(min_scores))
    print("Max", max(min_scores))
