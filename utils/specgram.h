#pragma once

/* Returns the shape of a specgram for the given
 * inputs.
 * Params:
 *   n : number of samples in the input.
 *   NFFT : number of samples in the stft.
 *   noverlap : samples overlapped between stft's.
 *
 * Returns:
 *   A pair of ints. The first is the number of
 *   dimensions and the second is the number of
 *   time steps.
 */
std::pair<int, int> specgram_shape(int n, int NFFT,
                                   int noverlap);

/* Computes the spectrogram for a given input.
 * Params:
 *   in : input array of 2 byte ints.
 *   n : number of samples in the input.
 *   NFFT : number of samples in the stft.
 *   Fs : sample rate of the audio (Hz).
 *   noverlap : samples overlapped between stft's.
 *   out : float memory buffer for the output.
 */
void cspecgram(const int16_t* const in, int n,
              int NFFT, int Fs,
              int noverlap, float* out);

/* Computes the log spectrogram for a given input.
 * Params:
 *   in : input array of 2 byte ints.
 *   n : number of samples in the input.
 *   NFFT : number of samples in the stft.
 *   Fs : sample rate of the audio (Hz).
 *   noverlap : samples overlapped between stft's.
 *   out : float memory buffer for the output.
 *   epsilon : add small constant prior to log.
 */
void log_cspecgram(const int16_t* const in, int n,
                  int NFFT, int Fs,
                  int noverlap, float* out,
                  float epsilon=1e-7);
