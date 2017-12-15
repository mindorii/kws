#include <cmath>
#include <fftw3.h>
#include <tuple>
#include <vector>

#include "specgram.h"

std::vector<float> window_hanning(int size) {
    std::vector<float> window;
    float scale =  2 * M_PI / (size - 1);
    for (int n = 0; n < size; n++)
        window.push_back(0.5 * (1 - cos(scale * n)));
    return window;
}

std::pair<int, int> specgram_shape(int n, int NFFT,
                                   int noverlap) {
    int stride = NFFT - noverlap;
    int time_steps = (n - noverlap) / stride;
    int n_freqs = NFFT / 2 + 1;
    return std::make_pair(n_freqs, time_steps);
}

void log_cspecgram(const int16_t* const in, int n,
                  int NFFT, int Fs,
                  int noverlap, float* out,
                  float epsilon /* default=1e-7 */) {
    cspecgram(in, n, NFFT, Fs, noverlap, out);
    auto spec_shape = specgram_shape(n, NFFT, noverlap);
    int size = spec_shape.first * spec_shape.second;
    for (int i = 0; i < size; i++)
        out[i] = log(out[i] + epsilon);
}

void cspecgram(const int16_t* const in, int n,
              int NFFT, int Fs,
              int noverlap, float* out) {

    int stride = NFFT - noverlap;
    int n_freqs;
    int time_steps;
    std::tie(n_freqs, time_steps) = specgram_shape(n, NFFT, noverlap);

    std::vector<float> window(window_hanning(NFFT));

    float *x = new float[NFFT * time_steps];

    float* in_f = new float[n];
    for (int i = 0; i < n; i++)
        in_f[i] = static_cast<float>(in[i]);

    for (int i = 0; i < time_steps; i++) {
        for (int j = 0; j < NFFT; j++) {
            x[i * NFFT + j] = window[j]  * in_f[i * stride + j];
        }
    }

    // Do an fft for each NFFT seg of x
    fftwf_complex* res = (fftwf_complex*) fftwf_malloc(
                            sizeof(fftwf_complex) * n_freqs * time_steps);

    fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &NFFT,
                              time_steps, x, NULL, 1, NFFT,
                              res, NULL, 1, n_freqs,
                              FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    for (int i = 0; i < time_steps; i++) {
        int s = i * n_freqs;
        for (int j = 0; j < n_freqs; j++) {
            out[s + j] = res[s + j][0] * res[s + j][0] + 
                            res[s + j][1] * res[s + j][1];
        }
    }

    // Cleanup temporaries
    fftwf_free(res);
    delete[] x;
    delete[] in_f;

    // Scale output in the same style as matplotlib
    float scale = 0;
    for (float w : window)
        scale += w * w;
    scale *= Fs;
    for (int i = 0; i < time_steps; i++) {
        int s = i * n_freqs;
        out[s] /= scale;
        for (int j = 1; j < n_freqs - 1; j++) {
            out[s + j] *= (2.0 / scale);
        }
        if (NFFT % 2 == 0) {
            out[s + n_freqs - 1] /= scale;
        } else {
            out[s + n_freqs - 1] *= (2.0 / scale);
        }
    }
}

#ifdef PYTHON
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <numpy/noprefix.h>

namespace py = boost::python;
namespace np = boost::python::numeric;

void specgram(np::array input, int NFFT, int Fs,
               int noverlap, np::array output) {
   
    py::object i_shape = input.attr("shape");
    py::object o_shape = output.attr("shape");

    if (len(i_shape) != 1 || len(o_shape) != 2)
        throw std::invalid_argument("Bad input or ouput shape");
    unsigned int n = py::extract<unsigned int>(i_shape[0]);
    
    int16_t* in_dat = static_cast<int16_t*>(PyArray_DATA(input.ptr()));
    float* out_dat = static_cast<float*>(PyArray_DATA(output.ptr()));
    cspecgram(in_dat, n, NFFT, Fs, noverlap, out_dat);
}

BOOST_PYTHON_MODULE(cspecgram) {
    np::array::set_module_and_type("numpy", "ndarray");
    py::def("specgram", &specgram);
}
#endif
