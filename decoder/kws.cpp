#include <algorithm>
#include <cmath>
#include <vector>
#include "kws.h"
#include "ctc_utils.h"
#include <iostream>

namespace {

static const float neginf = -std::numeric_limits<float>::infinity();

inline float log_add(float a, float b) {
    if (a == neginf) return b;
    if (b == neginf) return a;
    if (a > b)
        return log1p(exp(b-a)) + a;
    else
        return log1p(exp(a-b)) + b;
}

}

/* Computes forward probabilities for a given keyword.
 * Scores keyword as *keyword* allowing for arbitrary
 * pre and post characters.
 * *NB* undefined if keyword is empty string.
 */
float cscore_kws(const float* probs,
                 const int T, const int alphabet_size,
                 const int blank,
                 const std::vector<int>& labels) {

    std::vector<int> labels_w_blanks;
    std::vector<int> e_inc;
    std::vector<int> s_inc;
    int repeats = setup_labels(labels, blank, labels_w_blanks,
                               s_inc, e_inc);

    const int S = labels_w_blanks.size();
    float* prev_alphas = new float[S];
    float* next_alphas = new float[S];

    std::fill(prev_alphas, prev_alphas + S, neginf);

    int start = (((S /2) + repeats - T) < 0) ? 0 : 1,
            end = S > 1 ? 2 : 1;

    for (int i = start; i < end; ++i) {
        if (i == 0) {
            prev_alphas[i] = std::log(1 - probs[labels_w_blanks[1]]);
        } else {
            int l = labels_w_blanks[i];
            prev_alphas[i] = std::log(probs[l]);
        }
    }

    for(int t = 1; t < T; ++t) {
        std::fill(next_alphas, next_alphas + S, neginf);

        int remain = (S / 2) + repeats - (T - t);
        if(remain >= 0)
            start += s_inc[remain];
        if(t <= (S / 2) + repeats)
            end += e_inc[t - 1];
        int startloop = start;
        int idx = t * alphabet_size;

        if (start == 0) {
            float star_score = std::log(1 - probs[idx + labels_w_blanks[1]]);
            next_alphas[0] = prev_alphas[0] + star_score;
            startloop += 1;
        }

        for(int i = startloop; i < end; ++i) {
            int l = labels_w_blanks[i];
            float prev_sum = log_add(prev_alphas[i], prev_alphas[i-1]);

            // Skip two if not on blank and not on repeat.
            if (l != blank && i != 1 &&
                    l != labels_w_blanks[i-2])
                prev_sum = log_add(prev_sum, prev_alphas[i-2]);

            next_alphas[i] = prev_sum;
            if (i == labels_w_blanks.size() - 1) {
                float nl_score = probs[idx + labels_w_blanks[i-1]];
                next_alphas[i] += std::log(1 - nl_score);
            } else {
                next_alphas[i] += std::log(probs[l + idx]);
            }
        }
        std::swap(prev_alphas, next_alphas);
    }

    float loglike = neginf;
    for(int i = start; i < end; ++i) {
        loglike = log_add(loglike, prev_alphas[i]);
    }

    // Cleanup
    delete[] prev_alphas;
    delete[] next_alphas;

    return -loglike;
}

// Python Bindings
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/noprefix.h>

namespace py = boost::python;
namespace np = boost::python::numpy;

float score_kws(np::ndarray &probs, py::list labels,
                const int blank) {
    // *NB* logits must be type float and row-major.
    py::object shape = probs.attr("shape");

    unsigned int time = py::extract<unsigned int>(shape[0]);
    unsigned int num_classes = py::extract<unsigned int>(shape[1]);

    float* data = reinterpret_cast<float*>(probs.get_data());

    std::vector<int> label_vec;
    for (int i = 0; i < len(labels); i++) 
        label_vec.push_back(py::extract<int>(labels[i]));

    float result = cscore_kws(data, time, num_classes,
                              blank, label_vec);

    return result;
}

BOOST_PYTHON_MODULE(kws) {
    Py_Initialize();
    np::initialize();
    py::def("score_kws", &score_kws);
}
