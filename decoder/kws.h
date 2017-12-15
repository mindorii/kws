#pragma once

/* Score keyword against probs. */
float cscore_kws(const float* probs,
                 const int T, const int alphabet_size, 
                 const int blank,
                 const std::vector<int>& labels);
