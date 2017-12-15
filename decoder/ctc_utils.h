#pragma once

#include <vector>

int setup_labels(const std::vector<int>& labels, 
                 const int blank,
                 std::vector<int>& labels_w_blanks,
                 std::vector<int>& s_inc,
                 std::vector<int>& e_inc) {

    const int L = labels.size();
    int repeats = 0;
    s_inc.push_back(1);
    for (int i = 1; i < L; ++i) {
        if (labels[i-1] == labels[i]) {
            s_inc.push_back(1);
            s_inc.push_back(1);
            e_inc.push_back(1);
            e_inc.push_back(1);
            ++repeats;
        }
        else {
            s_inc.push_back(2);
            e_inc.push_back(2);
        }
    }
    e_inc.push_back(1);

    for (int i = 0; i < L; ++i) {
        labels_w_blanks.push_back(blank);
        labels_w_blanks.push_back(labels[i]);
    }
    labels_w_blanks.push_back(blank);

    return repeats;
}
