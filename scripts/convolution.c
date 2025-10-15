#include <stdio.h>

void convolve(const double* signal, int signal_len, const double* kernel, int kernel_len, double* output) {
    int n;
    int k;

    for (n = 0; n < signal_len - kernel_len + 1; n++) {
        output[n] = 0;  // Initialize the output element to 0
        for (k = 0; k < kernel_len; k++) {
            if (n + k >= 0 && n + k < signal_len) {
                if (signal[n + k] == kernel[k]) {
                    output[n] += 1;
                }
            }
        }
    }
}

