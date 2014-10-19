#include "mf.h"
#include <iostream>

int main(int argc, const char *argv[])
{
    if (argc < 2) {
        printf("argc < 2\n");
        return -1;
    }
    pug::MF mf; 
    mf.load_from_file(std::string(argv[1]), "::");

    // parameters
    int n_feat = 10;
    float stepsz = 0.0008;
    float reg = 0.002;
    float momentum = 0.9;
    uint32_t round_len = 100000000;
    size_t maxsec = 10000000;
    size_t maxiter = 100000;
    int itv_test = 1;
    int itv_save = 1;

    mf.train(n_feat, stepsz, reg, momentum, round_len, maxsec, maxiter, itv_test, itv_save);

    return 0;   
};


