#include <iostream>
#include <utility>
#include <memory>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <fstream>

#include "conv.hpp"
#include "conv_opencl.hpp"

using namespace std;



template<typename T_float>
std::pair<std::vector<T_float>, std::vector<uint>> signal_prepare(const std::vector<T_float> &sig)
{
    std::vector<T_float> sig_out;
    std::vector<uint> id;
    sig_out.reserve(sig.size()/2);
    id.reserve(sig.size()/2);

    for(size_t i = 0; i < sig.size(); i++)
    {
        if(std::abs(sig[i]) != 0)
        {
            sig_out.push_back(sig[i]);
            id.push_back(i);
        }
    }
    sig_out.shrink_to_fit();
    id.shrink_to_fit();
    return pair<std::vector<T_float>, std::vector<uint>>(std::move(sig_out),
                                                     std::move(id));
}



std::vector<std::complex<float>> sig_gen()
{
    size_t n = 1e3;
    std::vector<std::complex<float>> sig(n);

    int cnt = 0;
    for(size_t i = 0; i < sig.size();i++)
    {
        sig[i] = (cnt / 10) % 10;
        cnt++;
    }
    return sig;
}

void check_signals()
{
        auto sig1 = sig_gen();
        auto sig2 = sig1;
        std::vector<std::complex<float>> conv1(201);
        std::vector<std::complex<float>> conv2(201);

        conv_classic(sig1, sig2, conv1);

        auto pair1 = signal_prepare(sig1);
        auto pair2 = signal_prepare(sig2);

        conv_pulse(pair1.first, pair1.second,
                   pair2.first, pair2.second,
                   conv2);

        assert(conv1 == conv2);
}

void save_bin_conv(std::vector<std::complex<float>> &conv)
{
    std::vector<float> convf;
    convf.resize(conv.size());
    for(size_t i = 0; i < conv.size(); i++)
    {
        convf.at(i) = std::abs(conv.at(i));
//        std::cout<<convf.at(i)<<std::endl;
    }

    std::ofstream out;
    out.open("conv.bin", std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(convf.data()), sizeof(float) * convf.size());
    out.close();
}


int main()
{
    auto sig1 = sig_gen();
    auto sig2 = sig1;


    params param;

    param.sig_n = sig1.size();
    param.n_dist = 2*sig1.size() + 1;
    param.n_dopp = 101;
    param.dist_0_idx = sig1.size();
    param.dopp_0_idx= (param.n_dopp - 1)/2;

    std::vector<std::complex<float>> conv(param.n_dist * param.n_dopp);
    auto pair1 = signal_prepare(sig1);
    auto pair2 = signal_prepare(sig2);

    param.size_id1 = pair1.second.size();
    param.size_id2 = pair2.second.size();
    Timer t;
    conv_calculator calc;

    calc.LazyAllocate(pair1.first, pair1.second,
                      pair2.first, pair2.second, param.n_dist * param.n_dopp);

    calc.start_conv(param, -50);

    int ret = calc.get_conv(conv);
    assert(ret == 0);
    std::cout<<"Time gpu: "<<t.elapsed()<<std::endl;

    t.reset();
    std::vector<std::complex<float>> conv2(2*sig1.size()+1);
    conv_pulse(pair1.first, pair1.second,
               pair2.first, pair2.second, conv2);
    std::cout<<"time cpu: "<<t.elapsed()<<std::endl;
//    save_bin_conv(conv);
    return 0;
}
