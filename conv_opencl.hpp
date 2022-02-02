#include <CL/cl.hpp>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <complex>

#include "opencl_compile/outputFile.cpp"


struct params
{
    size_t size_id1;
    size_t size_id2;
    uint dist_0_idx;
    uint dopp_0_idx;
    uint sig_n;
    uint n_dist;
    uint n_dopp;
};

class conv_calculator
{
public:
    conv_calculator() :
        kerlan_code(ocl_src_inputFile),
        kernal_name("ambig_calculation")
    {
        cl_int ret = 0;
        //        conv = cl::Buffer(context, CL_MEM_READ_WRITE, 201*sizeof(cl_float2), nullptr, &ret);
        init_Device_and_Kernal();

    }

    ~conv_calculator(){}



    cl_int LazyAllocate(const std::vector<std::complex<float>> &_sig1,
                        const std::vector<uint> &_id1,
                        const std::vector<std::complex<float>> &_sig2,
                        const std::vector<uint> &_id2,
                        const size_t &conv_n)
    {

        cl_int ret = CL_SUCCESS;

        size_t new_size_sig1 = sizeof(cl_float2) * _sig1.size();
        sig1 = cl::Buffer(context, CL_MEM_READ_ONLY, new_size_sig1, nullptr, &ret);
        assert(ret == CL_SUCCESS);
        ret = queue.enqueueWriteBuffer(sig1,CL_TRUE,0,new_size_sig1,&_sig1.at(0));
        assert(ret == CL_SUCCESS);

        size_t new_size_sig2 = sizeof(cl_float2) * _sig2.size();
        sig2 = cl::Buffer(context, CL_MEM_READ_ONLY, new_size_sig2, nullptr, &ret);
        assert(ret == CL_SUCCESS);
        ret = queue.enqueueWriteBuffer(sig2, CL_TRUE,0, new_size_sig2, &_sig2.at(0));
        assert(ret == CL_SUCCESS);



        size_t new_size_id1 = sizeof(cl_uint) * _id1.size();
        id1 = cl::Buffer(context, CL_MEM_READ_ONLY, new_size_id1, nullptr, &ret);
        assert(ret == CL_SUCCESS);
        ret = queue.enqueueWriteBuffer(id1, CL_TRUE, 0,new_size_id1,&_id1.at(0));
        assert(ret == CL_SUCCESS);


        size_t new_size_id2 = sizeof(cl_uint) * _id2.size();
        id2 = cl::Buffer(context, CL_MEM_READ_ONLY, new_size_id2, nullptr, &ret);
        assert(ret == CL_SUCCESS);
        ret = queue.enqueueWriteBuffer(id2, CL_TRUE, 0, new_size_id2,&_id2.at(0));
        assert(ret == CL_SUCCESS);


        size_t new_size_conv = 2*sizeof(cl_float) * conv_n;
        conv = cl::Buffer(context, CL_MEM_READ_ONLY, new_size_conv, nullptr, &ret);
        assert(ret == CL_SUCCESS);
        float val = 0;
        assert(queue.enqueueFillBuffer(conv, val, 0, new_size_conv) == CL_SUCCESS);

        return CL_SUCCESS;
    }

    cl_int start_conv(const params &param, const int& min_dopp)
    {
        cl_int ret = CL_SUCCESS;
        auto kernel = cl::make_kernel
                <
                cl::Buffer,
                cl::Buffer,
                cl::Buffer,
                cl::Buffer,
                cl::Buffer,
                uint,
                int,
                uint,
                uint,
                uint
                >
                (program, kernal_name, &ret);

        assert(ret == CL_SUCCESS);

        size_t size_x = param.size_id1;
        assert(ret == CL_SUCCESS);
        size_t size_y = param.size_id2;
        assert(ret == CL_SUCCESS);
        size_t size_z = param.n_dopp;


        kernel(cl::EnqueueArgs(queue,cl::NullRange, cl::NDRange(size_x, size_y, size_z), cl::NullRange),
               sig1,
               sig2,
               id1,
               id2,
               conv,
               param.dist_0_idx,
               min_dopp,
               param.sig_n,
               param.n_dist,
               param.n_dopp);

        return queue.finish();
    }

    cl_int get_conv(std::vector<std::complex<float>> &_conv)
    {
        std::cout<<"conv size: "<<_conv.size()<<std::endl;

        return queue.enqueueReadBuffer(conv, CL_TRUE, 0,
                                _conv.size() * sizeof (cl_float)*2, _conv.data());
    }

private:
    cl_int log_error()
    {
        cl_int ret = CL_SUCCESS;
        std::string error = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &ret);// <--- use this for debbug kernal
        std::string copy_err(error);
        copy_err.erase(std::remove(copy_err.begin(), copy_err.end(), '\n'), copy_err.end());
        copy_err.erase(std::remove(copy_err.begin(), copy_err.end(), '\000'), copy_err.end());

        if(!copy_err.empty())
        {
            std::cout<<error<<std::endl;
        }
        return ret;
    }

    cl_int init_Device_and_Kernal()
    {
        cl_int ret = CL_SUCCESS;

        std::vector<cl::Platform> all_platforms;
        ret = cl::Platform::get(&all_platforms);
        assert(ret == CL_SUCCESS);

        cl::Platform default_platform=all_platforms[1];

        std::vector<cl::Device> all_devices;
        ret = default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        assert(ret == CL_SUCCESS);

        device = all_devices[0];

        std::cout<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;

        assert(ret == CL_SUCCESS);
        context = cl::Context(device);
        assert(ret == CL_SUCCESS);

        queue = cl::CommandQueue(context, device, 0, &ret);
        assert(ret == CL_SUCCESS);

        program = cl::Program(context, kerlan_code, true, &ret);
        assert(log_error() == CL_SUCCESS);
        return ret;
    }
    cl::Buffer conv;
    cl::Buffer sig1;
    cl::Buffer sig2;
    cl::Buffer id1;
    cl::Buffer id2;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    std::string kerlan_code;
    std::string kernal_name;
};
