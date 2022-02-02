cmake_minimum_required(VERSION 3.5)

project(pulse_conv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pulse_conv main.cpp conv.hpp conv_opencl.hpp)

target_compile_features(pulse_conv PRIVATE cxx_auto_type)

find_package(OpenCL REQUIRED)
target_link_libraries(pulse_conv OpenCL::OpenCL)
