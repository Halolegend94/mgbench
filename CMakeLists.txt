# MGBench: Multi-GPU Computing Benchmark Suite
# Copyright (c) 2016, Tal Ben-Nun
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of the copyright holders nor the names of its 
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 2.8)
project (mgbench)

############## DEPENDENCIES ##############

add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

include_directories(${PROJECT_SOURCE_DIR}/deps/maps/include)

################# FLAGS ##################

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(HIP_FLAGS -g;-lineinfo;-Xcompiler;-ggdb;-std=c++11)
else()
  set(HIP_FLAGS -O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11)
endif()

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (WIN32 AND NOT CYGWIN)
  set(EXTRA_LIBS "")
else()
  set(EXTRA_LIBS "pthread")
endif()

#########################################

# Level 0 (diagnostics)

add_executable(numgpus src/L0/numgpus.cpp)
add_executable(devinfo src/L0/devinfo.cpp)

# Level 1 (microbenchmarks)

add_executable(halfduplex src/L1/halfduplex.cpp)
target_link_libraries(halfduplex gflags-static ${EXTRA_LIBS})

add_executable(fullduplex src/L1/fullduplex.cpp)
target_link_libraries(fullduplex gflags-static ${EXTRA_LIBS})

add_executable(uva src/L1/uva.cpp)
target_link_libraries(uva gflags-static ${EXTRA_LIBS})

add_executable(scatter src/L1/scatter.cpp)
target_link_libraries(scatter gflags-static ${EXTRA_LIBS})

# Level 2 (micro-applications)

add_executable(gol src/L2/gol/golsample.cpp src/L2/gol/main.cpp)
target_link_libraries(gol gflags-static ${EXTRA_LIBS})

add_executable(sgemm src/L2/sgemm/sgemm.cpp src/L2/sgemm/main.cpp )
target_link_libraries(sgemm gflags-static hipblas ${EXTRA_LIBS})
target_link_libraries(sgemm /opt/rocm/rocblas/lib/librocblas.so)
