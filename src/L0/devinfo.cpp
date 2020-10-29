// MGBench: Multi-GPU Computing Benchmark Suite
// Copyright (c) 2016, Tal Ben-Nun
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

static void HandleError(const char *file, int line, hipError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           hipGetErrorString(err), err);
    hipGetLastError();
}

// CUDA assertions
#define CUDA_CHECK(err) do { hipError_t errr = (err); if(errr != hipSuccess) { HandleError(__FILE__, __LINE__, errr); exit(1); } } while(0)

// Device capability helper macros
#define CAP(cap) NCAP(cap, cap)
#define NCAP(cap, name) ((props.cap) ? (#name " ") : "")

int main(int argc, char **argv)
{
    int ndevs = 0;
    if (hipGetDeviceCount(&ndevs) != hipSuccess)
        return 1;

    int gpuid = -1;
    if (argc > 1) {
        gpuid = atoi(argv[1]);
    }

    int version = 0;
    CUDA_CHECK(hipDriverGetVersion(&version));
    std::cout << "Driver version: " << (version / 1000) << "."
              << ((version % 100) / 10) << std::endl;

    version = 0;
    CUDA_CHECK(hipRuntimeGetVersion(&version));
    std::cout << "Runtime version: " << (version / 1000) << "."
              << ((version % 100) / 10) << std::endl;
    std::cout << std::endl;

    // Print information for each GPU
    for (int i = 0; i < ndevs; ++i)
    {
        // Skip GPUs
        if (gpuid >= 0 && i != gpuid) continue;

        CUDA_CHECK(hipSetDevice(i));
        hipDeviceProp_t props;
        CUDA_CHECK(hipGetDeviceProperties(&props, i));

        std::cout << "GPU " << (i + 1) << ": " << props.name << " ("
                  << props.pciDomainID << "/" << props.pciBusID
                  << "/" << props.pciDeviceID << ")" << std::endl

                  << "Compute capability: sm_" << props.major << props.minor << std::endl
                  << "Global memory: " << (props.totalGlobalMem/1024.0/1024.0)
                  << " MB" << std::endl
                  << "Constant memory: " << props.totalConstMem
                  << " bytes" << std::endl
                  << "Shared memory: " << props.sharedMemPerBlock
                  << " bytes" << std::endl
                  << "Registers: " << props.regsPerBlock << std::endl
                  << "Warp size: " << props.warpSize << std::endl
                  << "Multiprocessors: " << props.multiProcessorCount
                  << std::endl
                  << "Clock rate: " << (props.clockRate / 1e6)
                  << " GHz" << std::endl
                  << "Threads per MP: " << props.maxThreadsPerMultiProcessor
                  << std::endl
                  << "Threads per block: " << props.maxThreadsPerBlock
                  << std::endl
                  << "Max block size: " << props.maxThreadsDim[0] << "x"
                  << props.maxThreadsDim[1] << "x" << props.maxThreadsDim[2]
                  << std::endl
                  << "Max grid size: " << props.maxGridSize[0] << "x"
                  << props.maxGridSize[1] << "x" << props.maxGridSize[2]
                  << std::endl
                  << "Pitch: " << props.memPitch << " bytes" << std::endl;

        std::cout << "Caps: " << NCAP(ECCEnabled, ecc)
                  << NCAP(kernelExecTimeoutEnabled, timeout)
                  << CAP(integrated) << NCAP(canMapHostMemory, hostdma)
                  << CAP(tccDriver) << std::endl;
        std::cout << std::endl;
    }

    std::cout << "DMA access: " << std::endl;
    int tmp = 0;

    // Print top row
    printf("   | ");
    for (int i = 0; i < ndevs; ++i)
        printf("%2d ", i + 1);
    printf("\n---+");
    for (int i = 0; i < ndevs; ++i)
        printf("---");
    printf("\n");
    
    for (int i = 0; i < ndevs; ++i)
    {
        printf("%2d | ", i + 1);
        for (int j = 0; j < ndevs; ++j)
        {
            if (i == j)
            {
                printf(" x ");
                continue;
            }

            hipDeviceCanAccessPeer(&tmp, i, j);
            printf("%2d ", tmp ? 1 : 0);
        }
        printf("\n");
    }
    
    return 0;
}
