!/bin/sh
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

NUMGPUS=`./build/numgpus`
echo "Number of GPUs: ${NUMGPUS}"
if [ $NUMGPUS -lt 1 ]
then
    echo "No GPUs found, aborting test."
    exit 0
fi

#######################################
# Find nvidia-smi for temperature tests
TEMPTEST=0
NVSMI=`which rocm-smi`
if ! [ -x "$NVSMI" ]
then
    NVSMI=`find /usr/local -name 'nvidia-smi' 2> /dev/null`
    if ! [ -x "$NVSMI" ]
    then
        NVSMI=`find -L /etc -name 'nvidia-smi' 2> /dev/null`
        if ! [ -x "$NVSMI" ]
        then
            echo "WARNING: nvidia-smi not found"
        else
            TEMPTEST=1
        fi
    else
        TEMPTEST=1
    fi
else
    TEMPTEST=1
fi

if [ $TEMPTEST -eq 1 ]
then
    echo "Found nvidia-smi at ${NVSMI}"
fi
#######################################


# Run L0 diagnostics
echo ""
echo "L0 diagnostics"
echo "--------------"

echo "1/2 Computer information"
echo "CPU Info:" > l0-info.log
cat /proc/cpuinfo >> l0-info.log
echo "Memory Info:" >> l0-info.log
cat /proc/meminfo >> l0-info.log

echo "2/2 Device information"
./build/devinfo > l0-devices.log


# Run L1 tests
echo ""
echo "L1 Tests"
echo "--------"

echo "1/8 Half-duplex (unidirectional) memory copy"
./build/halfduplex > l1-halfduplex.log

echo "2/8 Full-duplex (bidirectional) memory copy"
./build/fullduplex > l1-fullduplex.log

echo "3/8 Half-duplex DMA Read"
./build/uva > l1-uvahalf.log

echo "4/8 Full-duplex DMA Read"
./build/uva --fullduplex > l1-uvafull.log

echo "5/8 Half-duplex DMA Write"
./build/uva --write > l1-uvawhalf.log

echo "6/8 Full-duplex DMA Write"
./build/uva --write --fullduplex > l1-uvawfull.log

echo "7/8 Scatter-Gather"
./build/scatter > l1-scatter.log


# Run L2 tests
echo ""
echo "L2 Tests"
echo "--------"

# Matrix multiplication


# Stencil operator
echo "4/7 Stencil (correctness)"
./build/gol --repetitions=5 --regression=true > l2-gol-correctness.log
echo "5/7 Stencil (performance)"
./build/gol --repetitions=1000 --regression=false > l2-gol-perf.log

# Test each GPU separately
echo "6/7 Stencil (single GPU correctness)"
echo "" > l2-gol-single.log
i=0
while [ $i -lt $NUMGPUS ]
do
    echo "GPU $i" >> l2-gol-single.log
    echo "===========" >> l2-gol-single.log
    ./build/gol --num_gpus=1 --repetitions=5 --regression=true --gpuoffset=$i >> l2-gol-single.log
    i=`expr $i + 1`
done



echo "Done!"
