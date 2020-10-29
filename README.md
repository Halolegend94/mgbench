MGBench: Multi-GPU Computing Benchmark Suite
==========================================

This fork brings MGBench on AMD hardware.

This set of applications test the performance, bus speed, power efficiency and correctness of a multi-GPU node.

It is comprised of Level-0 tests (diagnostic utilities), Level-1 tests (microbenchmarks), and Level-2 tests (micro-applications).

Requirements
------------

rocm 3.9 or higher.

Instructions
------------

To build, execute the included `build.sh` script. Alternatively, create a 'build' directory and run `cmake ..` within, followed by `make`.

To run the tests, execute the included `run.sh` script. The results will then be placed in the working directory.

Tests
-----

A full list of the tests, their purpose, and command-line arguments can be found [here](https://github.com/tbennun/mgbench/blob/master/TESTS.md).

License
-------

MGBench is published under the New BSD license, see [LICENSE](https://github.com/tbennun/mgbench/blob/master/LICENSE).


Included Software and Licenses
------------------------------

The following dependencies are included within the repository for ease of compilation:

* [gflags](https://github.com/gflags/gflags): [New BSD License](https://github.com/tbennun/mgbench/blob/master/deps/gflags/COPYING.txt). Copyright (c) 2006, Google Inc. All rights reserved.

* [MAPS](https://github.com/maps-gpu/MAPS): [New BSD License](https://github.com/tbennun/mgbench/blob/master/deps/maps/LICENSE). Copyright (c) 2015, A. Barak. All rights reserved.


