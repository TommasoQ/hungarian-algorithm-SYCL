#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is compiling SYCL_Hungarian_Algorithm.cpp"
icpx -o3 -fsycl ./src/SYCL_Hungarian_Algorithm.cpp -o ./src/SYCL_Hungarian_Algorithm
if [ $? -eq 0 ]; then ./src/SYCL_Hungarian_Algorithm; fi

