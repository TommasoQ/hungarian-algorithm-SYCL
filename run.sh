#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is compiling SYCL_Hungarian_Algorithm.cpp"
icpx -o3 -fsycl SYCL_Hungarian_Algorithm.cpp -o SYCL_Hungarian_Algorithm
if [ $? -eq 0 ]; then ./SYCL_Hungarian_Algorithm; fi

