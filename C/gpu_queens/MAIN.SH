#!/bin/bash

gcc -W -Wall -O3 gpu_queens.c -framework OpenCL  ;
./a.out ;
#gcc gpu_queens.c -framework OpenCL  ;

