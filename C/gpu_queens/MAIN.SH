#!/bin/bash

gcc -W -Wall -O3 gpu_queens.c -framework OpenCL  ;
#gcc gpu_queens.c -framework OpenCL  ;

