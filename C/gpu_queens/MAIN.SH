#!/bin/bash

gcc  -W -Wall -O3 gpu_queens.c -framework OpenCL -o a.out  ;
#clang -W -Wall -O3 gpu_queens.c -framework OpenCL -o a.out  ;
#./a.out ;
#gcc gpu_queens.c -framework OpenCL -o b.out  ;
#./b.out ;

