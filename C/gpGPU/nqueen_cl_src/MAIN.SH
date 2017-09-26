#!/bin/bash

g++ -framework OpenCL *.cpp
echo "###";
./a.out -cpu 15
echo "###";
./a.out -clcpu 15
echo "###";
