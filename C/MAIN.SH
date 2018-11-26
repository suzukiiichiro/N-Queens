#!/bin/bash

nvcc gpuNQueen.cu  -o gpuNQueen ;
./gpuNQueen ;
exit ;
