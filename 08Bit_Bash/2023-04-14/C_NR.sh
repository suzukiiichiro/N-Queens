#!/usr/bin/bash

declare -i size=5;
declare -a board[$size];
declare -a left[$size];
declare -a down[$size];
declare -a right[$size];
declare -a bitmap[$size];
declare -a bit[$size];
#
#
declare -i n=0;
function output(){ 
  ((n++));
  echo "pattern $n\n"; 
  for ((i=0;i<size;i++));do
    for ((j=0;j<size;j++));do
      echo -n $(( bit[i]&1<<j ? "Q" : "*" )); 
    done
    echo "";
  done
}
#
function srch(){
  for ((i=0;i<size;i++));do
    left[$size]=down[$size]=right[$size]=bitmap[$size]=bit[$size]=0;
  done
  bitmap=
    
}
mask=$(( (1<<size)-1 ));
srch $mask;
exit;
