#!/bin/bash

: '
# コンパイルオプションなしの Main.c
nlp-dev_suzuki#4.2$ g++  Main_ani1.c && ./a.out
1./a.out
 6:            4        00:00:00:00.00
 7:           40        00:00:00:00.00
 8:           92        00:00:00:00.00
 9:          352        00:00:00:00.00
10:          724        00:00:00:00.00
11:         2680        00:00:00:00.00
12:        14200        00:00:00:00.00
13:        73712        00:00:00:00.02
14:       365596        00:00:00:00.11
15:      2279184        00:00:00:00.56
16:     14772512        00:00:00:03.42
17:     95815104        00:00:00:23.04
nlp-dev_suzuki#4.2$

# コンパイルオプション付きの Main.c
nlp-dev_suzuki#4.2$ bash MAIN.SH Main_ani1.c gcc
1./a.out
 6:            4        00:00:00:00.00
 7:           40        00:00:00:00.00
 8:           92        00:00:00:00.00
 9:          352        00:00:00:00.00
10:          724        00:00:00:00.00
11:         2680        00:00:00:00.00
12:        14200        00:00:00:00.00
13:        73712        00:00:00:00.01
14:       365596        00:00:00:00.05
15:      2279184        00:00:00:00.31
16:     14772512        00:00:00:02.02
17:     95815104        00:00:00:13.74
nlp-dev_suzuki#4.2$

# コンパイルオプション付きの Main.java
nlp-dev_suzuki#4.2$ bash MAIN.SH Main.java java
 N:            Total       Unique     hh:mm:ss.SSS
 4:                0            0        00:00:001
 5:               18            0        00:00:000
 6:                4            0        00:00:000
 7:               40            0        00:00:001
 8:               92            0        00:00:001
 9:              352            0        00:00:002
10:              724            0        00:00:002
11:             2680            0        00:00:013
12:            14200            0        00:00:012
13:            73712            0        00:00:021
14:           365596            0        00:00:075
15:          2279184            0        00:00:335
16:         14772512            0        00:01:882
17:         95815104            0        00:12:790
nlp-dev_suzuki#4.2$
';

# ソースファイルの指定
JavaSourceFile="$1";
gccSourceFile="$1";
#
# 実行方法
#
# Java
# $ bash MAIN.SH sourcefile java
#
# gcc
# $ bash MAIN.SH sourcefile gcc
#
function func_Java(){
  # コンパイル
  javac -cp .:commons-lang3-3.4.jar $JavaSourceFile;  
  # 実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy  Main  ;
}
#
function func_gcc(){
  # コンパイル
  # gcc $gccSourceFile ;
  # gcc -Wall -W -O3 -g -ftrapv -std=c99 $gccSourceFile ;
  # g++ -O3 -mtune=native -march=native -mfpmath=both $gccSourceFile && ./a.out  
  g++ -O3 -mtune=native -march=native $gccSourceFile && ./a.out  

}
if [ "java" == "$2" ];then
  func_Java ;
elif [ "gcc" == "$2" ];then
  func_gcc;
else 
  echo ""
  echo "usage"
  echo "$ bash MAIN.SH sourcefile java";
  echo "$ bash MAIN.SH sourcefile gcc";
fi

