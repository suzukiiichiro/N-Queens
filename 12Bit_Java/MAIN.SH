#!/bin/bash

: '
# コンパイルオプションなしの Main.c
#  gcc Main.c;
bash-3.2$ bash MAIN.SH gcc
 6:            4        00:00:00:00.00
 7:           40        00:00:00:00.00
 8:           92        00:00:00:00.00
 9:          352        00:00:00:00.00
10:          724        00:00:00:00.00
11:         2680        00:00:00:00.00
12:        14200        00:00:00:00.00
13:        73712        00:00:00:00.02
14:       365596        00:00:00:00.08
15:      2279184        00:00:00:00.41
16:     14772512        00:00:00:02.48
17:     95815104        00:00:00:16.66
bash-3.2$

# コンパイルオプション付きの Main.c
# gcc -Wall -W -O3 -g -ftrapv -std=c99 Main.c ;
3 warnings generated.
 6:            4        00:00:00:00.00
 7:           40        00:00:00:00.00
 8:           92        00:00:00:00.00
 9:          352        00:00:00:00.00
10:          724        00:00:00:00.00
11:         2680        00:00:00:00.00
12:        14200        00:00:00:00.00
13:        73712        00:00:00:00.00
14:       365596        00:00:00:00.03
15:      2279184        00:00:00:00.21
16:     14772512        00:00:00:01.36
17:     95815104        00:00:00:09.36
bash-3.2$

# コンパイルオプション付きの Main_khash.c
# gcc -Wall -W -O3 -g -ftrapv -std=c99 Main_khash.c ;
5 warnings generated.
 6:            4        00:00:00:00.00
 7:           40        00:00:00:00.00
 8:           92        00:00:00:00.00
 9:          352        00:00:00:00.00
10:          724        00:00:00:00.00
11:         2680        00:00:00:00.00
12:        14200        00:00:00:00.00
13:        73712        00:00:00:00.00
14:       365596        00:00:00:00.04
15:      2279184        00:00:00:00.23
16:     14772512        00:00:00:01.45
17:     95815104        00:00:00:10.02
bash-3.2$

# コンパイルオプション付きの Main.java
# bash-3.2$ bash MAIN.SH java
bash-3.2$ bash MAIN.SH java
 N:            Total       Unique     hh:mm:ss.SSS
 4:                0            0        00:00:001
 5:               18            0        00:00:000
 6:                4            0        00:00:000
 7:               40            0        00:00:000
 8:               92            0        00:00:001
 9:              352            0        00:00:002
10:              724            0        00:00:001
11:             2680            0        00:00:003
12:            14200            0        00:00:005
13:            73712            0        00:00:014
14:           365596            0        00:00:056
15:          2279184            0        00:00:228
16:         14772512            0        00:01:412
17:         95815104            0        00:09:768
bash-3.2$
';

# ソースファイルの指定
JavaSourceFile="Main.java";
# gccSourceFile="Main.c";
gccSourceFile="Main_khash.c";
#
# 実行方法
#
# Java
# $ bash MAIN.SH java
#
# gcc
# $ bash MAIN.SH gcc
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
  gcc -Wall -W -O3 -g -ftrapv -std=c99 $gccSourceFile ;
  ./a.out  

}
if [ "java" == "$1" ];then
  func_Java ;
elif [ "gcc" == "$1" ];then
  func_gcc;
else 
  echo ""
  echo "usage"
  echo "$ bash MAIN.SH java";
  echo "$ bash MAIN.SH gcc";
fi

