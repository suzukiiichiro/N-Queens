#!/usr/bin/bash
# 日本語


: '１０進数を２進数に変換';
# dec2bin 2 
# 00000010
dec2binlogic (){
  local num bin=""
  for((num=$1;num;num/=2));do
    bin=$((num%2))$bin
  done
  ((${#3})) &&
      printf -v ${3} "%0*d" ${2:-1} $bin ||
      printf "%0*d\n" ${2:-1} $bin
}
function dec2bin()
{
  local -i in="$1";
  [ $in -lt 0 ] &&
     dec2binlogic 256+$in 8 || dec2binlogic $in 8
}
dec2bin 2;
exit;

declare -i size=5;
declare -i  n=$size;
declare -a board;
declare -a x;
declare -a y;
declare -a d1;
declare -a d2;
declare -a r90;
declare -a r180;
declare -a r270;

# 10 100 1 11 0
#echo "obase=2; $1" | bc
echo "bc <<< "obase=2;00100""
board=(2 4 1 3 0);
echo "原型      ${board[@]}";

for ((index=0;index<size;index++));do
  x[$((n-1-index))]=$(( board[index] ));  
  : '
   2 4 1 3 0        0 3 1 4 2  
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      |Q| | | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O|      | | | |Q| |
  +-+-+-+-+-+  x   +-+-+-+-+-+
  | |Q| | | |------| |Q| | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | |Q| |      | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | | |Q| | |
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  : '
   2 4 1 3 0        2 0 3 1 4
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      | | |Q| | |
  +-+-+-+-+-+   |  +-+-+-+-+-+
  | | | | |O|   |  |Q| | | | |
  +-+-+-+-+-+   |  +-+-+-+-+-+
  | |Q| | | |  y|  | | | |Q| |
  +-+-+-+-+-+   |  +-+-+-+-+-+
  | | | |Q| |   |  | |Q| | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  y[$((index))]=$(( n-1-board[index] ));  
  : '
   2 4 1 3 0        4 2 0 3 1
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O| \    | | |Q| | |
  +-+-+-+-+-+  \   +-+-+-+-+-+
  | |Q| | | |   \  |Q| | | | |
  +-+-+-+-+-+    \ +-+-+-+-+-+
  | | | |Q| |      | | | |Q| |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | |Q| | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  d1[$((board[index]))]=$(( index ));  
  : '
   2 4 1 3 0        3 1 4 2 0 
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      | | | |Q| |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O|     /| |Q| | | |
  +-+-+-+-+-+  d2/ +-+-+-+-+-+
  | |Q| | | |   /  | | | | |Q|
  +-+-+-+-+-+  /   +-+-+-+-+-+
  | | | |Q| | /    | | |Q| | |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      |Q| | | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  d2[$((n-1-board[index]))]=$(( n-1-index ));  
  : '
   2 4 1 3 0        0 2 4 1 3
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      |Q| | | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O|      | | |Q| | |
  +-+-+-+-+-+ r90  +-+-+-+-+-+
  | |Q| | | |  ->  | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | |Q| |      | |Q| | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | | | |Q| |
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  r90[$((board[index]))]=$(( n-1-index ));  
  : '
   2 4 1 3 0        4 1 3 0 2
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O|      | |Q| | | |
  +-+-+-+-+-+ r180 +-+-+-+-+-+
  | |Q| | | |  ->  | | | |Q| |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | |Q| |      |Q| | | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | | |Q| | |
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  r180[$((n-1-index))]=$(( n-1-board[index] ));  
  : '
   2 4 1 3 0        1 3 0 2 4 
  +-+-+-+-+-+      +-+-+-+-+-+
  | | |Q| | |      | |Q| | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | | |O|      | | | |Q| |
  +-+-+-+-+-+ r270 +-+-+-+-+-+
  | |Q| | | |  ->  |Q| | | | |
  +-+-+-+-+-+      +-+-+-+-+-+
  | | | |Q| |      | | |Q| | |
  +-+-+-+-+-+      +-+-+-+-+-+
  |Q| | | | |      | | | | |Q|
  +-+-+-+-+-+      +-+-+-+-+-+
  ';
  r270[$((n-1-board[index]))]=$(( index ));  
done

echo "X反転     ${x[@]}";
echo "Y反転     ${y[@]}";
echo "d1斜軸    ${d1[@]}";
echo "d2斜軸    ${d2[@]}";
echo "r90回転   ${r90[@]}";
echo "r180回転  ${r180[@]}";
echo "r270回転  ${r270[@]}";
