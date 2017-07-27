#!/bin/bash
#えぬクイーン システムもにた
if [ -z "$1" ];then
  echo "引数にプログラム名を指定してください";
  echo "./smon.sh 7_26NQueen";
  exit;
fi
pg="$1";
tmp="tmp.txt";
tmp2="tmp2.txt";
tmp3="tmp3.txt";
ct="count.txt";
#pi=$(ps |grep -i nqueen|grep -v grep|awk '{print $1;}');
pi=$(ps |grep -i nqueen|grep -v "grep"|grep -v "smon"|grep -v "vim"|awk '{print $1;}');
tm=$(ps |grep -i nqueen|grep -v "grep"|grep -v "smon"|grep -v "vim"|awk '{print $3;}');
cat << EOT > gdb_test
attach $pi 
set print pretty on 
set print elements 0
shell echo -n "<size>"
p si
shell echo -n "</size>"
p lTotal
p lUnique
shell echo "<threadid>"
info threads
shell echo "</threadid>"
shell echo "<printl>"
thread apply all p l[0] 
shell echo "</printl>"
detach
EOT

gdb "$pg" -x gdb_test -batch >$tmp;
si=$(cat "$tmp"|tr -d "\n"|sed -e "s|.*<size>||" -e "s|</size>.*||" -e "s|.*= ||");
th=$(cat "$tmp"|tr -d "\n"|sed -e "s|^.*<threadid>||" -e "s|</threadid>.*$||" -e "s|Thread|\nThread|g"|grep "of process"|while read l;do
#Thread 0x1203 of process 5409 0x00007fff8acab10a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib  2    
#Thread 0x1303 of process 5409 0x00007fff8acab10a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib  3    
#Thread 0x1403 of process 5409 0x0000000101c875fd in backTrack1 (y=17, left=-480288568, down=2090735, right=7807, bm=256, l=0x70000007f4f0) at 07_26NQueen.c:245  4    
  if echo "$l"|grep "__semwait_signal" >/dev/null;then
    continue;
  fi
  tid=$(echo "$l"|sed -e "s|^.*Thread ||" -e "s| of process.*$||");
  me="";
  if echo "$l"|grep "backTrack1" >/dev/null;then
    me="BT1";
  elif echo "$l"|grep "backTrack2" >/dev/null;then 
    me="BT2";
  fi
  y=$(echo "$l"|sed -e "s|^.*y=||" -e "s|,.*$||");
  echo "$tid,$me,$y";
done);
l=$(cat "$tmp"|tr -d "\n"|sed -e "s|^.*<printl>||" -e "s|Thread|\nThread|g"|grep "of process"|grep "B1"|while read l;do
  #echo "$l"
  #Thread 0x1403 of process 5409):$12 = {  bit = 0,   own = 0,   ptn = 0,   you = 0,   B1 = 11,   B2 = 9,   TB = 1048576,   EB = 0,   msk = 2097151,   SM = 0,   LM = 0,   aB = {1, 2048, 262144, 32, 8, 16384, 524288, 32768, 4, 64, 8192, 131072, 16, 1048576, 1024, 65536, 2, 4096, 512, 128, 1024, 0, 0, 0, 0, 0, 0},   C2 = {0 <repeats 27 times>},   C4 = {0 <repeats 27 times>},   C8 = {0 <repeats 11 times>, 213275839, 0 <repeats 15 times>}}
  tid=$(echo "$l"|sed -e "s|^.*Thread ||" -e "s| of process.*$||");
  b1=$(echo "$l"|sed -e "s|^.*B1 = ||" -e "s|,.*$||");
  b2=$(echo "$l"|sed -e "s|^.*B2 = ||" -e "s|,.*$||");
  c2=$(echo "$l"|sed -e "s|^.*C2 = {||" -e "s|},.*$||" -e "s|<[^>]*>||" -e "s| ||g"|sed -e "s|,|\n|g"|awk '{m+=$1} END{print m;}');
  c4=$(echo "$l"|sed -e "s|^.*C4 = {||" -e "s|},.*$||" -e "s|<[^>]*>||" -e "s| ||g"|sed -e "s|,|\n|g"|awk '{m+=$1} END{print m;}');
  c8=$(echo "$l"|sed -e "s|^.*C8 = {||" -e "s|},.*$||" -e "s|<[^>]*>||" -e "s| ||g"|sed -e "s|,|\n|g"|awk '{m+=$1} END{print m;}');
  echo "$tid,$b1,$c2,$c4,$c8";
done);
echo "###########################"
echo "SIZE:$si";
tl=$(cat "$ct"|grep "^$si"|awk -F, '{print $2;}')
sl=$(cat "$ct"|grep "^$si"|awk -F, '{print $3;}')
echo "TIME:$tm";
#合計数を取得する
t=$(echo "$l"|awk -F, '{m+=$3*2+$4*4+$5*8}END{print m;}')
tp=$(echo "scale=3; ($t / $tl) * 100" | bc);
echo "lTotal:$tp %:$t/$tl"
s=$(echo "$l"|awk -F, '{m+=$3+$4+$5}END{print m;}')
sp=$(echo "scale=3; ($s / $sl) * 100" | bc);
echo "lUnique:$sp %:$s/$sl"
#x1b03,1,0,628,421024691 BT2:15
echo "###########################"
echo "THREAD:B1:C2:C4:C8 METHOD:Y"
echo "$l"|sort -nr -k1 -t,|while read l;do
 tid=$(echo "$l"|awk -F, '{print $1;}');
 m=$(echo "$th"|grep "$tid"|awk -F, '{print $2":"$3}');
 echo "$l $m"
done
echo "###########################"
#cat $tmp;
