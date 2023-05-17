#!/usr/bin/bash

: '
 ## bash 07Bash_carryChain.sh

 ## bash版
 <> 07Bash_carryChain.sh
 bash-3.2$ bash 07Bash_carryChain.sh
 N:        Total       Unique        hh:mm:ss
 4:            2            1         0:00:00
 5:           10            2         0:00:00
 6:            4            1         0:00:00
 7:           40            6         0:00:01
 8:           92           12         0:00:02
 9:          352           46         0:00:12
10:          724           92         0:00:44
11:         2680          341         0:02:39
12:        14200         1788         0:08:35
13:        73712         9237         0:27:05
14:       365596        45771         1:30:40
15:      2279184       285095         5:59:03

 <> symmetry.sh 対象解除
 N:        Total       Unique        hh:mm:ss
 4:            2            1         0:00:00
 5:           10            2         0:00:00
 6:            4            1         0:00:00
 7:           44            7         0:00:00
 8:           92           12         0:00:00
 9:          300           39         0:00:00
10:          412           52         0:00:01
11:          724           91         0:00:03
12:         2288          287         0:00:11
13:         8612         1078         0:00:49
14:        35376         4425         0:04:24
15:       204560        25582         0:23:51

 ## python版
 $ python py13_4_multiprocess_nqueen.py
１３＿４マルチプロセス版
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            1         0:00:00.124
 5:           10            2         0:00:00.110
 6:            4            1         0:00:00.116
 7:           40            6         0:00:00.115
 8:           92           12         0:00:00.119
 9:          352           46         0:00:00.118
10:          724           92         0:00:00.121
11:         2680          341         0:00:00.122
12:        14200         1787         0:00:00.228
13:        73712         9233         0:00:00.641
14:       365596        45752         0:00:03.227
15:      2279184       285053         0:00:19.973

ちなみにpythonシングルプロセス版
15:      2279184       285053         0:00:54.645


 ## Lua版
 $ luajit Lua12_N-Queen.lua
 N:            Total       Unique    hh:mm:ss
 2:                0            0    00:00:00
 3:                0            0    00:00:00
 4:                2            1    00:00:00
 5:               10            2    00:00:00
 6:                4            1    00:00:00
 7:               40            6    00:00:00
 8:               92           12    00:00:00
 9:              352           46    00:00:00
10:              724           92    00:00:00
11:             2680          341    00:00:00
12:            14200         1787    00:00:00
13:            73712         9233    00:00:00
14:           365596        45752    00:00:00
15:          2279184       285053    00:00:03
16:         14772512      1846955    00:00:20
17:         95815104     11977939    00:02:13

 ## OpenCL版
$ gcc -Wall -W -O3 -std=c99 -pthread -lpthread -lm -o 07_52NQueen 07_52gpu_queens.c -framework OpenCL
52. OpenCL (07_38 *N*si*si アルゴリムは全部のせ) 
 N:    Total          Unique      dd:hh:mm:ss.ms
 4:        2                   1  00:00:00:00.43
 5:       10                   2  00:00:00:00.35
 6:        4                   1  00:00:00:00.35
 7:       40                   6  00:00:00:00.35
 8:       92                  12  00:00:00:00.35
 9:      352                  46  00:00:00:00.35
10:      724                  92  00:00:00:00.35
11:     2680                 341  00:00:00:00.35
12:    14200                1787  00:00:00:00.35
13:    73712                9233  00:00:00:00.36
14:   365596               45752  00:00:00:00.37
15:  2279184              285053  00:00:00:01.58

 ## Java版
$ javac -cp .:commons-lang3-3.4.jar Java13c_NQueen.java && java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java13c_NQueen  ;
１３．Java 再帰 並列処理 
 N:        Total          Unique  hh:mm:ss.SSS
 4:            2               1  00:00:00.001
 5:           10               2  00:00:00.001
 6:            4               1  00:00:00.000
 7:           40               6  00:00:00.001
 8:           92              12  00:00:00.001
 9:          352              46  00:00:00.001
10:          724              92  00:00:00.001
11:         2680             341  00:00:00.003
12:        14200            1787  00:00:00.002
13:        73712            9233  00:00:00.005
14:       365596           45752  00:00:00.021
15:      2279184          285053  00:00:00.102
16:     14772512         1846955  00:00:00.631
17:     95815104        11977939  00:00:04.253

ちなみにシングルスレッド
15:      2279184          285053  00:00:00.324
16:     14772512         1846955  00:00:02.089
17:     95815104        11977939  00:00:14.524


 ## GCC版
$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13.c && ./a.out [-c|-r]
１３．CPU 非再帰 並列処理 pthread
 N:        Total          Unique  dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.00
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.00
 9:          352              46  00:00:00:00.00
10:          724              92  00:00:00:00.00
11:         2680             341  00:00:00:00.00
12:        14200            1787  00:00:00:00.00
13:        73712            9233  00:00:00:00.00
14:       365596           45752  00:00:00:00.01
15:      2279184          285053  00:00:00:00.10
16:     14772512         1846955  00:00:00:00.65
17:     95815104        11977939  00:00:00:04.33

ちなみにシングルスレッド
15:      2279184          285053            0.34
16:     14772512         1846955            2.24
17:     95815104        11977939           15.72

 ## GPU/CUDA版
$ nvcc -O3 CUDA13_N-Queen.cu  && ./a.out -g
１２．GPU 非再帰 並列処理 CUDA
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.37
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.01
 9:          352              46  00:00:00:00.01
10:          724              92  00:00:00:00.01
11:         2680             341  00:00:00:00.01
12:        14200            1787  00:00:00:00.02
13:        73712            9233  00:00:00:00.03
14:       365596           45752  00:00:00:00.03
15:      2279184          285053  00:00:00:00.04
16:     14772512         1846955  00:00:00:00.08
17:     95815104        11977939  00:00:00:00.35

ちなみにシングルスレッド
15:      2279184          285053            0.34
16:     14772512         1846955            2.24
17:     95815104        11977939           15.72

 ## nq27版
$ gcc -Wall -W -O3 nq27_N-Queen.c && ./a.out -r
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1788            0.02
13:        73712            9237            0.06
14:       365596           45771            0.25
15:      2279184          285095            1.14
16:     14772512         1847425            6.69
17:     95815104        11979381           43.82
';
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNTER[3];    # カウンター配列
declare -i COUNT2=0;  # 配列用
declare -i COUNT4=1;  # 配列用
declare -i COUNT8=2;  # 配列用
declare -a B; 
: 'B=(row     0:
      left    1:
      down    2:
      right   3:
      X[@]    4: 
      )';
declare -i DISPLAY=0;
#
#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecordCarryChain()
{
  ((TOTAL++));
  size="$1";
  flag="$2"; # bitmap版は1 それ以外は 0
  echo "$TOTAL";
  sEcho=" ";  
  : 'ビットマップ版
     ビットマップ版からは、左から数えます
     上下反転左右対称なので、これまでの上から数える手法と
     rowを下にたどって左から数える方法と解の数に変わりはありません。
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | | 0
    +-+-+-+-+-+
    | | |O| | | 2
    +-+-+-+-+-+
    | | | | |O| 4
    +-+-+-+-+-+
    | |O| | | | 1
    +-+-+-+-+-+
    | | | |O| | 3
    +-+-+-+-+-+
  ';
  if ((flag));then
    local -i i=0;
    local -i j=0;
    for ((i=0;i<size;i++));do
      for ((j=0;j<size;j++));do
       if (( board[i]&1<<j ));then
          sEcho="${sEcho}$((j)) ";
       fi 
      done
    done
  else 
  : 'ビットマップ版以外
     (ブルートフォース、バックトラック、配置フラグ)
     上から数えます
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | |
    +-+-+-+-+-+
    | | | |O| |
    +-+-+-+-+-+
    | |O| | | |
    +-+-+-+-+-+
    | | | | |O|
    +-+-+-+-+-+
    | | |O| | |
    +-+-+-+-+-+
     ';
    local -i i=0;
    for((i=0;i<size;i++)){
      sEcho="${sEcho}${board[i]} ";
    }
  fi
  echo "$sEcho";
  echo -n "+";
  local -i i=0;
  for((i=0;i<size;i++)){
    echo -n "-";
    if((i<(size-1)));then
      echo -n "+";
    fi
  }
  echo "+";
  local -i i=0;
  local -i j=0;
  for((i=0;i<size;i++)){
    echo -n "|";
    for((j=0;j<size;j++)){
      if ((flag));then
        if(( board[i]!=-1));then
          if (( board[i]&1<<j ));then
            echo -n "Q";
          else
            echo -n " ";
          fi
        else
          echo -n " ";
        fi
      else
        if((i==board[j]));then
          echo -n "Q";
        else
          echo -n " ";
        fi
      fi
      if((j<(size-1)));then
        echo -n "|";
      fi
    }
  echo "|";
  if((i<(size-1)));then
    echo -n "+";
    local -i j=0;
    for((j=0;j<size;j++)){
      echo -n "-";
      if((j<(size-1)));then
        echo -n "+";
      fi
    }
  echo "+";
  fi
  }
  echo -n "+";
  local -i i=0;
  for((i=0;i<size;i++)){
    echo -n "-";
    if((i<(size-1)));then
      echo -n "+";
    fi
  }  
  echo "+";
  echo "";
}
#
: 'ボード外側２列を除く内側のクイーン配置処理';
function solve()
{
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  # 配置完了の確認 
  # bh=-1 1111111111 すべての列にクイーンを置けると
  # -1になる
  (( down+1 ))|| return 1;
  # 新たなQを配置 colは置き換える。
  # row 右端にクイーンがすでに置かれていたら
  # クイーンを置かずに１行下に移動する。
  # rowを右端から１ビットずつ削っていく。
  # ここではrowはすでにクイーンが置かれているか
  # どうかだけで使う
  while(( row&1 ));do
    # (( row>>=1 ))     # 右に１ビットシフト
    # (( left<<=1 ));   # left 左に１ビットシフト
    # (( right>>=1 ));  # right 右に１ビットシフト
    (( row>>=1,left<<=1,right>>=1 ));
  done
  (( row>>=1 ));      # １行下に移動する
  local -i bit;
  local -i bitmap;
  local -i total=0;
  for (( bitmap=~(left|down|right);bitmap!=0;));do
    (( bit=-bitmap&bitmap ));
    solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))"  ; 
    (( total+=$? ));
    ((bitmap^=bit));
  done
  return $total;
}
#
: 'solve()を呼び出して再帰を開始する';
function process()
{
  local -i size="$1";
  local -i sym="$2";
  solve "$(( B[0]>>2 ))" \
        "$(( B[1]>>4 ))" \
        "$(( (((B[2]>>2 | ~0<<size-4)+1)<<size-5)-1 ))" \
        "$(( B[3]>>4<<size-5 ))";
  (( COUNTER[$sym]+=$? ));
}
#
: 'クイーンの効きをチェック';
function placement()
{
  local -i size="$1";
  local -i dimx="$2";     # dimxは行 dimyは列
  local -i dimy="$3";
  local -a t_x=(${B[4]}); # 同じ場所の配置を許す
  (( t_x[dimx]==dimy ))&& return 1;
  : '
  #
  #
  # 【枝刈り】Qが角にある場合の枝刈り
  #  ２．２列めにクイーンは置かない
  #  （１はcarryChainSymmetry()内にあります）
  #
  #  Qが角にある場合は、
  #  2行目のクイーンの位置 t_x[1]が BOUND1
  #  BOUND1行目までは2列目にクイーンを置けない
  # 
  #    +-+-+-+-+-+  
  #    | | | |X|Q| 
  #    +-+-+-+-+-+  
  #    | |Q| |X| | 
  #    +-+-+-+-+-+  
  #    | | | |X| |       
  #    +-+-+-+-+-+             
  #    | | | |Q| | 
  #    +-+-+-+-+-+ 
  #    | | | | | |      
  #    +-+-+-+-+-+  
  #';
  if (( t_x[0]==0 ));then
    if (( t_x[1]!=-1));then
      # bitmap=$(( bitmap|2 ));
      # bitmap=$(( bitmap^2 ));
      # 上と下は同じ趣旨
      (((t_x[1]>=dimx) && (dimy==1)))&&{ return 0; }
    fi
  else
  : '
  #
  # 【枝刈り】Qが角にない場合
  #
  #  +-+-+-+-+-+  
  #  |X|X|Q|X|X| 
  #  +-+-+-+-+-+  
  #  |X| | | |X| 
  #  +-+-+-+-+-+  
  #  | | | | | |
  #  +-+-+-+-+-+
  #  |X| | | |X|
  #  +-+-+-+-+-+
  #  |X|X| |X|X|
  #  +-+-+-+-+-+
  #
  #   １．上部サイド枝刈り
  #  if ((row<BOUND1));then        
  #    bitmap=$(( bitmap|SIDEMASK ));
  #    bitmap=$(( bitmap^=SIDEMASK ));
  #
  #  | | | | | |       
  #  +-+-+-+-+-+  
  #  BOUND1はt_x[0]
  #
  #  ２．下部サイド枝刈り
  #  if ((row==BOUND2));then     
  #    if (( !(down&SIDEMASK) ));then
  #      return ;
  #    fi
  #    if (( (down&SIDEMASK)!=SIDEMASK ));then
  #      bitmap=$(( bitmap&SIDEMASK ));
  #    fi
  #  fi
  #
  #  ２．最下段枝刈り
  #  LSATMASKの意味は最終行でBOUND1以下または
  #  BOUND2以上にクイーンは置けないということ
  #  BOUND2はsize-t_x[0]
  #  if(row==sizeE){
  #    //if(!bitmap){
  #    if(bitmap){
  #      if((bitmap&LASTMASK)==0){
  ';
    if (( t_x[0]!=-1));then
      ((  (dimx<t_x[0]||dimx>=size-t_x[0])
        &&(dimy==0||dimy==size-1)))&&{
        return 0;
      } 
      ((  (dimx==size-1)&&((dimy<=t_x[0])||
          dimy>=size-t_x[0])))&&{
        return 0;
      } 
    fi
  fi
  (( (B[0] & 1<<dimx)||
        (B[1] & 1<<(size-1-dimx+dimy))||
        (B[2] & 1<<dimy)||
        (B[3] & 1<<(dimx+dimy)) )) && return 0;
  ((B[0]|=1<<dimx));
  ((B[1]|=1<<(size-1-dimx+dimy)));
  ((B[2]|=1<<dimy));
  ((B[3]|=1<<(dimx+dimy)));
  t_x[$dimx]="$dimy"; 
  B[4]=${t_x[@]}; # Bに反映  
  #
  # ボードレイアウト出力
  board[$dimx]=$((1<<dimy));
  #
  return 1;
}
#
: 'キャリーチェーン対象解除法';
function carryChainSymmetry()
{
  local -i n="$1";
  local -i w="$2";
  local -i s="$3";
  local -i e="$4";
  # n,e,s=(N-2)*(N-1)-1-w の場合は最小値を確認する。
  local -i ww=$(( (size-2)*(size-1)-1-w ));
  local -i w2=$(( (size-2)*(size-1)-1 ));
  # 対角線上の反転が小さいかどうか確認する
  (( (s==ww)&&(n<(w2-e)) ))&& return;
  # 垂直方向の中心に対する反転が小さいかを確認
  (( (e==ww)&&(n>(w2-n)) ))&& return;
  # 斜め下方向への反転が小さいかをチェックする
  (( (n==ww)&&(e>(w2-s)) ))&& return ;
  #
  # 【枝刈り】 １行目が角の場合
  #  １．回転対称チェックせずCOUNT8にする
  local -a t_x=(${B[4]}); # 同じ場所の配置を許す
  (( t_x[0] ))||{
    process "$size" "$COUNT8";
    #
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    if ((DISPLAY==1));then printRecordCarryChain "$size" "1"; read -p ""; fi
    #
    return;
  }
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
  ((s==w))&&{
    (( (n!=w)||(e!=w) ))&& return;
    process "$size" "$COUNT2";
    #
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    if ((DISPLAY==1));then printRecordCarryChain "$size" "1"; read -p ""; fi
    #
    return ;
  }
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  (( (e==w)&&(n>=s) ))&&{
    ((n>s))&& return ;
    process "$size" "$COUNT4";
    #
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    if ((DISPLAY==1));then printRecordCarryChain "$size" "1"; read -p ""; fi
    #
    return ;
  }
  process "$size" "$COUNT8";
  #
  # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
  if ((DISPLAY==1));then printRecordCarryChain "$size" "1"; read -p ""; fi
  #
  return ;
}
#
: 'チェーンの構築';
function carryChain()
{
  local -i size="$1";
  # チェーンの初期化
  local -a pres_a;
  local -a pres_b;
  local -i idx=0;
  local -i a=b=0;
  for ((a=0;a<size;a++));do
    for ((b=0;b<size;b++));do
      (( ( (a>=b)&&((a-b)<=1) )||
            ( (b>a)&& ((b-a)<=1) ) )) && continue;
      pres_a[$idx]=$a;
      pres_b[$idx]=$b;
      ((idx++));
    done
  done
  #
  # チェーンのビルド
  local -a wB=sB=eB=nB=X; 
  wB=("${B[@]}");
  for ((w=0;w<=(size/2)*(size-3);w++));do
    B=("${wB[@]}");
    # Bの初期化 #0:row 1:left 2:down 3:right 4:dimx
    for ((bx_i=0;bx_i<size;bx_i++));do X[$bx_i]=-1; done
    B=([0]=0 [1]=0 [2]=0 [3]=0 [4]=${X[@]});
    placement "$size" "0" "$((pres_a[w]))"; # １　０行目と１行目にクイーンを配置
    [[ $? -eq 0 ]] && continue;
    placement "$size" "1" "$((pres_b[w]))";
    [[ $? -eq 0 ]] && continue;
    #
    # ２ 90度回転
    nB=("${B[@]}");
    local -i mirror=$(( (size-2)*(size-1)-w ));
    for ((n=w;n<mirror;n++));do 
      B=("${nB[@]}");
      placement "$size" "$((pres_a[n]))" "$((size-1))"; 
      [[ $? -eq 0 ]] && continue;
      placement "$size" "$((pres_b[n]))" "$((size-2))";
      [[ $? -eq 0 ]] && continue;
      #
      # ３ 90度回転
      eB=("${B[@]}");
      for ((e=w;e<mirror;e++));do 
        B=("${eB[@]}");
        placement "$size" "$((size-1))" "$((size-1-pres_a[e]))"; 
        [[ $? -eq 0 ]] && continue;
        placement "$size" "$((size-2))" "$((size-1-pres_b[e]))"; 
        [[ $? -eq 0 ]] && continue;
        #
        # ４ 90度回転
        sB=("${B[@]}");
        for ((s=w;s<mirror;s++));do
          B=("${sB[@]}")
          placement "$size" "$((size-1-pres_a[s]))" "0";
          [[ $? -eq 0 ]] && continue;
          placement "$size" "$((size-1-pres_b[s]))" "1"; 
          [[ $? -eq 0 ]] && continue;
          #
          #  対象解除法
          carryChainSymmetry "$n" "$w" "$s" "$e" ; 
          continue;
        done
      done
    done
  done
  # 集計
  UNIQUE=COUNTER[$COUNT2]+COUNTER[$COUNT4]+COUNTER[$COUNT8];
  TOTAL=COUNTER[$COUNT2]*2+COUNTER[$COUNT4]*4+COUNTER[$COUNT8]*8;
}
#
: '';
function NQ()
{
  local selectName="$1";
  local -i min=4;
  local -i max=15;
  local -i N="$min";
  local startTime=endTime=hh=mm=ss=0; 
  echo " N:        Total       Unique        hh:mm:ss" ;
  local -i N;
  for((N=min;N<=max;N++)){
    TOTAL=0;
    UNIQUE=0;
    COUNTER[0]=COUNTER[1]=COUNTER[2]=0;    # カウンター配列
    B=0; 
    startTime=$(date +%s);# 計測開始時間
    "$selectName" "$N";
    endTime=$(date +%s); 	# 計測終了時間
    ss=$((endTime-startTime));# hh:mm:ss 形式に変換
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}
#
#
DISPLAY=0; # ボードレイアウト表示しない
#DISPLAY=1; # ボードレイアウト表示する
#
NQ carryChain; 
exit;

