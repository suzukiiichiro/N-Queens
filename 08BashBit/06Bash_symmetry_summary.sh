#!/usr/bin/bash

: '
エイト・クイーンの解説をしています。ご興味のある方はどうぞ。

Ｎクイーン問題（１８）第四章　偉人のソースを読む「エイト・クイーンノスタルジー」
https://suzukiiichiro.github.io/posts/2023-04-18-01-n-queens-suzuki/
Ｎクイーン問題（１７）第四章　偉人のソースを読む「Ｎ２４を発見 Ｊｅｆｆ Ｓｏｍｅｒｓ」
https://suzukiiichiro.github.io/posts/2023-04-21-01-n-queens-suzuki/
Ｎクイーン問題（１６）第三章　対象解除法 ソース解説
https://suzukiiichiro.github.io/posts/2023-04-18-01-n-queens-suzuki/
Ｎクイーン問題（１５）第三章　対象解除法 ロジック解説
https://suzukiiichiro.github.io/posts/2023-04-13-02-nqueens-suzuki/
Ｎクイーン問題（１４）第三章　ミラー
https://suzukiiichiro.github.io/posts/2023-04-13-01-nqueens-suzuki/
Ｎクイーン問題（１３）第三章　ビットマップ
https://suzukiiichiro.github.io/posts/2023-04-05-01-nqueens-suzuki/
Ｎクイーン問題（１２）第二章　まとめ
https://suzukiiichiro.github.io/posts/2023-03-17-02-n-queens-suzuki/
Ｎクイーン問題（１１）第二章　配置フラグの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-17-01-n-queens-suzuki/
Ｎクイーン問題（１０）第二章　バックトラックの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-16-01-n-queens-suzuki/
Ｎクイーン問題（９）第二章　ブルートフォースの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-14-01-n-queens-suzuki/
Ｎクイーン問題（８）第一章　まとめ
https://suzukiiichiro.github.io/posts/2023-03-09-01-n-queens-suzuki/
Ｎクイーン問題（７）第一章　ブルートフォース再び
https://suzukiiichiro.github.io/posts/2023-03-08-01-n-queens-suzuki/
Ｎクイーン問題（６）第一章　配置フラグ
https://suzukiiichiro.github.io/posts/2023-03-07-01-n-queens-suzuki/
Ｎクイーン問題（５）第一章　進捗表示テーブルの作成
https://suzukiiichiro.github.io/posts/2023-03-06-01-n-queens-suzuki/
Ｎクイーン問題（４）第一章　バックトラック
https://suzukiiichiro.github.io/posts/2023-02-21-01-n-queens-suzuki/
Ｎクイーン問題（３）第一章　バックトラック準備編
https://suzukiiichiro.github.io/posts/2023-02-14-03-n-queens-suzuki/
Ｎクイーン問題（２）第一章　ブルートフォース
https://suzukiiichiro.github.io/posts/2023-02-14-02-n-queens-suzuki/
Ｎクイーン問題（１）第一章　エイトクイーンについて
https://suzukiiichiro.github.io/posts/2023-02-14-01-n-queens-suzuki/

エイト・クイーンのソース置き場 BashもJavaもPythonも！
https://github.com/suzukiiichiro/N-Queens
';

: '
 ## bash版
 <> symmetry.sh matome5.sh 対象解除
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
declare -i size;
declare -a board;
declare -i bit;
declare -i DISPLAY=0;   # ボード出力するか
declare -i TOTAL=UNIQUE=0;
declare -i COUNT2=COUNT4=COUNT8=0;
declare -i MASK=SIDEMASK=LASTMASK=0;
declare -i TOPBIT=ENDBIT=0;
declare -i BOUND1=BOUND2=0;

#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecord()
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
        if (( board[i]&1<<j ));then
          echo -n "O";
        else
          echo -n " ";
        fi
      else
        if((i==board[j]));then
          echo -n "O";
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
: '再帰・非再帰版 対象解除法';
function symmetryOps()
{
  : '
  ２．クイーンが右上角以外にある場合、
  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  ジナルと同型になる。
  こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  ンを加えて２個しかありません。
  ';
  ((board[BOUND2]==1))&&{
    for((ptn=2,own=1;own<=size-1;own++,ptn<<=1)){
      for((bit=1,you=size-1;(board[you]!=ptn)&&(board[own]>=bit);you--)){
        ((bit<<=1));
      }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
    #90度回転して同型なら180度回転も270度回転も同型である
    ((own>size-1))&&{
      ((COUNT2++)); 
      if ((DISPLAY==1));then
        # 出力 1:bitmap版 0:それ以外
        printRecord "$size" "1";          
      fi
      return; 
    }
  }
  : '
  ２．クイーンが右上角以外にある場合、
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
    とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
    る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
    型になる場合は４個(左右反転×縦横回転)
  ';
  #180度回転
  ((board[size-1]==ENDBIT))&&{ 
    for ((you=size-1-1,own=1;own<=size-1;own++,you--)){
      for ((bit=1,ptn=TOPBIT;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)){
          ((bit<<=1)) ;
        }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
    #90度回転が同型でなくても180度回転が同型であることもある
    ((own>size-1))&&{ 
      ((COUNT4++)); 
      if ((DISPLAY==1));then
        # 出力 1:bitmap版 0:それ以外
        printRecord "$size" "1";          
      fi
      return; 
    }
  }
  : '
  ２．クイーンが右上角以外にある場合、
    (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  ';
  #270度回転
  ((board[BOUND1]==TOPBIT))&&{ 
    for((ptn=TOPBIT>>1,own=1;own<=size-1;own++,ptn>>=1)){
      for((bit=1,you=0;(board[you]!=ptn)&&(board[own]>=bit);you++)){
          ((bit<<=1)) ;
        }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
  }
  ((COUNT8++));
  if ((DISPLAY==1));then
    # 出力 1:bitmap版 0:それ以外
    printRecord "$size" "1";          
  fi
}
#
: '非再帰版 角にQがない時の対象解除バックトラック';
function symmetry_backTrack_NR()
{
  local -i row="$1";
  local -i MASK="$(( (1<<size)-1 ))";
  local -a bitmap[$size];
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -i bit=0;
  left[$row]="$2";
  down[$row]="$3";
  right[$row]="$4";
  bitmap[$row]=$(( MASK&~(left[row]|down[row]|right[row]) ));
  while ((row>0));do
    if (( bitmap[row]>0 ));then
      # 一番右のビットを取り出す
      bit=$(( -bitmap[row]&bitmap[row] ));  
      # 配置可能なパターンが一つずつ取り出される
      bitmap[$row]=$(( bitmap[row]^bit ));  
      board[$row]="$bit";            # Qを配置
      if ((row<BOUND1));then         #上部サイド枝刈り
        bitmap[$row]=$(( bitmap[row]|SIDEMASK ));
        bitmap[$row]=$(( bitmap[row]^SIDEMASK ));
      elif ((row==BOUND2));then      #下部サイド枝刈り
        if (( (down[row]&SIDEMASK) ==0));then
          return ;
        fi
        if (((down[row]&SIDEMASK)!=SIDEMASK));then
          bitmap[$row]=$(( bitmap[row]&SIDEMASK ));
        fi
      fi
      if (( row==(size-1) ));then
        if ((!(bitmap[row]&LASTMASK)));then
          symmetryOps ;
        fi
        ((row--));
      else
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";         # Qを配置
        # クイーンが配置可能な位置を表す
        bitmap[$row]=$(( MASK&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done
}
#
: '非再帰版 角にQがある時の対象解除バックトラック';
function symmetry_backTrack_corner_NR()
{
  local -i row="$1";
  local -i MASK="$(( (1<<size)-1 ))";
  local -a bitmap[$size];
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -i bit=0;
  left[$row]="$2";
  down[$row]="$3";
  right[$row]="$4";
  bitmap[$row]=$(( MASK&~(left[row]|down[row]|right[row]) ));
  while ((row>0));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        if((bitmap[row]));then
          #枝刈りによりsymmetryOpsは不要
          #symmetryOps ;
          ((COUNT8++)) ;
          if ((DISPLAY==1));then
            # 出力 1:bitmap版 0:それ以外
            printRecord "$size" "1";          
          fi
        fi
        ((row--));
      else
        if ((row<BOUND1));then
          bitmap[$row]=$(( bitmap[row]|2 ));
          bitmap[$row]=$(( bitmap[row]^2 ));
        fi
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";                # Qを配置
        # クイーンが配置可能な位置を表す
        bitmap[$row]=$(( MASK&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done
}
#
: '非再帰版 対象解除';
function symmetry_NR()
{
  size="$1";
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
  MASK=$(( (1<<size)-1 ));
  TOPBIT=$(( 1<<(size-1) )); 
  ENDBIT=LASTMASK=SIDEMASK=0;
  BOUND1=2; BOUND2=0;
  board[0]=1;
  while (( BOUND1>1 && BOUND1<size-1 ));do
    if (( BOUND1<size-1 ));then
      bit=$(( 1<<BOUND1 ));
      board[1]="$bit";          # ２行目にQを配置
      # 角にQがある時のバックトラック
      symmetry_backTrack_corner_NR "2" "$(( (2|bit)<<1 ))" "$(( 1|bit ))" "$(( (2|bit)>>1 ))";
    fi
    (( BOUND1++ ));
  done
  TOPBIT=$(( 1<<(size-1) )); 
  ENDBIT=$(( TOPBIT>>1 ));
  SIDEMASK=$(( TOPBIT|1 ));
  LASTMASK=$(( TOPBIT|1 )); 
  BOUND1=1; 
  BOUND2=$size-2;
  while (( BOUND1>0 && BOUND2<size-1 && BOUND1<BOUND2 ));do
    if (( BOUND1<BOUND2 ));then
      bit=$(( 1<<BOUND1 ));
      board[0]="$bit";          # Qを配置
      # 角にQがない時のバックトラック
      symmetry_backTrack_NR "1" "$(( bit<<1 ))" "$bit" "$(( bit>>1 ))";
    fi 
    (( BOUND1++,BOUND2-- ));
    ENDBIT=$(( ENDBIT>>1 ));
    LASTMASK=$(( LASTMASK<<1 | LASTMASK | LASTMASK>>1 )) ;
  done
  UNIQUE=$(( COUNT8+COUNT4+COUNT2 )) ;
  TOTAL=$(( COUNT8*8+COUNT4*4+COUNT2*2 ));
}
#
: '再帰版 角にQがない時の対象解除バックトラック';
function symmetry_backTrack()
{
  local row=$1;
  local left=$2;
  local down=$3;
  local right=$4; 
  local bitmap=$(( MASK&~(left|down|right) ));
  if ((row==(size-1) ));then
    if ((bitmap));then
      if (( !(bitmap&LASTMASK) ));then
        board[row]="$bitmap";     # Qを配置
        symmetryOps ;             # 対象解除
      fi
    fi
  else
    if ((row<BOUND1));then        # 上部サイド枝刈り
      bitmap=$(( bitmap|SIDEMASK ));
      bitmap=$(( bitmap^=SIDEMASK ));
    else 
      if ((row==BOUND2));then     # 下部サイド枝刈り
        if (( !(down&SIDEMASK) ));then
          return ;
        fi
        if (( (down&SIDEMASK)!=SIDEMASK ));then
          bitmap=$(( bitmap&SIDEMASK ));
        fi
      fi
    fi
    while((bitmap));do
      bit=$(( -bitmap & bitmap )) ;
      bitmap=$(( bitmap^bit));
      board[row]="$bit"             # Qを配置
      symmetry_backTrack $((row+1)) $(((left|bit)<<1))  $((down|bit)) $(((right|bit)>>1));
    done
  fi
}
#
: '再帰版 角にQがある時の対象解除バックトラック';
function symmetry_backTrack_corner()
{
  local row=$1;
  local left=$2;
  local down=$3;
  local right=$4; 
  local bitmap=$(( MASK&~(left|down|right) ));
  if ((row==(size-1) ));then
    if ((bitmap));then
      board[$row]="$bitmap";
      if ((DISPLAY==1));then
        printRecord "$size" 1 ;
      fi
      ((COUNT8++)) ;              
    fi
  else
    if ((row<BOUND1));then        # 枝刈り
      bitmap=$(( bitmap|2 ));
      bitmap=$(( bitmap^2 ));
    fi
    while((bitmap));do
      bit=$(( -bitmap & bitmap )) ;
      bitmap=$(( bitmap^bit));
      board[row]="$bit"           # Qを配置
      symmetry_backTrack_corner $((row+1)) $(((left|bit)<<1))  $((down|bit)) $(((right|bit)>>1));
    done
  fi
}
#
: '再帰版 対象解除';
function symmetry()
{
  size="$1";
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
  MASK=$(( (1<<size)-1 ));
  TOPBIT=$(( 1<<(size-1) )); 
  ENDBIT=LASTMASK=SIDEMASK=0;
  BOUND1=2; BOUND2=0;
  board[0]=1;
  while (( BOUND1>1 && BOUND1<size-1 ));do
    if (( BOUND1<size-1 ));then
      bit=$(( 1<<BOUND1 ));
      board[1]="$bit";          # ２行目にQを配置
      # 角にQがある時のバックトラック
      symmetry_backTrack_corner "2" "$(( (2|bit)<<1 ))" "$(( 1|bit ))" "$(( (2|bit)>>1 ))";
    fi
    (( BOUND1++ ));
  done
  TOPBIT=$(( 1<<(size-1) )); 
  ENDBIT=$(( TOPBIT>>1 ));
  SIDEMASK=$(( TOPBIT|1 ));
  LASTMASK=$(( TOPBIT|1 )); 
  BOUND1=1; 
  BOUND2=size-2;
  while (( BOUND1>0 && BOUND2<size-1 && BOUND1<BOUND2 ));do
    if (( BOUND1<BOUND2 ));then
      bit=$(( 1<<BOUND1 ));
      board[0]="$bit";          # Qを配置
      # 角にQがない時のバックトラック
      symmetry_backTrack "1" "$(( bit<<1 ))" "$bit" "$(( bit>>1 ))";
    fi 
    (( BOUND1++,BOUND2-- ));
    ENDBIT=$(( ENDBIT>>1 ));
    LASTMASK=$(( LASTMASK<<1 | LASTMASK | LASTMASK>>1 )) ;
  done
  UNIQUE=$(( COUNT8+COUNT4+COUNT2 )) ;
  TOTAL=$(( COUNT8*8+COUNT4*4+COUNT2*2 ));
}
#
: '非再帰版ミラーロジック';
function mirror_solve_NR()
{
  local -i size="$1";
  local -i row="$2";
  local -i mask="$(( (1<<size)-1 ))";
  local -a bitmap[$size];
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -i bit=0;
  left[$row]="$3";
  down[$row]="$4";
  right[$row]="$5";
  bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((COUNT2++));
        if ((DISPLAY==1));then
          printRecord "$size" "1";            # 出力 1:bitmap版 0:それ以外
        fi
        ((row--));
      else
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";                 # Qを配置
        # クイーンが配置可能な位置を表す
        bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done
}
#
: '非再帰版ミラー';
function mirror_NR()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  : '
    if ((size%2));then                #  以下のif文と等価です
      limit="$((size/2-1))";
    else
      limit="$((size/2))";
    fi
  ';
  local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    mirror_solve_NR "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      mirror_solve_NR "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする

}
#
: '再帰版ミラーロジック';
function mirror_solve_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i left="$3";
  local -i down="$4";
  local -i right="$5";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit;
  local -i bitmap;
  if (( row==size ));then
    ((COUNT2++));
    if ((DISPLAY));then
      printRecord "$size" "1";       # 出力 1:bitmap版 0:それ以外
    fi
  else
    # Qが配置可能な位置を表す
    bitmap="$(( mask&~(left|down|right) ))";
    while ((bitmap));do
      bit="$(( -bitmap&bitmap ))"; # 一番右のビットを取り出す
      bitmap="$(( bitmap^bit ))";  # 配置可能なパターンが一つずつ取り出される
      board["$row"]="$bit";        # Qを配置
      mirror_solve_R "$size" "$((row+1))" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
#
: '再帰版ミラー';
function mirror_R()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  : '
    if ((size%2));then                #  以下のif文と等価です
      limit="$((size/2-1))";
    else
      limit="$((size/2))";
    fi
  ';
  local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    mirror_solve_R "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      mirror_solve_R "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする
}
#
: '非再帰版ビットマップ';
function bitmap_NR()
{ 
  local -i size="$1";
  local -i row="$2";
  local -i mask=$(( (1<<size)-1 ));
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -a bitmap[$size]
  local -i bitmap[$row]=mask;
  local -i bit=0;
  bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((TOTAL++));
        if ((DISPLAY==1));then
          printRecord "$size" "1";            # 出力 1:bitmap版 0:それ以外
        fi
        ((row--));
      else
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";                 # Qを配置
        # クイーンが配置可能な位置を表す
        bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done 

}
#
: '再帰版ビットマップ';
function bitmap_R()
{
  local -i size="$1"; 
  local -i row="$2";
  local -i mask="$3";
  local -i left="$4"; 
  local -i down="$5"; 
  local -i right="$6";
  local -i bitmap=;
  local -i bit=;
  local -i col=0;                     # 再帰に必要
  if (( row==size ));then
    ((TOTAL++));
    if ((DISPLAY==1));then
      printRecord "$size" "1";         # 出力 1:bitmap版 0:それ以外
    fi
  else
    bitmap=$(( mask&~(left|down|right) )); # クイーンが配置可能な位置を表す
    while (( bitmap ));do
      bit=$((-bitmap&bitmap)) ;       # 一番右のビットを取り出す
      bitmap=$((bitmap&~bit)) ;       # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";             # Qを配置
      bitmap_R "$size" "$((row+1))" "$mask" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
#
: '非再帰版配置フラグ(right/down/left flag)';
function postFlag_NR()
{
  local -i size="$1";
  local -i row="$2"
  local -i matched=0;
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    matched=0;
    for ((col=board[row]+1;col<size;col++)){
      if (( !down[col]
        &&  !right[col-row+size-1]
        &&  !left[col+row] ));then
        dix=$col;
        rix=$((row-col+(size-1)));
        lix=$((row+col));
        if ((board[row]!=-1));then
          down[${board[$row]}]=0;
          right[${board[$row]}-$row+($size-1)]=0;
          left[${board[$row]}+$row]=0;
        fi       
        board[$row]=$col;   # Qを配置
        down[$col]=1;
        right[$col-$row+($size-1)]=1;
        left[$col+$row]=1;  # 効き筋とする
        matched=1;          # 配置した
        break;
      fi
    }
    if ((matched));then     # 配置済み
      ((row++));            #次のrowへ
      if ((row==size));then
        ((TOTAL++));
        if ((DISPLAY==1));then
          printRecord "$size";# 出力
        fi
        ((row--));
      fi
    else
      if ((board[row]!=-1));then
        down[${board[$row]}]=0;
        right[${board[$row]}-$row+($size-1)]=0;
        left[${board[$row]}+$row]=0;
        board[$row]=-1;
      fi
      ((row--));            # バックトラック
    fi
  done
}
#
: '再帰版配置フラグ';
function postFlag_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=0;       # 再帰に必要
  if (( row==size ));then
     ((TOTAL++));
    if (( DISPLAY==1 ));then
      printRecord "$size";# 出力
    fi
  else
    for(( col=0;col<size;col++ )){
      board[$row]="$col";
      if (( down[col]==0 
        && right[row-col+size-1]==0
        && left[row+col]==0));then
        down[$col]=1;
        right[$row-$col+($size-1)]=1;
        left[$row+$col]=1;
        postFlag_R "$size" "$((row+1))";
        down[$col]=0;
        right[$row-$col+($size-1)]=0;
        left[$row+$col]=0;
      fi
    }
  fi
}
#
: 'バックトラック版効き筋をチェック';
function check_backTracking()
{
  local -i row="$1";
  local -i flag=1;
  for ((i=0;i<row;++i)){
    if (( board[i]>=board[row] ));then
      val=$(( board[i]-board[row] ));
    else
      val=$(( board[row]-board[i] ));
    fi
    if (( board[i]==board[row] || val==(row-i) ));then
      flag=0;
    fi
  }
  [[ $flag -eq 0 ]]
  return $?;
}
#
: '非再帰版バックトラック';
function backTracking_NR()
{
  local -i size="$1";
  local -i row="$2";
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    local -i matched=0;
    local -i col=0;  
    for((col=board[row]+1;col<size;col++)){
      board[$row]=$col;
      check_backTracking "$row";  # 効きをチェック
      if (($?==1));then # 直前のreturnを利用
        matched=1;
        break;
      fi
    }
    if ((matched));then
      ((row++));
      if ((row==size));then  # 最下部まで到達
        ((row--));
        ((TOTAL++));
        if (( DISPLAY==1 ));then
          printRecord "$size";# 出力
        fi
      fi
    else
      if ((board[row]!=-1));then
        board[$row]=-1;
      fi
      ((row--));
    fi
 done  
}
#
: '再帰版バックトラック';
function backTracking_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=0;
  if ((row==size));then
    ((TOTAL++));
    if (( DISPLAY==1 ));then
      printRecord "$size";# 出力
    fi
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      check_backTracking "$row";
      if (($?==1));then 
        backTracking_R  $size $((row+1));
      fi
    }
  fi
}
#
: 'ブルートフォース版効き筋をチェック';
function check_bluteForce()
{
  local -i size="$1";
  local -i flag=1;
  for ((r=1;r<size;++r)){
    for ((i=0;i<r;++i)){
      if (( board[i]>=board[r] ));then
        val=$(( board[i]-board[r] ));
      else
        val=$(( board[r]-board[i] ));
      fi

      if (( board[i]==board[r] || val==(r-i) ));then
        flag=0; 
      fi
    }
  }
  [[ $flag -eq 0 ]]
  return $?;
}
#
: '非再帰版ブルートフォース';
function bluteForce_NR()
{
  local -i size="$1";
  local -i row="$2";
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    local -i matched=0;
    local -i col=0;  
    for((col=board[row]+1;col<size;col++)){
      board[$row]=$col;
      matched=1;
      break;
    }
    if ((matched));then
      ((row++));
      if ((row==size));then  # 最下部まで到達
        ((row--));
        check_bluteForce "$size";  # 効きをチェック
        if (($?==1));then # 直前のreturnを利用
          ((TOTAL++));
          if (( DISPLAY==1 ));then
            printRecord "$size";# 出力
          fi
        fi
      fi
    else
      if ((board[row]!=-1));then
        board[$row]=-1;
      fi
      ((row--));
    fi
 done  
}
#
: '再帰版ブルートフォース';
function bluteForce_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=;
  if ((row==size));then
    check_bluteForce "$size";
    if (( $?==1 ));then 
      ((TOTAL++));
      if (( DISPLAY==1 ));then
        printRecord "$size";# 出力
      fi
    fi
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      bluteForce_R  $size $((row+1));
    }
  fi
}
#
function NQ()
{
  local selectName="$1";
  local -i max=17;
  local -i min=4;
  local -i N="$min";
  local -i mask=0;
  local -i bit=0
  local -i row=0;
  local startTime=0;
  local endTime=0;
  local hh=mm=ss=0; 
  echo " N:        Total       Unique        hh:mm:ss" ;
  local -i N;
  for((N=min;N<=max;N++)){
    TOTAL=0; UNIQUE=0; COUNT2=0; row=0;
    mask=$(( (1<<N)-1 ));
    startTime=$(date +%s);# 計測開始時間

    "$selectName" "$N" "$row" "$mask" 0 0 0;

    endTime=$(date +%s); 	# 計測終了時間
    ss=$((endTime-startTime));# hh:mm:ss 形式に変換
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}

while :
do
read -n1 -p "
エイト・クイーン メニュー
実行したい番号を選択
6) 対象解除法
5) ミラー
4) ビットマップ
3) 配置フラグ 
2) バックトラック 
1) ブルートフォース 

echo "行頭の番号を入力してください";

" selectNo;
echo 
case "$selectNo" in
  6)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ symmetry; break; ;;
        n|N) NQ symmetry_NR; break; ;;
      esac
    done
    ;;
  5)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ mirror_R; break; ;;
        n|N) NQ mirror_NR; break; ;;
      esac
    done
    ;;
  4)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ bitmap_R; break; ;;
        n|N) NQ bitmap_NR; break; ;;
      esac
    done
    ;;
  3)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ postFlag_R; break; ;;
        n|N) NQ postFlag_NR; break; ;;
      esac
    done
    ;;
  2)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰

      " select;
      echo; 
      case "$select" in
    
        y|Y) NQ backTracking_R; break; ;;
        n|N) NQ backTracking_NR; break; ;;
      esac
    done
    ;;
  1)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰

      " select;
      echo; 
      case "$select" in
        y|Y) NQ bluteForce_R; break; ;;
        n|N) NQ bluteForce_NR;break; ;;
      esac
    done
    ;;
  *)
    ;; 
esac
done
exit;

