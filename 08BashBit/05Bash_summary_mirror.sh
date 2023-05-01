#!/usr/bin/bash

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
: ' 対象解除説明
  クイーンの利き筋を辿っていくと、オリジナルがユニーク解ではない可能性があり、
  それは下図の A,B,C の位置のどこかにクイーンがある場合に限られます。

  symmetryOpsは、以下の図のＡ，Ｂ，ＣにＱが置かれた場合にユニーク解かを判定します。
  原型と、90,180,270回転させたもののユニーク値を比較します。

     0 1 2 3 4 
    +-+-+-+-+-+  
    | | | | |Q|    4
    +-+-+-+-+-+  
    | | |Q| | |    2 
    +-+-+-+-+-+  
    |Q| | | | |    0  ----> 4 2 0 3 1 （ユニーク判定値）
    +-+-+-+-+-+             数が大きい方をユニークとみなす 
    | | | |Q| |    3
    +-+-+-+-+-+  
    | |Q| | | |    1 
    +-+-+-+-+-+  

     0 1 2 3 4     左右反転！
    +-+-+-+-+-+  
    |Q| | | | |    0
    +-+-+-+-+-+  
    | | |Q| | |    2 
    +-+-+-+-+-+  
    | | | | |Q|    4  ----> 0 2 4 1 3 
    +-+-+-+-+-+            数が小さいのでユニーク解とはしません 
    | |Q| | | |    1
    +-+-+-+-+-+  
    | | | |Q| |    3 
    +-+-+-+-+-+  

  Qができるだけ右に置かれている方がユニーク値は大きくなります。
  例えば1行目の2列目にQが置かれている方が、
  3列目に置かれているよりユニーク値は大きくユニーク解に近い。
  1行目のクイーンの位置が同じなら2行目のクイーンの位置がより右の列におかれてい
  るものがユニーク値は大きくユニーク解に近くなります。

  それ以外はユニーク解なのでCOUNT8にする
   +-+-+-+-+-+-+-+-+  
   |X|X| | | |Q|X|X| 
   +-+-+-+-+-+-+-+-+  
   |X| | | |x|x|x|X| 
   +-+-+-+-+-+-+-+-+  
   |C| | |x| |x| |x|       
   +-+-+-+-+-+-+-+-+             
   | | |x| | |x| | | 
   +-+-+-+-+-+-+-+-+ 
   | |x| | | |x| | |      
   +-+-+-+-+-+-+-+-+  
   |x| | | | |x| |A|
   +-+-+-+-+-+-+-+-+
   |X| | | | |x| |X|
   +-+-+-+-+-+-+-+-+
   |X|X|B| | |x|X|X|
   +-+-+-+-+-+-+-+-+
   
   Aの場合 右90度回転   board[BOUND2]==1
   Bの場合 右180度回転  board[size-1]==ENDBIT
   Cの場合 右270度回転  board[BOUND1]==TOPBIT
';
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
    ((own>size-1))&&{ ((COUNT2++)); return; }
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
    ((own>size-1))&&{ ((COUNT4++)); return; }
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
}
#
: '再帰版対象解除バックトラック';
function symmetry_backTrack()
{
  local -i row=$1;
  local -i left=$2;
  local -i down=$3;
  local -i right=$4; 
  local -i corner=$5;                  # Qが角にある:1 ない:0
  local -i MASK=$(( (1<<size)-1 ));
  local bitmap=$(( MASK&~(left|down|right) ));
  if ((row==(size-1) ));then
    if ((bitmap));then
      if ((corner));then            # Qが角にある
        : '
        １．クイーンが右上角にある場合、ユニーク解が属する
        グループの要素数は必ず８個(＝２×４)
        ';
        board[$row]="$bitmap";
        if ((DISPLAY==1));then
          printRecord "$size" 1 ;
        fi
        ((COUNT8++)) ;              # 角にある場合は８倍するカウンター
      else                          # Qが角にない
        if (( !(bitmap&LASTMASK) ));then
          board[row]="$bitmap";     # Qを配置
          symmetryOps ;             # 対象解除
        fi
      fi
    fi
  else
    if (( corner ));then            # Qが角にある
      if ((row<BOUND1));then        # 枝刈り
        #bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        #bitmap=$(( bitmap&~2 ))
         bitmap=$(( bitmap|2 ));
         bitmap=$(( bitmap^2 ));
        : '
        上から２行目のクイーンの位置が左から何番目にあるかと、
        右から２列目のクイーンの位置が上から何番目にあるかを、
        比較するだけで判定します。
        具体的には、２行目と２列目の位置を数値とみなし、
        ２行目＜２列目という条件を課せばよい
        結論： 以下の図では、１，２，４を枝刈りを入れる
          
          +-+-+-+-+-+  
          | | | |X|Q| 
          +-+-+-+-+-+  
          | |Q| |X| |  8（左から数えて１，２，４，８）
          +-+-+-+-+-+  
          | | | |X| |       
          +-+-+-+-+-+             
          | | | |Q| |  8（上から数えて１，２，４，８） 
          +-+-+-+-+-+ 
          | | | | | |      
          +-+-+-+-+-+  
        ';
      fi
    else                            # Qが角にない
      : '
      オリジナルがユニーク解であるためには先ず、
      前提：symmetryOpsは回転・鏡像変換により得られる状態の
      ユニーク値を比較し最小のものだけがユニーク解となるようにしている。
      Qができるだけ右に置かれている方がユニーク値は小さい。
      例えば1行目の2列目にQが置かれている方が3列目に置かれているより
      ユニーク値は小さくユニーク解に近い。
      1行目のクイーンの位置が同じなら2行目のクイーンの位置がより右の
      列におかれているものがユニーク値は小さくユニーク解に近い。

      下図の X への配置は禁止されます。
      Qの位置より右位置の８対象位置（X）にクイーンを置くことはできない。
      置いた場合回転・鏡像変換したユニーク値が最小にならなくなり、symmetryOps
      で負けるので枝刈りをする


      1行目のクイーンが3列目に置かれている場合
      +-+-+-+-+-+-+-+-+  
      |X|X| | | |Q|X|X| 
      +-+-+-+-+-+-+-+-+  
      |X| | | | | | |X| 
      +-+-+-+-+-+-+-+-+  
      | | | | | | | | |       
      +-+-+-+-+-+-+-+-+             
      | | | | | | | | | 
      +-+-+-+-+-+-+-+-+ 
      | | | | | | | | |      
      +-+-+-+-+-+-+-+-+  
      | | | | | | | | |
      +-+-+-+-+-+-+-+-+
      |X| | | | | | |X|
      +-+-+-+-+-+-+-+-+
      |X|X| | | | |X|X|
      +-+-+-+-+-+-+-+-+

      1行目のクイーンが4列目に置かれている場合
      +-+-+-+-+-+-+-+-+  
      |X|X|X| |Q|X|X|X| 
      +-+-+-+-+-+-+-+-+  
      |X| | | | | | |X| 
      +-+-+-+-+-+-+-+-+  
      |X| | | | | | |X|       
      +-+-+-+-+-+-+-+-+             
      | | | | | | | | | 
      +-+-+-+-+-+-+-+-+ 
      | | | | | | | | |      
      +-+-+-+-+-+-+-+-+  
      |X| | | | | | |X|
      +-+-+-+-+-+-+-+-+
      |X| | | | | | |X|
      +-+-+-+-+-+-+-+-+
      |X|X|X| | |X|X|X|
      +-+-+-+-+-+-+-+-+

      プログラムではこの枝刈を上部サイド枝刈り、下部サイド枝刈り、最下段枝刈り
      の3か所で行っている
      それぞれ、1,2,3の数字で表すと以下の通り

      1行目のクイーンが3列目に置かれている場合
      +-+-+-+-+-+-+-+-+  
      |X|X| | | |Q|X|X| 
      +-+-+-+-+-+-+-+-+  
      |1| | | | | | |1| 
      +-+-+-+-+-+-+-+-+  
      | | | | | | | | |       
      +-+-+-+-+-+-+-+-+             
      | | | | | | | | | 
      +-+-+-+-+-+-+-+-+ 
      | | | | | | | | |      
      +-+-+-+-+-+-+-+-+  
      | | | | | | | | |
      +-+-+-+-+-+-+-+-+
      |2| | | | | | |2|
      +-+-+-+-+-+-+-+-+
      |2|3| | | | |3|2|
      +-+-+-+-+-+-+-+-+
      1行目にXが残っているが当然Qの効き筋なので枝刈する必要はない
      ';
      if ((row<BOUND1));then        # 上部サイド枝刈り
        bitmap=$(( bitmap|SIDEMASK ));
        bitmap=$(( bitmap^=SIDEMASK ));
      fi
      if ((row==BOUND2));then       # 下部サイド枝刈り
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
      symmetry_backTrack $((row+1)) $(((left|bit)<<1))  $((down|bit)) $(((right|bit)>>1)) "$corner" ;
    done
  fi
}
#
: '再帰・非再帰版版 対象解除';
function symmetry()
{
  size="$1"
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
  : '
  角にQがある時の処理
    +-+-+-+-+-+  
    | | | | |Q| 
    +-+-+-+-+-+  
    | | | | | | 
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  MASK=$(( (1<<size)-1 ));
  : '
    +-+-+-+-+-+  
    |M|M|M|M|M|  MASK=$(( (1<<size)-1 )) 
    +-+-+-+-+-+  $(( (1<<5)-1 )):31
    | | | | | |  $(( 2#11111 )) 
    +-+-+-+-+-+  $ 31
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  TOPBIT=$(( 1<<(size-1) )); 
  : '
    +-+-+-+-+-+  
    |T| | | | |  TOPBIT=$(( 1<<(size-1) ))
    +-+-+-+-+-+  $(( (1<<(5-1) )):16 
    | | | | | |  $(( 2#11111 )) 
    +-+-+-+-+-+  $ 16 
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  ENDBIT=LASTMASK=SIDEMASK=0;
  : '
    ０は何も置かない（なにもはじまってない）
    +-+-+-+-+-+  
    | | | | | |  0#00000
    +-+-+-+-+-+  
    | | | | | | 
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  BOUND1=2;
  board[0]=1;
  while (( BOUND1>1 && BOUND1<(size-1) ));do
    if (( BOUND1<size-1 ));then
      bit=$(( 1<<BOUND1 ));
      : '
       Ｎ５の場合、BOUND1は２と３になる
        (( 1<<BOUND1 )) ２行目は真ん中に置く
        +-+-+-+-+-+  
        | | | | |Q|  1#00001
        +-+-+-+-+-+  
        | | |B| | |  BOUND1=2 4#00100
        +-+-+-+-+-+  $(( 1<<2 ))
        | | | | | |  $ 4
        +-+-+-+-+-+  1->2->4               
        | | | | | | 
        +-+-+-+-+-+ 
        | | | | | |      
        +-+-+-+-+-+  
        (( 1<<BOUND1 )) ２行目は真ん中の次に置く
        +-+-+-+-+-+  
        | | | | |Q|  1#00001
        +-+-+-+-+-+  
        | |B| | | |  BOUND1=3 8#01000
        +-+-+-+-+-+  $(( 1<<3 ))
        | | | | | |  $ 8
        +-+-+-+-+-+  1->2->4->8             
        | | | | | | 
        +-+-+-+-+-+ 
        | | | | | |      
        +-+-+-+-+-+  
      ';
      board[1]="$bit";          # ２行目にQを配置
      symmetry_backTrack "2" "$(( (2|bit)<<1 ))" "$(( 1|bit ))" "$(( bit>>1 ))" "1";
    fi
    (( BOUND1++ ));
  done
  : '
  角にQがない時の処理
    +-+-+-+-+-+  
    | | | |Q| | $(( 2#00010 ))
    +-+-+-+-+-+ $ 2 
    | | | | | | 
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  TOPBIT=$(( 1<<(size-1) )); 
  : '
    +-+-+-+-+-+  
    |T| | | | | TOPBIT=$(( 1<<(size-1) ))
    +-+-+-+-+-+ $(( 2#10000 )) 
    | | | | | | $ 16 
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  ENDBIT=$(( TOPBIT>>1 ));
  : '
    +-+-+-+-+-+  
    |T|E| | | | ENDBIT=$(( TOPBIT>>1 ))
    +-+-+-+-+-+ $(( 2#01000 )) 
    | | | | | | $ 8
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  SIDEMASK=$(( TOPBIT|1 ));
  : '
    +-+-+-+-+-+  
    |S| | | |S| SIDEMASK=$(( TOPBIT|1 ))
    +-+-+-+-+-+ $(( 2#10001 )) 
    | | | | | | $ 17
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  LASTMASK=$(( TOPBIT|1 )); 
  : '
    +-+-+-+-+-+  
    |L| | | |L| LASTMASK=$(( TOPBIT|1 ))
    +-+-+-+-+-+ $(( 2#10001 )) 
    | | | | | | $ 17
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
  BOUND1=1; 
  BOUND2=size-2;
  while (( BOUND1>0 && BOUND2<size-1 && BOUND1<BOUND2 ));do
    if (( BOUND1<BOUND2 ));then
      bit=$(( 1<<BOUND1 ));
      board[0]="$bit";          # Qを配置
      symmetry_backTrack "1" "$(( bit<<1 ))" "$bit" "$(( bit>>1 ))" "0";
    fi 
    (( BOUND1++,BOUND2-- ));
    ENDBIT=$(( ENDBIT>>1 ));
    : '
      +-+-+-+-+-+  
      |T|E| | | | ENDBIT=$(( TOPBIT>>1 ))
      +-+-+-+-+-+ $(( 2#01000 )) 
      | | | | | | $ 8
      +-+-+-+-+-+  
      | | | | | |       
      +-+-+-+-+-+             
      | | | | | | 
      +-+-+-+-+-+ 
      | | | | | |      
      +-+-+-+-+-+  
           ↓
      +-+-+-+-+-+  
      |T| |E| | | ENDBIT=$(( ENDBIT>>1 ))
      +-+-+-+-+-+ $(( 2#00100 )) 
      | | | | | | $ 4
      +-+-+-+-+-+  
      | | | | | |       
      +-+-+-+-+-+             
      | | | | | | 
      +-+-+-+-+-+ 
      | | | | | |      
      +-+-+-+-+-+  
    ';
    LASTMASK=$(( LASTMASK<<1 | LASTMASK | LASTMASK>>1 )) ;
  : '
    +-+-+-+-+-+  
    |L|L| |L|L| LASTMASK=$(( LASTMASK<<1 | LASTMASK | LASTMASK>>1 ))
    +-+-+-+-+-+ $(( 2#11011 )) 
    | | | | | | $ 27
    +-+-+-+-+-+  
    | | | | | |       
    +-+-+-+-+-+             
    | | | | | | 
    +-+-+-+-+-+ 
    | | | | | |      
    +-+-+-+-+-+  
  ';
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
  local -i flag=0;
  for ((i=0;i<row;++i)){
    if (( board[i]>=board[row] ));then
      val=$(( board[i]-board[row] ));
    else
      val=$(( board[row]-board[i] ));
    fi
    if (( board[i]==board[row] || val==(row-i) ));then
      flag=0;
      return ;
    fi
  }
  flag=1;
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
      #echo `$(($1-$2)) | sed -e "s/^-//g"`;
      if (( board[i]>=board[r] ));then
        val=$(( board[i]-board[r] ));
      else
        val=$(( board[r]-board[i] ));
      fi

      if (( board[i]==board[r] || val==(r-i) ));then
        flag=0; 
        return ;
      fi
    }
  }
  flag=1;
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
    #for(( col=0;col<(size-row);col++ )){
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
  local -i max=15;
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
      n|N) 非再帰 ※未実装
      " select;
      echo; 
      case "$select" in
        y|Y) NQ symmetry; break; ;;
        n|N) NQ symmetry; break; ;;
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

