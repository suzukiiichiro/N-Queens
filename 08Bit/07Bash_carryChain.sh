#!/usr/bin/bash

# :'
#  ## bash版
#  <> 07Bash_carryChain.sh
#  N:        Total       Unique        hh:mm:ss
#  4:            2            1         0:00:00
#  5:           10            2         0:00:00
#  6:            4            1         0:00:00
#  7:           40            6         0:00:01
#  8:           92           12         0:00:02
#  9:          352           46         0:00:12
# 10:          724           92         0:00:44
# 11:         2680          341         0:02:39
# 12:        14200         1788         0:08:35
# 13:        73712         9237         0:27:05
# 14:       365596        45771         1:30:40
# 15:      2279184       285095         5:59:03
# 
#  <> 06Bash_symmetry.sh 対象解除
#  N:        Total       Unique        hh:mm:ss
#  4:            2            1         0:00:00
#  5:           10            2         0:00:00
#  6:            4            1         0:00:00
#  7:           40            6         0:00:00
#  8:           92           12         0:00:00
#  9:          352           46         0:00:00
# 10:          724           92         0:00:02
# 11:         2680          341         0:00:05
# 12:        14200         1787         0:00:26
# 13:        73712         9233         0:02:28
# 14:       365596        45752         0:14:18
# 15:      2279184       285053         1:23:34
# ';
# 

declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -a pres_a;        # チェーン
declare -a pres_b;        # チェーン
declare -i COUNTER[3];    # カウンター 0:COUNT2 1:COUNT4 2:COUNT8
: 'B=(row     0:
      left    1:
      down    2:
      right   3:
      X[@]    4: 
      )';
declare -a B; 
declare -i DISPLAY=0;
#
#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecordCarryChain()
{
  local -a board=(${B[4]}); # 同じ場所の配置を許す
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
  # if (( !(down+1) ));then return 1; fi
  ((down+1))||return 1; # ↑を高速化
  while(( row&1 ));do
    # ((row>>=1));
    # ((left<<=1));
    # ((right>>=1));
    (( row>>=1,left<<=1,right>>=1 )); # 上記３行をまとめて書けます
  done
  (( row>>=1 ));      # １行下に移動する
  #
  local -i bitmap;  # 再帰に必要な変数は必ず定義する必要があります。
  local -i total=0; 
  #
  # 以下のwhileを一行のforにまとめると高速化が期待できます。
  # local -i bitmap=~(left|down|right);
  # while ((bitmap!=0));do
  # :
  # (( bitmap^=bit ))
  # done
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit));do
    local -i bit=$(( -bitmap&bitmap ));

    # ret=$( solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))")  ; 
    #  ret=$?;
    # [[ $ret -gt 0 ]] && { 
    # ((total+=$ret));
    # }  # solve()で実行したreturnの値は $? に入ります。
    # 上記はやや冗長なので以下の２行にまとめて書くことができます。
  solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))"; 
    ((total+=$?));  # solve()で実行したreturnの値は $? に入ります。
  done
  return $total;  # 合計を戻り値にします
}
#
: 'solve()を呼び出して再帰を開始する';
function process()
{
  local -i size="$1";
  local -i sym="$2"; # COUNT2 COUNT4 COUNT8
  # B[0]:row B[1]:left B[2]:down B[3]:right
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
  # if (( t_x[dimx]==dimy ));then
  #   return 1;
  # fi
  # 上記を以下のように書くことができます
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
  if (( t_x[0] ));then
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
    #if (( t_x[0]!=-1));then
    # 上記は if コマンドすら不要です
    [[ t_x[0] -ne -1 ]]&&{    # -ne は != と同じです
      (((dimx<t_x[0]||dimx>=size-t_x[0])
        &&(dimy==0||dimy==size-1)))&&{ return 0; } 
      (((dimx==size-1)&&((dimy<=t_x[0])||
          dimy>=size-t_x[0])))&&{ return 0; } 
    }
  else
    #if (( t_x[1]!=-1));then
    # 上記は if コマンドすら不要です
    [[ t_x[1] -ne -1 ]]&&{
      # bitmap=$(( bitmap|2 )); # 枝刈り
      # bitmap=$(( bitmap^2 )); # 枝刈り
      #((bitmap&=~2)); # 上２行を一行にまとめるとこうなります
      # ちなみに上と下は同じ趣旨
      # if (( (t_x[1]>=dimx)&&(dimy==1) ));then
      #   return 0;
      # fi
      (((t_x[1]>=dimx) && (dimy==1)))&&{ return 0; }
    }
  fi
  # B[0]:row B[1]:left B[2]:down B[3]:right
  (( (B[0] & 1<<dimx)|| (B[1] & 1<<(size-1-dimx+dimy))||
     (B[2] & 1<<dimy)|| (B[3] & 1<<(dimx+dimy)) )) && return 0;
  # ((B[0]|=1<<dimx));
  # ((B[1]|=1<<(size-1-dimx+dimy)));
  # ((B[2]|=1<<dimy));
  # ((B[3]|=1<<(dimx+dimy)));
  # 上記４行を一行にまとめることができます。
  ((B[0]|=1<<dimx, B[1]|=1<<(size-1-dimx+dimy),B[2]|=1<<dimy,B[3]|=1<<(dimx+dimy) ));
  #
  # 配列の中に配列があるので仕方がないですが要検討箇所です。
  t_x[$dimx]="$dimy"; 
  B[4]=${t_x[@]}; # Bに反映  
  #
  # ボードレイアウト出力
  # if [[ DISPLAY ]];then 
  #   board[$dimx]=$((1<<dimy)); 
  # fi
  # 上記を一行にまとめることができます。
  [[ $DISPLAY ]] && board[$dimx]=$((1<<dimy));
  #
  return 1;
}
#
: 'キャリーチェーン対称解除法';
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
  (( t_x[0] ))||{ # || は 条件が！であることを示します
    process "$size" "2";  #COUNT8
    #
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    ((DISPLAY==1))&& { printRecordCarryChain "$size" "0";
    read -p ""; }
    return;
  }
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
  ((s==w))&&{
    (( (n!=w)||(e!=w) ))&& return;
    process "$size" "0" # COUNT2
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    ((DISPLAY==1))&& { printRecordCarryChain "$size" "0";
    read -p ""; }
    return ;
  }
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  (( (e==w)&&(n>=s) ))&&{
    ((n>s))&& return ;
    process "$size" "1" # COUNT4
    # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
    ((DISPLAY==1))&& { printRecordCarryChain "$size" "0";
    read -p ""; }
    return ;
  }
  process "$size" "2" ; #COUNT8
  # ボードレイアウト出力 # 出力 1:bitmap版 0:それ以外
  ((DISPLAY==1))&& { printRecordCarryChain "$size" "0";
  read -p ""; }
  return ;
}
: 'チェーンのビルド';
function buildChain()
{
  local -i size="$1";
  local -a wB=sB=eB=nB=X; 
  wB=("${B[@]}");
  #
  # １ 上の２行に配置
  #
  #for ((w=0;w<=(size/2)*(size-3);w++));do
  # i++ よりも ++i のほうが断然高速です。
  for ((w=0;w<=(size/2)*(size-3);++w));do
    B=("${wB[@]}");
    # Bの初期化 #0:row 1:left 2:down 3:right 4:dimx
    #for ((bx_i=0;bx_i<size;bx_i++));do 
    #  X[$bx_i]=-1; 
    #  board[$bx_i]=-1;
    #done
    # i++ よりも ++i のほうが断然高速です。
    for ((bx_i=0;bx_i<size;++bx_i));do 
      X[$bx_i]=-1; 
      board[$bx_i]=-1;
    done
    B=([0]=0 [1]=0 [2]=0 [3]=0 [4]=${X[@]} [5]=${board[@]});
    placement "$size" "0" "$((pres_a[w]))"; # １　０行目と１行目にクイーンを配置
    [[ $? -eq 0 ]] && continue;
    placement "$size" "1" "$((pres_b[w]))";
    [[ $? -eq 0 ]] && continue;
    #
    # ２ 90度回転
    #
    nB=("${B[@]}");
    local -i mirror=$(( (size-2)*(size-1)-w ));
    #for ((n=w;n<mirror;n++));do 
    # i++ よりも ++i のほうが断然高速です。
    for ((n=w;n<mirror;++n));do 
      B=("${nB[@]}");
      placement "$size" "$((pres_a[n]))" "$((size-1))"; 
      [[ $? -eq 0 ]] && continue;
      placement "$size" "$((pres_b[n]))" "$((size-2))";
      [[ $? -eq 0 ]] && continue;
      #
      # ３ 90度回転
      #
      eB=("${B[@]}");
      #for ((e=w;e<mirror;e++));do 
      # i++ よりも ++i のほうが断然高速です。
      for ((e=w;e<mirror;++e));do 
        B=("${eB[@]}");
        placement "$size" "$((size-1))" "$((size-1-pres_a[e]))"; 
        [[ $? -eq 0 ]] && continue;
        placement "$size" "$((size-2))" "$((size-1-pres_b[e]))"; 
        [[ $? -eq 0 ]] && continue;
        #
        # ４ 90度回転
        #
        sB=("${B[@]}");
        #for ((s=w;s<mirror;s++));do
        # i++ よりも ++i のほうが断然高速です。
        for ((s=w;s<mirror;++s));do
          B=("${sB[@]}")
          placement "$size" "$((size-1-pres_a[s]))" "0";
          [[ $? -eq 0 ]] && continue;
          placement "$size" "$((size-1-pres_b[s]))" "1"; 
          [[ $? -eq 0 ]] && continue;
          #
          #  対象解除法
          carryChainSymmetry "$n" "$w" "$s" "$e" ; 
          #
        done
      done
    done
  done
}
: 'チェーンの初期化';
function initChain()
{
  local -i size="$1";
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
}
#
: 'チェーンの構築';
function carryChain()
{
  local -i size="$1";
  initChain "$size";  # チェーンの初期化
  buildChain "$size"; # チェーンのビルド
  # 集計
  UNIQUE=$(( COUNTER[0]+COUNTER[1]+COUNTER[2] ));
  TOTAL=$(( COUNTER[0]*2+COUNTER[1]*4+COUNTER[2]*8 ));
}
#
: 'Nを連続して実行';
function NQ()
{
  local selectName="$1";
  local -i min=4;
  local -i max=10;
  local -i N="$min";
  local startTime=endTime=hh=mm=ss=0; 
  echo " N:        Total       Unique        hh:mm:ss" ;
  local -i N;
  for((N=min;N<=max;N++)){
    TOTAL=UNIQUE=0;
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
#DISPLAY=0; # ボードレイアウト表示しない
DISPLAY=1; # ボードレイアウト表示する
#
NQ carryChain; 
exit;


