#!/usr/bin/bash

declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=0;
declare -i COUNT8=0;
declare -a B; 
: 'B=(row     0:
      left    1:
      down    2:
      right   3:
      X[@]    4: 
      )';
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
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit));do
    (( bit=-bitmap&bitmap ));
    solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))"  ; 
    (( total+=$? ));
  done
  return $total;
}
#
: 'solve()を呼び出して再帰を開始する';
function solveQueen()
{
  local -i size="$1";
  solve "$(( B[0]>>2 ))" \
        "$(( B[1]>>4 ))" \
        "$(( (((B[2]>>2 | ~0<<size-4)+1)<<size-5)-1 ))" \
        "$(( B[3]>>4<<size-5 ))";
  return $?;
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
  #
  if (( t_x[0]==0 ));then
    if (( t_x[1]!=-1));then
      # bitmap=$(( bitmap|2 ));
      # bitmap=$(( bitmap^2 ));
      # 上と下は同じ趣旨
      (((t_x[1]>=dimx) && (dimy==1)))&&{ return 0; }
    fi
  else
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
    solveQueen "$size";
    [[ $? -eq 0 ]] || COUNT8+=$?; 
    return;
  }
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
  ((s==w))&&{
    (( (n!=w)||(e!=w) ))&& return;
    solveQueen "$size";
    [[ $? -eq 0 ]] || COUNT2+=$?; 
    return ;
  }
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  (( (e==w)&&(n>=s) ))&&{
    ((n>s))&& return ;
    solveQueen "$size";
    [[ $? -eq 0 ]] || COUNT4+=$?; 
    return ;
  }
  solveQueen "$size";
  [[ $? -eq 0 ]] || COUNT8+=$?; 
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
  UNIQUE=$(($COUNT2+$COUNT4+$COUNT8));
  TOTAL=$(($COUNT2*2+COUNT4*4+COUNT8*8));
}
#
# 実行
size=8;
DISPLAY=0;
time carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
exit;
