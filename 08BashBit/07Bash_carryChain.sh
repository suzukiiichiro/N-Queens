#!/usr/bin/bash

: '

';
declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=0;
declare -i COUNT8=0;
declare -a pres_a;
declare -a pres_b;
declare -A B; # B=(row left down right X[@])
declare -A X; # dimx=(0 0 0 0 0)
declare -i n=w=s=e=0;
#
: 'ボード外側２列を除く内側のクイーン配置処理';
function solve()
{
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  # 配置完了の確認 # bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  (( (down+1)==0 ))&& return 1;
  # 新たなQを配置 colは置き換える。row 右端にクイーンがすでに置かれていたらクイー
  # ンを置かずに１行下に移動する。rowを右端から１ビットずつ削っていく。ここでは
  # rowはすでにクイーンが置かれているかどうかだけで使う
  while(( (row&1)!=0 ));do
    (( row>>=1 ))     # 右に１ビットシフト
    (( left<<=1 ));   # left 左に１ビットシフト
    (( right>>=1 ));  # right 右に１ビットシフト
  done
  (( row>>=1 ));      # １行下に移動する
  local -i bit=0;
  local -i bitmap=0;
  local -i total=0;
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit));do
    (( bit=bitmap&-bitmap ));
    solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))"  ; 
    total+=$?;
  done
  return $total;
}
#
: 'solve()を呼び出して再帰を開始する';
function solveQueen()
{
  solve "$(( B[row]>>2 ))" \
        "$(( B[left]>>4 ))" \
        "$(( (((B[down]>>2 | ~0<<size-4)+1)<<size-5)-1 ))" \
        "$(( B[right]>>4<<size-5 ))";
  return $?;
}
#
: 'クイーンの効きをチェック';
function placement()
{
  local -i dimx="$1";     # dimxは行 dimyは列
  local -i dimy="$2";
  local -a t_x=(${B[x]}); # 同じ場所の配置を許す
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
  if (( t_x[0]==0 ));then
    if (( t_x[1]!=-1));then
      # bitmap=$(( bitmap|2 ));
      # bitmap=$(( bitmap^2 ));
      # 上と下は同じ趣旨
      (((t_x[1]>=dimx) && (dimy==1)))&&{ return 0; }
    fi
  else
  # 【枝刈り】Qが角にない場合
  #   １．上部サイド枝刈り
  #  if ((row<BOUND1));then        
  #    bitmap=$(( bitmap|SIDEMASK ));
  #    bitmap=$(( bitmap^=SIDEMASK ));
  #
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
  #
  t_x[$dimx]="$dimy" B[x]=${t_x[@]}; # Bに反映  
  if (( (B[row] & 1<<dimx)||
        (B[down] & 1<<dimy)||
        (B[left] & 1<<(size-1-dimx+dimy))||
        (B[right] & 1<<(dimx+dimy)) ));then
    return 0;
  fi 
  ((B[row]|=1<<dimx));
  ((B[down]|=1<<dimy));
  ((B[left]|=1<<(size-1-dimx+dimy)));
  ((B[right]|=1<<(dimx+dimy)));
  return 1;
}
#
: 'キャリーチェーン対象解除法';
function carryChainSymmetry()
{
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
  local -a t_x=(${B[x]}); # 同じ場所の配置を許す
  (( t_x[0]==0 ))&&{
    solveQueen;
    COUNT8+=$?; 
    return;
  }
  #
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
  ((s==w))&&{
    (( (n!=w)||(e!=w) ))&& return;
    solveQueen;
    COUNT2+=$?; 
    return ;
  }
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  (( (e==w)&&(n>=s) ))&&{
    ((n>s))&& return ;
    solveQueen;
    COUNT4+=$?;
    return ;
  }
  solveQueen;
  COUNT8+=$?;
  return ;
}
#
: 'チェーンの構築';
function buildChain()
{
  # Bの初期化
  B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${X[@]});
  #
  # １ 上２行にクイーンを置く 上１行は２分の１だけ実行 90度回転
  # wB=( $B[@] );
  local -A wB=B; # bashの連想配列は↓が必要
  for key_B in ${!B[@]};do wB["$key_B"]="${B[$key_B]}" ; done
  for ((w=0;w<=(size/2)*(size-3);w++));do
    # B=( $wB[@] );
    B=wB;  # bashの連想配列は↓が必要
    for key_wB in ${!wB[@]};do B["$key_wB"]="${wB[$key_wB]}" ; done
    for ((bx_i=0;bx_i<size;bx_i++));do X[$bx_i]=-1; done
    B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${X[@]});
    placement "0" "$((pres_a[w]))"; # １　０行目と１行目にクイーンを配置
    [[ $? -eq 0 ]] && continue;
    placement "1" "$((pres_b[w]))";
    [[ $? -eq 0 ]] && continue;
    #
    # ２ 90度回転
    # nB=( ${B[@]} );
    local -A nB=B;  # bashの連想配列は↓が必要
    for key_B in "${!B[@]}";do nB["$key_B"]="${B[$key_B]}"; done
    local -i mirror=$(( (size-2)*(size-1)-w ));
    for ((n=w;n<mirror;n++));do 
      # B=( $nB[@] );
      B=nB;  # bashの連想配列は↓が必要
      for key_nB in ${!nB[@]};do B["$key_nB"]="${nB[$key_nB]}"; done
      placement "$((pres_a[n]))" "$((size-1))"; 
      [[ $? -eq 0 ]] && continue;
      placement "$((pres_b[n]))" "$((size-2))";
      [[ $? -eq 0 ]] && continue;
      #
      # ３ 90度回転
      # eB=( ${B[@]} );
      local -A eB=B;  # bashの連想配列は↓が必要
      for key_B in ${!B[@]};do eB["$key_B"]="${B[$key_B]}"; done
      for ((e=w;e<mirror;e++));do 
        #B=( ${eB[@]} );
        B=eB; # bashの連想配列は↓が必要
        for key_eB in ${!eB[@]};do B["$key_eB"]="${eB[$key_eB]}"; done
        placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        [[ $? -eq 0 ]] && continue;
        placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        [[ $? -eq 0 ]] && continue;
        #
        # ４ 90度回転
        #sB=( ${B[@]} );
        local -A sB=B; # bashの連想配列は↓が必要
        for key_B in ${!B[@]};do sB["$key_B"]="${B[$key_B]}"; done
        for ((s=w;s<mirror;s++));do
          #B=( ${sB[@]} );
          B=sB; # bashの連想配列は↓が必要
          for key_sB in ${!sB[@]};do B["$key_sB"]="${sB[$key_sB]}"; done
          placement "$((size-1-pres_a[s]))" "0";
          [[ $? -eq 0 ]] && continue;
          placement "$((size-1-pres_b[s]))" "1"; 
          [[ $? -eq 0 ]] && continue;
          #
          #  対象解除法
          carryChainSymmetry; 
          continue;
        done
      done
    done
  done
}
#
: 'チェーンの初期化';
function initChain()
{
  local -i idx=0;
  local -i a=b=0;
  for ((a=0;a<size;a++));do
    for ((b=0;b<size;b++));do
      if (( ( (a>=b)&&((a-b)<=1) )||
            ( (b>a)&& ((b-a)<=1) ) ));then
        continue;
      fi
      pres_a[$idx]=$a;
      pres_b[$idx]=$b;
      ((idx++));
    done
  done
}
#
: 'キャリーチェーン';
function carryChain()
{
  # チェーンの初期化
  initChain ;     
  # チェーンの構築 
  buildChain ;    
  # 集計
  UNIQUE=$(($COUNT2+$COUNT4+$COUNT8));
  TOTAL=$(($COUNT2*2+COUNT4*4+COUNT8*8));
}
#
# 実行
size=8;
time carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
exit;
