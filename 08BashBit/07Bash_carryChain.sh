#!/usr/bin/bash

declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=0;
declare -i COUNT8=0;
declare -a pres_a;
declare -a pres_b;
declare -A x;
declare -A B;
declare -i w;
declare -i n;
declare -i e;
declare -i s;
#
: 'ボード外側２列を除く内側のクイーン配置処理';
function solve()
{
  local -i total=0;
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  # 配置完了の確認 # bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if (( (down+1)==0 ));then return 1; fi
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
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit ));do
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
        "$(( (((B[down]>>2 | \
        ~0<<size-4)+1)<<size-5)-1 ))" \
        "$(( B[right]>>4<<size-5 ))";
  return $?;
}
#
: 'クイーンの効きをチェック';
function placement()
{
  # dimxは行 dimyは列
  local -i dimx="$1";
  local -i dimy="$2";
  # 同じ場所の配置を許す
  local t_x=(${B[x]});
  if (( t_x[$dimx]=="$dimy" ));then return 1; fi
  t_x[$dimx]="$dimy"
  B[x]=${t_x[@]}  
  local -i _row=_down=_left=_right=0;
  ((_row=1<<dimx));
  ((_down=1<<dimy));
  ((_left=1<<(size-1-dimx+dimy))); #右上から左下 
  ((_right=1<<(dimx+dimy)));     # x+yは左上から右下
  if (( (B[row] & _row)||
        (B[down] & _down)||
        (B[left] & _left)||
        (B[right] & _right) ));then
    return 0;
  fi 
  ((B[row]|=_row));
  ((B[down]|=_down));
  ((B[left]|=_left));
  ((B[right]|=_right));
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
  if (( (s==ww)&&(n<(w2-e)) ));then return; fi
  # 垂直方向の中心に対する反転が小さいかを確認
  if (( (e==ww)&&(n>(w2-n)) ));then return; fi
  # 斜め下方向への反転が小さいかをチェックする
  if (( (n==ww)&&(e>(w2-s)) ));then return; fi
  # n,e,s==w の場合は最小値を確認する。
  if ((s==w));then
    # : '右回転で同じ場合は、
    # w=n=e=sでなければ値が小さいのでskip
    # w=n=e=sであれば90度回転で同じ可能性 ';
    if(( (n!=w)||(e!=w) ));then return; fi
    solveQueen;
    COUNT2+=$?; 
    return ;
  fi
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  if (( (e==w)&&(n>=s) ));then
    if((n>s));then return;  fi
    solveQueen;
    COUNT4+=$?;
    return ;
  fi
  solveQueen;
  COUNT8+=$?;
  return ;
}
#
: 'チェーンの構築';
function buildChain()
{
  # Bの初期化
  B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${x[@]});
  #
  # １ 上２行にクイーンを置く 
  #    上１行は２分の１だけ実行 90度回転 ';
  #
  local -A wB;
  # wB=( $B[@] );
  for key_B in ${!B[@]};do wB["$key_B"]="${B[$key_B]}" ; done
  # q=7なら (7/2)*(7-4)=12
  # 1行目は0,1,2で,2行目0,1,2,3,4,5,6 で
  # 利き筋を置かないと13パターンになる
  for ((w=0;w<=(size/2)*(size-3);w++));do
    #B=wB;
    # B=( $wB[@] );
    for key_wB in ${!wB[@]};do B["$key_wB"]="${wB[$key_wB]}" ; done
    # B構造体の初期化
    for ((bx_i=0;bx_i<size;bx_i++));do x[$bx_i]=-1; done
    B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${x[@]});
    #
    # １　０行目と１行目にクイーンを配置
    # 
    placement "0" "$((pres_a[w]))";
    placement "1" "$((pres_b[w]))";
    #
    # ２ 90度回転
    #
    local -A nB;
    # nB=( ${B[@]} );
    for key_B in "${!B[@]}";do nB["$key_B"]="${B[$key_B]}"; done
    local -i mirror=$(( (size-2)*(size-1)-w ));
    for ((n=w;n<mirror;n++));do 
      # B=( $nB[@] );
      for key_nB in ${!nB[@]};do B["$key_nB"]="${nB[$key_nB]}"; done
      placement "$((pres_a[n]))" "$((size-1))"; 
      if (( $?==0 ));then continue; fi
      placement "$((pres_b[n]))" "$((size-2))";
      if (( $?==0 ));then continue; fi
      #
      # ３ 90度回転
      #
      local -A eB;
      # eB=( ${B[@]} );
      for key_B in ${!B[@]};do eB["$key_B"]="${B[$key_B]}"; done
      for ((e=w;e<mirror;e++));do 
        #B=( ${eB[@]} );
        for key_eB in ${!eB[@]};do B["$key_eB"]="${eB[$key_eB]}"; done
        placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        if (( $?==0 ));then continue; fi
        placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        if (( $?==0 ));then continue; fi
        #
        # ４ 90度回転
        #
        local -A sB;
        # sB=( ${B[@]} );
        for key_B in ${!B[@]};do sB["$key_B"]="${B[$key_B]}"; done
        for ((s=w;s<mirror;s++));do
          # B=( ${sB[@]} );
          for key_sB in ${!sB[@]};do B["$key_sB"]="${sB[$key_sB]}"; done
          placement "$((size-1-pres_a[s]))" "0";
          if (( $?==0 ));then continue; fi
          placement "$((size-1-pres_b[s]))" "1"; 
          if (( $?==0 ));then continue; fi
          #
          #  対象解除法
          #
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
  local -i a=0;
  local -i b=0;
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
