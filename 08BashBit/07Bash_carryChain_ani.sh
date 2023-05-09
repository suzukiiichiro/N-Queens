#!/usr/bin/bash

declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=0;
declare -i COUNT8=0;
declare -A x;
declare -A B=(
  ["row"]="0" 
  ["left"]="0" 
  ["down"]="0" 
  ["right"]="0"
  ["x"]=${x[@]}
);
#
: '';
function solve()
{
  local -i total=0;
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  # Placement Complete?
  # bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if (( (down+1)==0 ));then 
    return 1; 
  fi
  # at least one more queen to place
  # Column is covered by pre-placement
  # row 右端にクイーンがすでに置かれていたら
  # クイーンを置かずに１行下に移動する
  # rowを右端から１ビットずつ削っていく。
  # ここではrowはすでにクイーンが置かれているか
  # どうかだけで使う
  while (( (row&1)!=0 ));do
    ((row>>=1))     # 右に１ビットシフト
    ((left<<=1));   # left 左に１ビットシフト
    ((right>>=1));  # right 右に１ビットシフト
  done
  ((row>>=1));      # １行下に移動する
  # Column needs to be placed
  local -i bit=0;
  local -i bitmap=$(( ~(left|down|right) ));
  while (( bitmap!=0 ));do
    ((bit=bitmap&-bitmap ));
    ((bitmap^=bit ));
    solve "$row" "$(( (left|bit)<<1 ))" "$(( (down|bit) ))" "$(( (right|bit)>>1 ))"  ; 
    total+=$?;      # total はグローバル変数
  done
  return $total;
}
#
: 'クイーンの効きをチェック';
function placement()
{
  local -i dimx="$1";
  local -i dimy="$2";
  local -i _row=_down=_left=_right=0;
  local t_x=(${B[x]});
  if (( t_x[$dimx]=="$dimy" ));then   # 同じ場所の配置を許す
 
    return 1;
  fi
  t_x[$dimx]="$dimy"
  B[x]=${t_x[@]}
  ((_row=1<<dimx));
  ((_down=1<<dimy));
  ((_left=1<<(size-1-dimx+dimy)));    #右上から左下 
  ((_right=1<<(dimx+dimy)));          # x+yは左上から右下
  if (( (B[row] & $_row)||
        (B[down] & $_down)||
        (B[left] & $_left)||
        (B[right] & $_right) ));then
    return 0;
  fi 
  ((B[row]|=_row));
  ((B[down]|=_down));
  ((B[left]|=_left));
  ((B[right]|=_right));
  return 1;
}
: 'キャリーチェーン';
function carryChain()
{
  local -a pres_a;
  local -a pres_b;
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
  # Bの初期化
  B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${x[@]});
  #
  #
  #
  # 1 
  # 上２行にクイーンを置く 
  # 上１行は２分の１だけ実行 90度回転
  #
  #
  #
  local -A wB;
  # wB=( $B[@] );
  for key_B in ${!B[@]};do 
    wB["$key_B"]="${B[$key_B]}" ; 
  done
  : '
    q=7なら (7/2)*(7-4)=12
    1行目は0,1,2で,2行目0,1,2,3,4,5,6 で
    利き筋を置かないと13パターンになる
  ';
  for ((w=0;w<=(size/2)*(size-3);w++));do
    # echo "w:$w 上";
    # B=( $wB[@] );
    for key_wB in ${!wB[@]};do 
      B["$key_wB"]="${wB[$key_wB]}" ; 
    done
    #
    # B構造体の初期化
    for ((bx_i=0;bx_i<size;bx_i++));do x[$bx_i]=-1; done
    B=(["row"]="0" ["down"]="0" ["left"]="0" ["right"]="0" ["x"]=${x[@]});
    # board配列の初期化
    for ((bx_i=0;bx_i<size;bx_i++));do B_x[$bx_i]=-1; done
    # ０行目にQを配置 
    placement "0" "$((pres_a[w]))";
    # １行目にQを配置
    placement "1" "$((pres_b[w]))";
    #
    #
    #
    # ２ 
    # 左２列にクイーンを置く 90度回転
    #
    #
    # nBの初期化
    local -A nB;
    local -A nB_x;
    #nB=( ${B[@]} );
    for key_B in "${!B[@]}";do 
      nB["$key_B"]="${B[$key_B]}"; 
    done
    for ((n=w;n<(size-2)*(size-1)-w;n++));do 
      # echo "n:$n 左";
      #B=( $nB[@] );
      for key_nB in ${!nB[@]};do 
        B["$key_nB"]="${nB[$key_nB]}"; 
      done
      placement "$((pres_a[n]))" "$((size-1))"; 
      ret="$?"
      # echo -n "$ret";
      : 'Cの結果
      0000011000000000001100011000000000001000
         bashの結果
      0000011000000000001100011000000000001
      ';
      if (( $ret==0 ));then continue; fi
      placement "$((pres_b[n]))" "$((size-2))";
      ret="$?"
      if (( $ret==0 ));then continue; fi
      #
      #
      #
      # ３ 
      # 下２行に置く 90度回転
      #
      #
      # eBの初期化
      local -A eB;
      local  -A eB_x;
      #eB=( ${B[@]} );
      for key_B in ${!B[@]};do 
        eB["$key_B"]="${B[$key_B]}"; 
      done
      for ((e=w;e<(size-2)*(size-1)-w;e++));do 
        # echo "e:$e 下";
        #B=( ${eB[@]} );
        for key_eB in ${!eB[@]};do 
          B["$key_eB"]="${eB[$key_eB]}"; 
        done
        placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        ret="$?";
        if (( $ret==0 ));then continue; fi
        placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        ret="$?";
        if (( $ret==0 ));then continue; fi
        #
        #
        # ４ 
        # 右２列に置く 90度回転
        #
        #
        local -A sB;
        local  -A sB_x;
        #sB=( ${B[@]} );
        for key_B in ${!B[@]};do 
          sB["$key_B"]="${B[$key_B]}"; 
        done
        for ((s=w;s<(size-2)*(size-1)-w;s++));do
          # echo "s:$s 右";
          # B=( ${sB[@]} );
          for key_sB in ${!sB[@]};do 
            B["$key_sB"]="${sB[$key_sB]}"; 
          done
          placement "$((size-1-pres_a[s]))" "0";
          ret="$?";
          if (( $ret==0 ));then continue; fi
          placement "$((size-1-pres_b[s]))" "1"; 
          ret="$?";
          if (( $ret==0 ));then continue; fi
          #
          #
          # 対象解除法
          #
          #
          # Check for minimum if n, e, s = (N-2)*(N-1)-1-w
          local -i ww=$(( (size-2)*(size-1)-1-w ));
          local -i w2=$(( (size-2)*(size-1)-1 ));
          # check if flip about the up diagonal is smaller
          if (( (s==ww)&&(n<(w2-e)) ));then continue; fi
          # check if flip about the vertical center is smaller
          if (( (e==ww)&&(n>(w2-n)) ));then continue; fi
          # check if flip about the down diagonal is smaller
          if (( (n==ww)&&(e>(w2-s)) ));then continue; fi
          # Check for minimum if n, e, s = w
          if ((s==w));then
            # right rotation is smaller unless  w = n = e = s
            # 右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
            if(( (n!=w)||(e!=w) ));then continue; fi
            # 上下左右２行２列配置完了
            # w=n=e=sであれば90度回転で同じ可能性
            # この場合はミラーの2
            solve "$(( B[row]>>2 ))" "$(( B[left]>>4 ))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
            COUNT2+=$?; 
            continue;
          fi
          # e==wは180度回転して同じ
          if (( (e==w)&&(n>=s) ));then
            # 180度回転して同じ時n>=sの時はsmaller?
            if((n>s));then continue; fi
            # この場合は4
            # 上下左右２行２列配置完了
            solve "$(( B[row]>>2 ))" "$(( B[left]>>4 ))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
            COUNT4+=$?;
            continue;
          fi
          # 上下左右２行２列配置完了"
          solve "$(( B[row]>>2 ))" "$(( B[left]>>4 ))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
          COUNT8+=$?;
          continue;
        done
      done
    done
  done
  UNIQUE=$(($COUNT2+$COUNT4+$COUNT8));
  TOTAL=$(($COUNT2*2+COUNT4*4+COUNT8*8));
}
#
size=5;
time carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE";
echo "COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
size=8;
time carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE";
echo "COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
exit;

