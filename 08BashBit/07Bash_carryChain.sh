#!/usr/bin/bash

declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=0;
declare -i COUNT8=0;
declare -i total=0;
# declare -A board=(
#   ["row"]="0" 
#   ["bh"]="0" 
#   ["bu"]="0" 
#   ["bd"]="0" 
#   ["left"]="0" 
#   ["down"]="0" 
#   ["right"]="0"
# );
declare -A B=(
  ["row"]="0" 
  ["left"]="0" 
  ["down"]="0" 
  ["right"]="0"
);
declare -a B_x;
#
: '';
function solve()
{
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
    bit=$(( bitmap&-bitmap ));
    bitmap=$(( bitmap^bit ));
    solve "$row" "(left|bit)<<1" "(down|bit)" "(right|bit)>>1"  ; 
    total+=$?;      # total はグローバル変数
  done
  return $total;    # 途中でクイーンを置くところがなくなるとここに来る
}
#
: 'クイーンの効きをチェック';
declare -i bpFlag=0; # 
function placement()
{
  local -i dimx="$1";
  local -i dimy="$2";
  local -i flag=0;       
  if (( B_x[$dimx]==$dimy ));then   # 同じ場所の配置を許す
    flag=1;
    return $?; 
  fi
  B_x[$dimx]="$dimy";               # dimxは行 dimyは列 
  row=$((1<<dimx));
  down=$((1<<dimy));
  left=$((1<<(size-1-dimx+dimy)));  #右上から左下 
  right=$((1<<(dimx+dimy)));        # x+yは左上から右下
  if (( (B[row]&row)||
        (B[down]&down)||
        (B[left]&left)||
        (B[right]&right) ));then
    flag=0;
    return $?;
  fi 
  B[row]=$((B[row]|row));
  B[down]=$((B[down]|down));
  B[left]=$((B[left]|left));
  B[right]=$((B[right]|right));
  flag=1;
  [[ $flag -eq 0 ]]
  return $?;
}
: 'キャリーチェーン';
function carryChain()
{
  local -a pres_a;
  local -a pres_b;
  : '
    N5
    0 0 0 1 1
    2 3 4 3 4
  ';
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
  #
  #
  #
  # 1 上２行にクイーンを置く 上１行は２分の１だけ実行
  #
  # 90度回転
  # wB=( $B[@] );
  for i in ${!B[@]};do wB[$i]=${B[$i]} ; done
  # q=7なら (7/2)*(7-4)=12
  # 1行目は0,1,2で,2行目0,1,2,3,4,5,6 で
  # 利き筋を置かないと13パターンになる
  local -i w=0;
  for ((w=0;w<=(size/2)*(size-3);w++));do
    #
    # B=( $wB[@] );
    for i in ${!wB[@]};do B[$i]=${wB[$i]} ; done
    #
    # B構造体の初期化
    B["row"]="0";
    B["down"]="0";
    B["left"]="0";
    B["right"]="0";
    #
    # board配列の初期化
    local -i i=0;
    for ((i=0;i<size;i++));do B_x[$i]=-1; done
    #
    # ０行目にQを配置
    local -i pna;
    : ' Cの結果
        pna:1
        pna:1
        pna:1
        pna:1
        pna:1 ';
    #
    # return で返却する場合
    placement "0" "$((pres_a[w]))";
    #echo "pna: $?";
    #
    # １行目にQを配置
    placement "1" "$((pres_b[w]))";
    # return で返却する場合
    #echo "pna: $?";
    : ' Cの結果
        pna:1
        pna:1
        pna:1
        pna:1
        pna:1 ';




    #
    # ２ 左２列にクイーンを置く
    #
    # 90度回転
    local -A nB=( 
      ["row"]="0"
      ["left"]="0"
      ["down"]="0"
      ["right"]="0"
    );
    #nB=( ${B[@]} );
    for i in ${!B[@]};do nB[$i]=${B[$i]}; done
    local -i n;
    for ((n=w;n<(size-2)*(size-1)-w;n++));do 
      #B=( $nB[@] );
      for i in ${!nB[@]};do B[$i]=${nB[$i]}; done
      #
      # Qを配置
      placement "$((pres_a[n]))" "$((size-1))"; 
      echo -n "$?";
      : 'Cの結果
      0000011000000000001100011000000000001000
         bashの結果
      0000010000000000001000000000000000000000
      ';
      if (( $?==0 ));then continue; fi
      # if (( bpFlag==0 ));then continue; fi
      placement "$((pres_b[n]))" "$((size-2))";
      if (( $?==0 ));then continue; fi
      # if (( bpFlag==0 ));then continue; fi



      #
      # 3 下２行に置く
      #
      # 90度回転
      local -A eB=( 
        ["row"]="0"
        ["left"]="0"
        ["down"]="0"
        ["right"]="0"
      );
      #eB=( ${B[@]} );
      for i in ${!B[@]};do eB[$i]=${B[$i]}; done
      local -i e;
      for ((e=w;e<(size-2)*(size-1)-w;e++));do 
        #B=( ${eB[@]} );
        for i in ${!eB[@]};do B[$i]=${eB[$i]}; done
        placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        if (( $?==0 ));then continue; fi
        # if (( bpFlag==0 ));then continue; fi
        placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        if (( $?==0 ));then continue; fi
        # if (( bpFlag==0 ));then continue; fi



        #
        # 4 右２列に置く
        #
        # 90度回転
        local -A sB=( 
          ["row"]="0"
          ["left"]="0"
          ["down"]="0"
          ["right"]="0"
        );
        #sB=( ${B[@]} );
        for i in ${!B[@]};do sB[$i]=${B[$i]}; done
        #
        local -i s;
        for ((s=w;s<(size-2)*(size-1)-w;s++));do
          #
          # B=( ${sB[@]} );
          for i in ${!sB[@]};do B[$i]=${sB[$i]}; done
          #
          placement "$((size-1-pres_a[s]))" "0";
          if (( $?==0 ));then continue; fi
          # if (( bpFlag==0 ));then continue; fi
          placement "$((size-1-pres_b[s]))" "1"; 
          if (( $?==0 ));then continue; fi
          # if (( bpFlag==0 ));then continue; fi



          #
          # 対象解除法
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
            solve "$((B[row]>>2))" "$((B[left]>>4))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
            COUNT2+=$?; total=0;
            continue;
          fi
          # e==wは180度回転して同じ
          if (( (e==w)&&(n>=s) ));then
            # 180度回転して同じ時n>=sの時はsmaller?
            if((n>s));then continue; fi
            # この場合は4
            # 上下左右２行２列配置完了
            solve "$((B[row]>>2))" "$((B[left]>>4))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
            COUNT4+=$total;total=0;
            continue;
          fi
          # 上下左右２行２列配置完了"
            solve "$((B[row]>>2))" "$((B[left]>>4))" "$(( ((((B[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1 ))" "$(( (B[right]>>4)<<(size-5) ))";
          COUNT8+=$total;total=0;
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
carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE";
echo "COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
exit;

