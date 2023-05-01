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
declare -a pres_a;
declare -a pres_b;
declare -a B_x;
#
: '';
function solve_nqueenr()
{
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  if (( (down+1)==0 ));then 
    return 1; 
  fi
  while (( (row&1)!=0 ));do
    ((row>>=1))
    ((left<<=1));
    ((right>>=1));
  done
  ((row>>=1));
  local -i bit=0;
  local -i bitmap=0;
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit ));do
    (( bit=bitmap&-bitmap ));
    #(( total+=solve_nqueenr "$row" "(left|bit)<<1" "(down|bit)" "(right|bit)>>1" )); 
    solve_nqueenr "$row" "(left|bit)<<1" "(down|bit)" "(right|bit)>>1"  ; 
    total+=$?;
  done
}
#
: 'クイーンの効きをチェック';
function board_placement()
{
  local -i dimx="$1";
  local -i dimy="$2";
  # read -p "dimx:$dimx dimy:$dimy";
  local -i flag=1;
  # echo "dimx:$dimx dimy:$dimy";
  if (( B_x[dimx]==dimy ));then # 同じ場所に置くかチェック
    # 同じ場所に置くのはOK
    flag=1;
    # return 1;
    # read -p "flag1";
  fi
  B_x[$dimx]="$dimy";             # dimxは行 dimyは列 
  row=$((1<<dimx));
  down=$((1<<dimy));
  left=$((1<<(size-1-dimx+dimy))); # size-1-x+yは右上から左下 
  right=$((1<<(dimx+dimy)));       # x+yは左上から右下
  if (( (B["row"]&row)||(B["down"]&down)||(B["left"]&left)||(B["right"]&right) ));then
    flag=0;
    return 0 ;
  fi 
  B["row"]=$((B["row"]|row));
  B["down"]=$((B["down"]|down));
  B["left"]=$((B["left"]|left));
  B["right"]=$((B["right"]|right));
  [[ $flag -eq 0 ]]
  return $?;
}
: 'キャリーチェーン';
function carryChain()
{
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
  # 90度回転
  # wB=( $B[@] );
  for i in ${!B[@]};do wB[$i]=${B[$i]} ; done

  # 1
  # 上２行にクイーンを置く
  # 上１行は２分の１だけ実行
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
    # 上２行　0行目,1行目にQを配置
    # ０行目にQを配置
    board_placement "0" "$((pres_a[w]))" ;
    # echo $?; # OK
    # １行目にQを配置
    board_placement "1" "$((pres_b[w]))" ;
    # echo $?; # OK
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
    # ２ 左２列にクイーンを置く
    local -i n;
    for ((n=w;n<(size-2)*(size-1)-w;n++));do 
      #B=( $nB[@] );
      for i in ${!nB[@]};do B[$i]=${nB[$i]}; done
      #
      # Qを配置
      board_placement "$((pres_a[n]))" "$((size-1))"; 
      if (( $?==0 ));then continue; fi
      board_placement "$((pres_b[n]))" "$((size-2))";
      if (( $?==0 ));then continue; fi
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
      # 3 下２行に置く
      local -i e;
      for ((e=w;e<(size-2)*(size-1)-w;e++));do 
        #B=( ${eB[@]} );
        for i in ${!eB[@]};do B[$i]=${eB[$i]}; done
        board_placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        if (( $?==0 ));then continue; fi
        board_placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        if (( $?==0 ));then continue; fi



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
        # 4 右２列に置く
        local -i s;
        for ((s=w;s<(size-2)*(size-1)-w;s++));do
          #
          # B=( ${sB[@]} );
          for i in ${!sB[@]};do B[$i]=${sB[$i]}; done
          #
          board_placement "$((size-1-pres_a[s]))" "0";
          if (( $?==0 ));then continue; fi
          board_placement "$((size-1-pres_b[s]))" "1"; 
          if (( $?==0 ));then continue; fi



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
            solve_nqueenr "$((B[row]>>2))" "$((B[left]>>4))" "$(( ((((B[down]>>2)|(~0<<($size-4)))+1)<<($size-5))-1 ))" "$(( (B[right]>>4)<<($size-5) ))";
            COUNT2+=$total; total=0;
            continue;
          fi
          # e==wは180度回転して同じ
          if (( (e==w)&&(n>=s) ));then
            # 180度回転して同じ時n>=sの時はsmaller?
            if((n>s));then continue; fi
            # この場合は4
            # 上下左右２行２列配置完了
            solve_nqueenr "B[row]>>2" "B[left]>>4" "((((B[down]>>2)|(~0<<($size-4)))+1)<<($size-5))-1" "(B[right]>>4)<<($size-5)";
            COUNT4+=$total;total=0;
            continue;
          fi
          # 上下左右２行２列配置完了"
          solve_nqueenr "B[row]>>2" "B[left]>>4" "((((B[down]>>2)|(~0<<($size-4)))+1)<<($size-5))-1" "(B[right]>>4)<<($size-5)";
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

