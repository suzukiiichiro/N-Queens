#!/bin/bash

##
 # Bash(シェルスクリプト)で学ぶアルゴリズムとデータ構造  
 # 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 #
 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 #
 # ステップバイステップでＮ−クイーン問題を最適化
 #  １．ブルートフォース（力まかせ探索） NQueen1()
 #  ２．配置フラグ（制約テスト高速化）   NQueen2()
 #<>３．バックトラック                   NQueen3()
 #  ４．ビットマップ                     NQueen4()
 #  ５．ユニーク解                       NQueen5()
##
 # ３．バックトラック
 #  　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 #  さない事が明らかな場合は、それ以降のパターン生成を行わない。
 #  　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 #  　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 #  は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 #  けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 #  を降りても解がないと判明した時点で木を引き返すということができます。
##
# N:        Total       Unique     hh:mm:ss
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            0            0
# 5:           10            0            0
# 6:            4            0            0
# 7:           40            0            0
# 8:           92            0            1
# 9:          352            0            1
#10:          724            0            7
#11:         2680            0           33
#12:        14200            0          183
##

T=1 ;
N-Queen3_rec(){
  local -i i="$1" j s=$2 ; # s:size
  for((j=0;j<s;j++)){
    [[ -z "${fa[j]}" ]] && [[ -z "${fb[i+j]}" ]] && [[ -z "${fc[i-j+s-1]}" ]] && {
      pos[$i]=$j ;
      ((i==(s-1)))&&((T++))||{
        fa[j]="true" ;
        fb[i+j]="true" ; 
        fc[i-j+s-1]="true" ; 
        N-Queen3_rec "$((i+1))" "$s" ; 
        fa[j]="" ;           
        fb[i+j]="" ;   
        fc[i-j+s-1]="" ; 
      }          
    }
  }  
}
N-Queen3(){
  # m: max mi:min s:size st:starttime t:time
  local -i m=12 mi=2 s=$mi st= t= ;
  echo " N:        Total       Unique        hh:mm" ;
  for((s=mi;s<=m;s++)){
    T=0 U=0 st=`date +%s` ;
    N-Queen3_rec 0 "$s";
    t=$((`date +%s` - st)) ;
    printf "%2d:%13d%13d%13d\n" $s $T $U $t ;
  } 
}

N-Queen3 		      # バックトラック
exit ;

