"""
CentOS-5.1$ pypy 18Python_carryChain_class.py
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            0         0:00:00.002
 6:            4            0         0:00:00.009
 7:           40            0         0:00:00.026
 8:           92            0         0:00:00.080
 9:          352            0         0:00:00.153
10:          724            0         0:00:00.404
11:         2680            0         0:00:01.042
12:        14200            0         0:00:02.831
13:        73712            0         0:00:07.429
14:       365596            0         0:00:19.511
15:      2279184            0         0:00:48.769

CentOS-5.1$ pypy 15Python_NodeLayer_symmetoryOps_class.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:05.425

CentOS-5.1$ pypy 14Python_NodeLayer_symmetoryOps_param.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.345

CentOS-5.1$ pypy 13Python_NodeLayer_mirror_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:02.926

CentOS-5.1$ pypy 11Python_NodeLayer.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.160

CentOS-5.1$ pypy 10Python_bit_symmetry_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:01.998

CentOS-5.1$ pypy 09Python_bit_symmetry_ThreadPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:02.111

CentOS-5.1$ pypy 08Python_bit_symmetry.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:03.026

CentOS-5.1$ pypy 07Python_bit_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.274

CentOS-5.1$ pypy 06Python_bit_backTrack.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:12.610

CentOS-5.1$ pypy 05Python_optimize.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:14.413

CentOS-5.1$ pypy 04Python_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:46.629

CentOS-5.1$ pypy 03Python_backTracking.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184            0         0:00:44.993
"""
from datetime import datetime
import copy
# pypyを使うときは以下を活かしてcodon部分をコメントアウト
import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# pypy では ThreadPool/ProcessPoolが動きます 
#

# from threading import Thread
# from multiprocessing import Pool as ThreadPool
# import concurrent
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
#

class NQueens17:
  def __init__(self):
    pass
  #
  # ボード外側２列を除く内側のクイーン配置処理
  def solve(self,row:int,left:int,down:int,right:int):
    total:int=0
    if not down+1:
      return 1
    while row&1:
      row>>=1
      left<<=1
      right>>=1
    row>>=1           # １行下に移動する
    bitmap:int=~(left|down|right)
    while bitmap!=0:
      bit=-bitmap&bitmap
      total+=self.solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
      bitmap^=bit
    return total
  #
  # キャリーチェーン　solve()を呼び出して再起を開始する
  def process(self,size:int,sym:int,B:list[int])->int:
    # sym 0:COUNT2 1:COUNT4 2:COUNT8
    return sym*self.solve(
        B[0]>>2,
        B[1]>>4,
        (((B[2]>>2|~0<<size-4)+1)<<size-5)-1,
        B[3]>>4<<size-5
    )
  #
  # キャリーチェーン　対象解除
  def carryChainSymmetry(self,size:int,n:int,w:int,s:int,e:int,B:list[int],B4:list[int])->int:
    # n,e,s=(N-2)*(N-1)-1-w の場合は最小値を確認する。
    ww:int=(size-2)*(size-1)-1-w
    w2:int=(size-2)*(size-1)-1
    # 対角線上の反転が小さいかどうか確認する
    if s==ww and n<(w2-e): return 0
    # 垂直方向の中心に対する反転が小さいかを確認
    if e==ww and n>(w2-n): return 0
    # 斜め下方向への反転が小さいかをチェックする
    if n==ww and e>(w2-s): return 0
    # 【枝刈り】１行目が角の場合
    # １．回転対称チェックせずにCOUNT8にする
    if not B4[0]:      
      return self.process(size,8,B) # COUNT8
    # n,e,s==w の場合は最小値を確認する。
    # : '右回転で同じ場合は、
    # w=n=e=sでなければ値が小さいのでskip
    # w=n=e=sであれば90度回転で同じ可能性 ';
    if s==w:
      if n!=w or e!=w: return 0      
      return self.process(size,2,B) # COUNT2
    # : 'e==wは180度回転して同じ
    # 180度回転して同じ時n>=sの時はsmaller?  ';
    if e==w and n>=s:
      if n>s: return 0
      return self.process(size,4,B) # COUNT4    
    return self.process(size,8,B)   # COUNT8
  #
  # キャリーチェーン 効きのチェック dimxは行 dimyは列
  def placement(self,size:int,dimx:int,dimy:int,B:list[int],B4:list[int]):
    if B4[dimx]==dimy:
      return 1
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
    if B4[0]:
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
      if B4[0]!=-1:
        #if((dimx<B[4][0] or dimx>=size-B[4][0]) and (dimy==0 or dimy==size-1)):
        #  return 0
        if ((dimx < B4[0] or dimx >= size - B4[0]) and (dimy == 0 or dimy == size - 1)):
          return 0
        if ((dimx == size - 1) and (dimy <= B4[0] or dimy >= size - B4[0])):
          return 0
    else:
      if B4[1]!=-1:
      # bitmap=$(( bitmap|2 )); # 枝刈り
      # bitmap=$(( bitmap^2 )); # 枝刈り
      #((bitmap&=~2)); # 上２行を一行にまとめるとこうなります
      # ちなみに上と下は同じ趣旨
      # if (( (t_x[1]>=dimx)&&(dimy==1) ));then
      #   return 0;
      # fi
        if B4[1]>=dimx and dimy==1:
          return 0
    if ((B[0] & (1 << dimx)) or 
    (B[1] & (1 << (size - 1 - dimx + dimy))) or
    (B[2] & (1 << dimy)) or
    (B[3] & (1 << (dimx + dimy)))):
      return 0
    B[0]|=1<<dimx
    B[1]|=1<<(size-1-dimx+dimy)
    B[2]|=1<<dimy
    B[3]|=1<<(dimx+dimy)
    B4[dimx]=dimy
    return 1
  #
  # チェーンのビルド
  def buildChain(self,size:int,pres_a:list[int],pres_b:list[int])->int:
    total:int=0
    B:list[int]=[0,0,0,0]
    B4:list[int]=[-1]*size  # Bの初期化
    for w in range((size//2)*(size-3)+1):
      wB=copy.deepcopy(B)
      wB4=copy.deepcopy(B4)
      # １．０行目と１行目にクイーンを配置
      if self.placement(size,0,pres_a[w],wB,wB4)==0:
        continue
      if self.placement(size,1,pres_b[w],wB,wB4)==0:
        continue
      # ２．９０度回転
      mirror=(size-2)*(size-1)-w
      for n in range(w,mirror,1):
        nB=copy.deepcopy(wB)
        nB4=copy.deepcopy(wB4)
        if self.placement(size,pres_a[n],size-1,nB,nB4)==0:
          continue
        if self.placement(size,pres_b[n],size-2,nB,nB4)==0:
          continue
        # ３．９０度回転
        for e in range(w,mirror,1):
          eB=copy.deepcopy(nB)
          eB4=copy.deepcopy(nB4)
          if self.placement(size,size-1,size-1-pres_a[e],eB,eB4)==0:
            continue
          if self.placement(size,size-2,size-1-pres_b[e],eB,eB4)==0:
            continue
          # ４．９０度回転
          for s in range(w,mirror,1):
            sB=copy.deepcopy(eB)
            sB4=copy.deepcopy(eB4)
            if self.placement(size,size-1-pres_a[s],0,sB,sB4)==0:
              continue
            if self.placement(size,size-1-pres_b[s],1,sB,sB4)==0:
              continue
            # 対象解除法
            total+=self.carryChainSymmetry(size,n,w,s,e,sB,sB4)
    return total
  #
  # チェーンの初期化
  def initChain(self,size:int,pres_a:list[int],pres_b:list[int]):
    idx=0
    for a in range(size):
      for b in range(size):
        if (a>=b and (a-b)<=1) or (b>a and (b-a<=1)):
          continue
        pres_a[idx]=a
        pres_b[idx]=b
        idx+=1
  #
  # キャリーチェーン
  def carryChain(self,size:int)->int:
    pres_a:list[int]=[0]*930
    pres_b:list[int]=[0]*930
    # Bの初期化  [0, 0, 0, 0, [0, 0, 0, 0, 0]]
    #B=[0]*5             # row/left/down/right/X
    #B[4]=[-1]*size       # X を0でsize分を初期化
    self.initChain(size,pres_a,pres_b)     # チェーンの初期化
    return self.buildChain(size,pres_a,pres_b)    # チェーンのビルド
#
# 実行
#
class NQueens17_carryCain:
  def main(self)->None:
    nmin:int=5
    nmax:int=16
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens17()
      total:int=NQ.carryChain(size)  
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]  
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

# メイン実行部分
if __name__=="__main__":
    NQueens17_carryChain().main()
