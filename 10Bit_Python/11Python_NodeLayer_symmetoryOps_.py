from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
# from threading import Thread
# from multiprocessing import Pool as ThreadPool
# import concurrent
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
# Codon環境の判定
# Codon用の型定義
step=0
class Local:
  TOTAL:int
  UNIQUE:int
  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  board:list
  def __init__(self,TOTAL:int,UNIQUE:int,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list):
    self.TOTAL,self.UNIQUE,self.TOPBIT,self.ENDBIT,self.LASTMASK,self.SIDEMASK,self.BOUND1,self.BOUND2,self.board=TOTAL,UNIQUE,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board
class NQueens21:
  def __init__(self):
    pass
  """ビットが1である数をカウント"""
  def count_bits_nodeLayer(self,n:int)->int:
    counter:int=0
    while n:
      n&=n-1
      counter+=1
    return counter
  """対称解除操作"""
  def symmetryOps(self,size:int,local:Local)->int:
    # 90度回転
    if local.board[local.BOUND2]==1:
      ptn:int=2
      own:int=1
      while own<size:
        bit:int=1
        you:int=size-1
        while you>=0 and local.board[you] != ptn and local.board[own] >= bit:
          bit<<=1
          you-=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn<<=1
        own+=1
      # 90度回転が同型
      if own>size-1:
        return 2
    # 180度回転
    if local.board[size-1]==local.ENDBIT:
      you:int=size-2
      own:int=1
      while own <= size-1:
        bit:int=1
        ptn:int=local.TOPBIT
        while ptn!=local.board[you] and local.board[own]>=bit:
          ptn>>=1
          bit<<=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        you-=1
        own+=1
      # 180度回転が同型
      if own>size-1:
        return 4
    # 270度回転
    if local.board[local.BOUND1]==local.TOPBIT:
      ptn:int=local.TOPBIT>>1
      own:int=1
      while own<=size-1:
        bit:int=1
        you:int=0
        while you<size and local.board[you]!=ptn and local.board[own] >= bit:
          bit<<=1
          you+=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn>>=1
        own+= 1
    # すべての回転が異なる
    return 8

  """ 角にQがある場合のバックトラック """
  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local)->int:
    global step
    step+=1
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        local.board[row]=bitmap
        return 8
    else:
      if row<local.BOUND1:
        bitmap|=2
        bitmap^=2
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit # Qを配置
      counter+=self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
    return counter


  """ 角にQがない場合のバックトラック """
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local)->int:
    global step
    step+=1
    """ノードレイヤーでの対称解除法"""
    counter: int = 0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        if (bitmap&local.LASTMASK)==0:
          local.board[row]=bitmap # Qを配置
          return self.symmetryOps(size,local)
    else:
      if row<local.BOUND1:
        bitmap|=local.SIDEMASK
        bitmap^=local.SIDEMASK
      elif row==local.BOUND2:
        if (down&local.SIDEMASK)==0:
          return 0
        if (down&local.SIDEMASK)!=local.SIDEMASK:
          bitmap&=local.SIDEMASK
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap ^=bit
      local.board[row]=bit
      counter+=self.symmetry_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
    return counter


  """ """
  def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local)

  """ 角にQがある場合のバックトラック """
  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list):
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    bit:int=0
    row:int= self.count_bits_nodeLayer(down)
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      local_list.append(Local( # 現在の`local`のコピーを追加
        TOTAL=local.TOTAL,UNIQUE=local.UNIQUE,TOPBIT=local.TOPBIT,
        ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,
        BOUND1=local.BOUND1,BOUND2=local.BOUND2,
        board=local.board.copy()
      ))
    if row<local.BOUND1:
      bitmap|=2
      bitmap^=2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      self.kLayer_nodeLayer_backtrack_corner(
        size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,
        local,local_list
      )
    return

  """ 角にQがない場合のバックトラック """
  def kLayer_nodeLayer_backtrack(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list)->int:
   counter:int=0
   mask:int=(1<<size)-1
   bitmap:int=mask&~(left|down|right)
   row:int= self.count_bits_nodeLayer(down)
   if row==k:
     nodes.append(left)
     nodes.append(down)
     nodes.append(right)
     local_list.append(Local( # 現在の`local`のコピーを追加
       TOTAL=local.TOTAL,UNIQUE=local.UNIQUE,TOPBIT=local.TOPBIT,
       ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,
       BOUND1=local.BOUND1,BOUND2=local.BOUND2,
       board=local.board.copy()
     ))
     return
   else:
     if row<local.BOUND1:
       bitmap|=local.SIDEMASK
       bitmap^=local.SIDEMASK
     elif row==local.BOUND2:
       if (down&local.SIDEMASK)==0:
         return
       if (down&local.SIDEMASK)!=local.SIDEMASK:
         bitmap&=local.SIDEMASK
   while bitmap:
     bit:int=-bitmap&bitmap
     bitmap^=bit
     local.board[row]=bit
     self.kLayer_nodeLayer_backtrack(
       size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,
       local,local_list
     )
   return counter










  """ kレイヤーのすべてのノードを含むベクトルを返す """
  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,types:list,local_list:list):
    # 初期化
    local=Local(
      TOTAL=0,UNIQUE=0,
      TOPBIT=1<<(size-1),ENDBIT=0,LASTMASK=0,SIDEMASK=0,
      BOUND1=2,BOUND2=0,board=[0]*size
    )
    local.board[0]=1
    # 角にQがある場合のバックトラック
    while local.BOUND1>1 and local.BOUND1<size-1:
      if local.BOUND1<size-1:
       bit:int=1<<local.BOUND1
       local.board[1]=bit
       self.kLayer_nodeLayer_backtrack_corner(
         size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,local,local_list
      )
      local.BOUND1+= 1
    local.TOPBIT=1<<(size-1)
    local.ENDBIT=local.TOPBIT>>1
    local.SIDEMASK=local.TOPBIT|1
    local.LASTMASK=local.TOPBIT|1
    local.BOUND1=1
    local.BOUND2=size-2
    # 角にQがない場合のバックトラック
    while local.BOUND1>0 and local.BOUND2<size-1 and local.BOUND1<local.BOUND2:
      if local.BOUND1<local.BOUND2:
        bit=1<<local.BOUND1
        local.board[0]=bit
        self.kLayer_nodeLayer_backtrack(
         size,nodes,k,bit<<1,bit,bit>>1,
         local,local_list
        )
      local.BOUND1+=1
      local.BOUND2-=1
      local.ENDBIT=local.ENDBIT>>1
      local.LASTMASK=(local.LASTMASK<<1)|local.LASTMASK|(local.LASTMASK>>1)
  """" """
  def symmetry_build_nodeLayer(self,size:int)->int:
    global step
    # ツリーの3番目のレイヤーにあるノードを生成
    nodes:list[int]=[]
    types:list[int]=[]
    local_list:list[Local]=[] # Localの配列を用意
    k:int=4 # 3番目のレイヤーを対象 
    self.kLayer_nodeLayer(size,nodes,k,types,local_list)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
    # ミラーでは/6 を /3に変更する 
    num_solutions=len(nodes)//3
    total:int=0
    for i in range(num_solutions):
      total+=self.symmetry_solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local_list[i])
    # print(step)
    
    return total

class NQueens21_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=16
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total:int=NQ.symmetry_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3] 
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
# メイン実行部分
if __name__=="__main__":
  NQueens21_NodeLayer().main()
