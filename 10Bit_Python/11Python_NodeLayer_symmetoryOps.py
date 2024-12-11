from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
#
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# from threading import Thread
# from multiprocessing import Pool as ThreadPool
# import concurrent
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
# Codon環境の判定
# Codon用の型定義

class Local:
      # フィールドをクラスレベルで定義
  TOTAL:int
  UNIQUE:int
  COUNT2:int
  COUNT4:int
  COUNT8:int
  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  TYPE:int
  board:list[int]
  def __init__(self,TOTAL:int,UNIQUE:int,COUNT2:int,COUNT4:int,COUNT8:int,
             TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,
             BOUND1:int,BOUND2:int,TYPE:int,board:list[int]):
    self.TOTAL=TOTAL
    self.UNIQUE=UNIQUE
    self.COUNT2=COUNT2
    self.COUNT4=COUNT4
    self.COUNT8=COUNT8
    self.TOPBIT=TOPBIT
    self.ENDBIT=ENDBIT
    self.LASTMASK=LASTMASK
    self.SIDEMASK=SIDEMASK
    self.BOUND1=BOUND1
    self.BOUND2=BOUND2
    self.TYPE=TYPE
    self.board=board

class NQueens21:
  def __init__(self):
    pass
  def count_bits_nodeLayer(self,n:int)->int:
      """ビットが1である数をカウント"""
      counter:int=0
      while n:
          n&=n-1
          counter+=1
      return counter  
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local):
      """ノードレイヤーでの対称解除法"""
      mask:int=(1<<size)-1
      bitmap:int=mask&~(left|down|right)
      row:int=self.count_bits_nodeLayer(down)
      if row==(size-1):
          if bitmap:
              if (bitmap&local.LASTMASK)==0:
                  local.board[row]=bitmap  # Qを配置
                  self.symmetryOps(size,local)
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
          bitmap ^=bit
          local.board[row]=bit
          self.symmetry_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
      return

  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local):
      """角にQがある場合の対称解除法"""
      mask:int=(1<<size)-1
      bitmap:int=mask&~(left|down|right)
      row:int=self.count_bits_nodeLayer(down)
      if row==(size-1):
          if bitmap:
              local.board[row]=bitmap
              local.COUNT8+=1
              return
      else:
          if row<local.BOUND1: # 枝刈り
              bitmap|=2
              bitmap^=2
      while bitmap:
          bit:int=-bitmap&bitmap
          bitmap^=bit
          local.board[row]=bit  # Qを配置
          self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
      return

  def symmetryOps(self,size:int,local:Local):
      """対称解除操作"""
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
                  return
              if local.board[own]<bit:
                  break
              ptn<<=1
              own+=1
          # 90度回転が同型
          if own>size-1:
              local.COUNT2+=1
              return
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
                  return
              if local.board[own]<bit:
                  break
              you-=1
              own+=1
          # 180度回転が同型
          if own>size-1:
              local.COUNT4+=1
              return
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
                  return
              if local.board[own]<bit:
                  break
              ptn>>=1
              own+= 1
      # すべての回転が異なる
      local.COUNT8+=1
      return

  def kLayer_nodeLayer_backtrack(self,size:int,nodes:list,k:int,
                               left:int,down:int,right:int,
                               local:Local,types:list,local_list:list[Local])->int:
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int= self.count_bits_nodeLayer(down)
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      types.append(1)
      local_list.append(Local(  # 現在の`local`のコピーを追加
          TOTAL=local.TOTAL,UNIQUE=local.UNIQUE,COUNT2=local.COUNT2,
          COUNT4=local.COUNT4,COUNT8=local.COUNT8,TOPBIT=local.TOPBIT,
          ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,
          BOUND1=local.BOUND1,BOUND2=local.BOUND2,TYPE=local.TYPE,
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
          local,types,local_list
      )
    return counter

  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:list,k:int,
                                      left:int,down:int,right:int,local:Local,types:list,local_list:list[Local]):
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    bit:int=0
    row:int= self.count_bits_nodeLayer(down)
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      types.append(0)
      local_list.append(Local(  # 現在の`local`のコピーを追加
          TOTAL=local.TOTAL,UNIQUE=local.UNIQUE,COUNT2=local.COUNT2,
          COUNT4=local.COUNT4,COUNT8=local.COUNT8,TOPBIT=local.TOPBIT,
          ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,
          BOUND1=local.BOUND1,BOUND2=local.BOUND2,TYPE=local.TYPE,
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
          local,types,local_list
      )
    return

  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,types:list,local_list:list[Local]):
    """kレイヤーのすべてのノードを含むベクトルを返す"""
    # 初期化
    local=Local(
      TOTAL=0,UNIQUE=0,COUNT2=0,COUNT4=0,COUNT8=0,
      TOPBIT=1<<(size-1),ENDBIT=0,LASTMASK=0,SIDEMASK=0,
      BOUND1=2,BOUND2=0,TYPE=0,board=[0]*size
    )
    local.board[0]=1
    # 角にQがある場合のバックトラック
    while local.BOUND1>1 and local.BOUND1<size-1:
      if local.BOUND1<size-1:
          bit:int=1<<local.BOUND1
          local.board[1]=bit
          self.kLayer_nodeLayer_backtrack_corner(
              size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,local,types,local_list
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
              local,types,local_list
          )
      local.BOUND1+=1
      local.BOUND2-=1
      local.ENDBIT=local.ENDBIT>>1
      local.LASTMASK=(local.LASTMASK<<1)|local.LASTMASK|(local.LASTMASK>>1)
  
  def symmetry_build_nodeLayer(self,size:int)->int:
    # ツリーの3番目のレイヤーにあるノードを生成
    nodes:list[int]=[]
    types:list[int]=[]
    local_list:list[Local]=[]  # Localの配列を用意
    k:int=5  # 3番目のレイヤーを対象      
    self.kLayer_nodeLayer(size,nodes,k,types,local_list)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
    # ミラーでは/6 を /3に変更する 
    num_solutions=len(nodes)//3
    total:int=0
    for i in range(num_solutions):
      local=local_list[i]
      if types[i]==0:
        self.symmetry_solve_nodeLayer_corner(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local)
      else:
        self.symmetry_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local)
    total=sum(l.COUNT2*2+l.COUNT4*4+l.COUNT8*8 for l in local_list)
    return total

class NQueens21_NodeLayer:
  def main(self)->None:
    nmin:int=7
    nmax:int=15
    print(" N:       Total       Unique        hh:mm:ss.ms")
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
