
from datetime import datetime 
# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

class NQueens08():
  total:int
  unique:int
  bound1:int
  bound2:int
  topbit:int
  endbit:int
  sidemask:int
  lastmask:int
  board:list[int]
  count2:int
  count4:int
  count8:int
  def __init__(self):
    self.total=0
    self.unique=0
    self.bound1=0
    self.bound2=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    self.count2=0
    self.count4=0
    self.count8=0
  def symmetryops(self,size:int):
    if self.board[self.bound2]==1:
      own:int=1
      ptn:int=2
      you:int
      bit:int
      while own<=size-1:
        bit=1
        you=size-1
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          you-=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        ptn<<=1
      # 90度回転が同型
      if own>size-1:
        self.count2+=1
        return
    # 180度回転
    if self.board[size-1]==self.endbit:
      own=1
      you=size-1-1
      while own<=size-1:
        bit=1
        ptn=self.topbit
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          ptn>>=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        you-=1
      if own>size-1:
        self.count4+=1
        return
    if self.board[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=size-1:
        bit=1
        you=0
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          you+=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        ptn>>=1
    self.count8+=1
  # 角にQがない場合のバックトラック
  def backTrack2(self,size:int,row:int,left:int,down:int,right:int):
    bit:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    if row==(size-1):
      if bitmap:
        if (bitmap&self.lastmask==0):
          self.board[row]=bitmap
          self.symmetryops(size)
    else:
      if row<self.bound1:
        # bitmap&=~self.sidemask
        bitmap=bitmap|self.sidemask
        bitmap=bitmap^self.sidemask
      else:
        if row==self.bound2:
          if down&self.sidemask==0:
            return
          if (down&self.sidemask)!=self.sidemask:
            bitmap&=self.sidemask
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        self.board[row]=bit
        self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  # 角にQがある場合のバックトラック
  def backTrack1(self,size:int,row:int,left:int,down:int,right:int):
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    bit:int=0
    if row==(size-1):
      if bitmap:
        self.board[row]=bitmap
        self.count8+=1
    else:
      if row<self.bound1:
        # bitmap&=~2
        bitmap=bitmap|2
        bitmap=bitmap^2
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        self.board[row]=bit
        self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def NQueens(self,size:int):
    bit:int=0
    self.total=self.unique=self.count2=self.count4=self.count8=0
    self.topbit=1<<(size-1)
    self.endbit=self.lastmask=self.sidemask=0
    self.bound1=2
    self.bound2=0
    self.board[0]=1
    # 角にQがある場合のバックトラック
    while self.bound1>1 and self.bound1<size-1:
      if self.bound1<(size-1):
        bit=1<<self.bound1
        self.board[1]=bit
        self.backTrack1(size,2,(2|bit)<<1,1|bit,(2|bit)>>1)
      self.bound1+=1
    self.topbit=1<<(size-1)
    self.endbit=self.topbit>>1
    self.sidemask=self.lastmask=self.topbit|1
    self.bound1=1
    self.bound2=size-2
    # 角にQがない場合のバックトラック
    while self.bound1>0 and self.bound2<size-1 and self.bound1<self.bound2:
      if self.bound1<self.bound2:
        bit=1<<self.bound1
        self.board[0]=bit
        self.backTrack2(size,1,bit<<1,bit,bit>>1)
      # self.bound1+=1
      self.bound2-=1
      self.endbit=self.endbit>>1
      self.lastmask=self.lastmask<<1|self.lastmask|self.lastmask>>1
    self.unique=self.count2+self.count4+self.count8
    self.total=self.count2*2+self.count4*4+self.count8*8
  def main(self):
    nmin:int=4
    nmax:int=16
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      self.bound1=0
      self.bound2=0
      self.topbit=0
      self.endbit=0
      self.sidemask=0
      self.lastmask=0
      self.board=[0 for i in range(size)]
      start_time = datetime.now()
      self.NQueens(size)
      time_elapsed = datetime.now()-start_time
      # _text = '{}'.format(time_elapsed)
      # text = _text[:-3]
      # print("%2d:%13d%13d%20s" % (size,self.total,self.unique, text))
      text = str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 7.対象解除法とビットマップ
# 15:      2279184       285053         0:00:00.377
if __name__=='__main__':
  NQueens08().main();


