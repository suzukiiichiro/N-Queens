
from datetime import datetime 
# pypyを使う場合はコメントを解除
# import pypyjit
# pypyで再帰が高速化できる
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens07():
  total:int
  unique:int
  def __init__(self):
    self.total=0
    self.unique=0
  def mirror(self,size:int,row:int,left:int,down:int,right:int):
    if row==size:
      self.total+=1
    else:
      bit:int=0
      mask:int=(1<<size)-1
      bitmap:int=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        bitmap=bitmap&~bit
        self.mirror(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def NQueens(self,size:int,row:int,left:int,down:int,right:int):
    bit:int=0
    limit:int
    if size%2:
      limit=size//2-1
    else:
      limit=size//2
    for i in range(0,size//2):
      bit=1<<i
      self.mirror(size,1,bit<<1,bit,bit>>1)
    if size%2:
      bit=1<<(size-1)//2
      left=bit<<1
      down=bit
      right=bit>>1
      for i in range(0,limit):
        bit=1<<i
        self.mirror(size,2,(left|bit)<<1,down|bit,(right|bit)>>1)
    self.total=self.total<<1; # 倍にする
  def main(self):
    nmin:int = 4
    nmax:int = 16
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      self.total=0
      self.unique=0
      start_time = datetime.now()
      self.NQueens(size,0,0,0,0)
      time_elapsed = datetime.now()-start_time
      # _text = '{}'.format(time_elapsed)
      # text = _text[:-3]
      # print("%2d:%13d%13d%20s" % (i,self.total,self.unique, text))  
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# 6.バックトラックとビットマップ
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184            0         0:00:00.716
if __name__=='__main__':
  NQueens07().main();

