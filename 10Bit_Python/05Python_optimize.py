

from datetime import datetime 
# pypyを使う場合はコメントを解除
# import pypyjit
# pypyで再帰が高速化できる
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens05:
  max:int
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  trial:list[int]
  scratch:list[int]
  def __init__(self):
   self.max=16
   self.total=0
   self.unique=0
   self.aboard=[0 for i in range(self.max)]
   self.fa=[0 for i in range(2*self.max-1)]
   self.fb=[0 for i in range(2*self.max-1)]
   self.fc=[0 for i in range(2*self.max-1)]
   self.trial=[0 for i in range(self.max)]
   self.scratch=[0 for i in range(self.max)]

  def rotate(self,chk:list[int],scr:list[int],_n:int,neg:int):
    incr:int=0
    k:int=0
    # k=0 if neg else _n-1
    if neg:
      k=0
    else:
      k=_n-1
    # incr=1 if neg else -1
    if neg:
      incr=1
    else:
      incr-=1
    for i in range(_n):
      scr[i]=chk[k]
      k+=incr
    k=_n-1 if neg else 0
    for i in range(_n):
      chk[scr[i]]=k
      k-=incr

  def vmirror(self,chk:list[int],neg:int):
    for i in range(neg):
      chk[i]=(neg-1)-chk[i]

  def intncmp(self,_lt:list[int],_rt:list[int],neg)->int:
    rtn:int=0
    for i in range(neg):
      rtn=_lt[i]-_rt[i]
      if rtn!=0:
        break
    return rtn

  def symmetryops(self,size:int)->int:
    nequiv:int=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90
    self.rotate(self.trial,self.scratch,size,0)
    k:int=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if k==0:
      nequiv=1
    else:
      #180
      self.rotate(self.trial,self.scratch,size,0)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if k==0:
        nequiv=2
      else:
        #270
        self.rotate(self.trial,self.scratch,size,0)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0: 
          return 0
        nequiv=4
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 垂直反転
    self.vmirror(self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    # 90
    if nequiv > 1:
      self.rotate(self.trial,self.scratch,size,1)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      # 180
      if nequiv>2:
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        #270
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
    return nequiv*2

  def nqueens_rec(self,row:int,size:int):
    if row==size-1:
      if self.fb[row-self.aboard[row]+size-1] or self.fc[row+self.aboard[row]]:
        return
      stotal:int=self.symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
    else:
      lim:int=size if row!=0 else (size+1) //2
      tmp:int
      for i in range(row,lim):
        tmp=self.aboard[i]
        self.aboard[i]=self.aboard[row]
        self.aboard[row]=tmp
        if self.fb[row-self.aboard[row]+size-1]==0 and self.fc[row+self.aboard[row]]==0:
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=1
          self.nqueens_rec(row+1,size)
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=0
      tmp=self.aboard[row]
      for i in range(row+1,size):
        self.aboard[i-1]=self.aboard[i]
      self.aboard[size-1]=tmp

  def main(self):
    nmin:int=4
    nmax:int=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,self.max):
      self.total=0
      self.unique=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens_rec(0,size)
      time_elapsed=datetime.now()-start_time
      # _text='{}'.format(time_elapsed)
      # text=_text[:-3]
      # print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# 5.枝刈りと最適化
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184       285053         0:00:15.677
if __name__=='__main__':
  NQueens05().main();

