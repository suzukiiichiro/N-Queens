#!/usr/bin/env luajit

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
 ８．バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化

 	実行結果

 N:            Total       Unique    hh:mm:ss
 2:                0            0    00:00:00
 3:                0            0    00:00:00
 4:                2            1    00:00:00
 5:               10            2    00:00:00
 6:                4            1    00:00:00
 7:               40            6    00:00:00
 8:               92           12    00:00:00
 9:              352           46    00:00:00
10:              724           92    00:00:00
11:             2680          341    00:00:00
12:            14200         1787    00:00:00
13:            73712         9233    00:00:01
14:           365596        45752    00:00:01
15:          2279184       285053    00:00:10
16:         14772512      1846955    00:01:01
17:         95815104     11977939    00:07:33

  ]]--

NQueen={}; NQueen.new=function()
  -- 
  local this={
    size=0;
    UNIQUE=0;
    MASK=0;
    --nTotal=0;nUniq=0;nEquiv=0; 
    COUNT2=0;COUNT4=0;COUNT8=0;
    BOUND1=0;
    BOUND2=0;
    TOPBIT=0;
    ENDBIT=0;
    SIDEMASK=0;
    LASTMASK=0;
    board={};trial={};scratch={};
    -- trial={};scratch={};
  };
  --
  function NQueen:secstotime(secs)
    sec=math.floor(secs);
	  if(sec>59) then
		  local hour = math.floor(sec*0.000277777778)
		  local minute = math.floor(sec*0.0166666667) - hour*60
		  sec = sec - hour*3600 - minute*60
		  if(sec<10)then sec = "0"..sec end
		  if(hour<10)then hour = "0"..hour end
		  if(minute<10)then minute = "0"..minute end
		  return hour..":"..minute..":"..sec
	  end
	  if(sec<10)then sec = "0"..sec end
	  return "00:00:"..sec
  end 
  --
  function NQueen:rotate(bf,af,si)
    for i=0,si,1 do
      local t=0;
      for j=0,si,1 do
        --t=t|((bf[j]>>i)&1)<<(si-j-1); 
        t=bit.bor(t,bit.lshift(bit.band(bit.rshift(bf[j],i),1),(si-j-1)));
      end 
      af[i]=t;
    end 
  end
  --
  function NQueen:vMirror(bf,af,si)
    local score ;
    for i=0,si,1 do 
      score=bf[i];
      af[i]=self:rh(score,si-1);
    end 
  end
  --
  function NQueen:rh(a,sz)
    local tmp=0;
    for i=0,sz,1 do
      --if(a&(1<<i))then  
      if bit.band(a,bit.lshift(1,i))~=0 then
         --return tmp|=(1<<(sz-i)); 
         return bit.bor(tmp,bit.lshift(1,(sz-i)));
      end
    end 
    return tmp;
  end
  --
  function NQueen:intncmp(lt,rt,si)
    local rtn=0;
    for k=0,si,1 do
      rtn=lt[k]-rt[k];
      if(rtn~=0)then break;end
    end 
    return rtn;
  end
  --
  function NQueen:symmetryOps(si)
    local nEquiv;
    --// 回転・反転・対称チェックのためにboard配列をコピー
    for i=0,si,1 do self.trial[i]=self.board[i]; end
    --//時計回りに90度回転
    self:rotate(self.trial,self.scratch,si);    
    local k=self:intncmp(self.board,self.scratch,si);
    if(k>0)then return; end
    if(k==0)then nEquiv=2;
    else
      --//時計回りに180度回転
      self:rotate(self.scratch,self.trial,si);  
      k=self:intncmp(self.board,self.trial,si);
      if(k>0)then return; end 
      if(k==0)then nEquiv=4;
      else
        --//時計回りに270度回転
        self:rotate(self.trial,self.scratch,si);
        k=self:intncmp(self.board,self.scratch,si);
        if(k>0) then return; end 
        nEquiv=8;
      end 
    end 
    --// 回転・反転・対称チェックのためにboard配列をコピー
    for i=0,si,1 do
      self.scratch[i]=self.board[i];
    end
    --//垂直反転
    self:vMirror(self.scratch,self.trial,si);   
    k=self:intncmp(self.board,self.trial,si);
    if(k>0)then return; end 
    if(nEquiv>2)then    --//-90度回転 対角鏡と同等       
      self:rotate(self.trial,self.scratch,si);
      k=self:intncmp(self.board,self.scratch,si);
      if(k>0)then return; end 
      if(nEquiv>4)then  --//-180度回転 水平鏡像と同等
        self:rotate(self.scratch,self.trial,si);
        k=self:intncmp(self.board,self.trial,si);
        if(k>0)then return; end
        --//-270度回転 反対角鏡と同等
        self:rotate(self.trial,self.scratch,si);
        k=self:intncmp(self.board,self.scratch,si);
        if(k>0)then return; end 
      end 
    end 
    if(nEquiv==2)then self.COUNT2=self.COUNT2+1;end 
    if(nEquiv==4)then self.COUNT4=self.COUNT4+1;end 
    if(nEquiv==8)then self.COUNT8=self.COUNT8+1;end
  end
  function NQueen:backTrack2(min,left,down,right) 
    local BIT;
    local bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right ),self.size-1));
    if min==self.size then
      --self.TOTAL=self.TOTAL+1 ;
      if bitmap==0 then
        self.board[min]=bitmap;
        self:symmetryOps(self.size);
      end
    else
      if min < self.BOUND1 then
        bitmap=bit.bor(bitmap,self.SIDEMASK);
        bitmap=bit.bxor(bitmap,self.SIDEMASK);
      end
      if min==self.BOUND2 then
        if bit.band(down,self.SIDEMASK) ==0 then
          return;
        end
        if bit.band(down,self.SIDEMASK)~=self.SIDEMASK then
           bitmap=bit.band(bitmap,self.SIDEMASK)
        end
     end
      while bitmap~=0 do
        BIT=bit.band(-bitmap,bitmap);
        self.board[min]=BIT;
        bitmap=bit.bxor(bitmap,BIT);
        self:backTrack2(min+1,bit.lshift(bit.bor(left,BIT),1),bit.bor(down,BIT),bit.rshift(bit.bor(right,BIT),1));
      end
    end
  end
  --
  function NQueen:backTrack1(min,left,down,right) 
    local BIT;
    local bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right ),self.size-1));
    if min==self.size then
      --self.TOTAL=self.TOTAL+1 ;
      if bitmap==0 then
        self.board[min]=bitmap;
        --self:symmetryOps(self.size);
        self.COUNT8=self.COUNT8+1;
      end
    else
      if min<self.BOUND1 then
        bitmap=bit.bor(bitmap,2);
        bitmap=bit.bxor(bitmap,2);

      end
      while bitmap~=0 do
        BIT=bit.band(-bitmap,bitmap);
        self.board[min]=BIT;
        bitmap=bit.bxor(bitmap,BIT);
        self:backTrack1(min+1,bit.lshift(bit.bor(left,BIT),1),bit.bor(down,BIT),bit.rshift(bit.bor(right,BIT),1));
      end
    end
  end
  --
  function NQueen:NQueens(min) 
    local BIT;
    self.TOPBIT=bit.lshift(1,(self.size-1));
    self.board[0]=1;
    for BOUND1=2,self.size-1,1 do
      BIT=bit.lshift(1,BOUND1);
      self.board[1]=BIT;
      self.BOUND1=BOUND1;
      self:backTrack1(2,bit.lshift(bit.bor(2,BIT),1),bit.bor(1,BIT),bit.rshift(BIT,1));

    end
    self.LASTMASK=bit.bor(self.TOPBIT,1);
    self.SIDEMASK=self.LASTMASK;
    self.ENDBIT=bit.rshift(self.TOPBIT,1);
    self.BOUND2=self.size-2;
    for BOUND1=1,self.BOUND2,1 do
      BIT=bit.lshift(1,BOUND1);
      self.board[0]=BIT;
      self.BOUND1=BOUND1;
      self:backTrack2(1,bit.lshift(BIT,1),BIT,bit.rshift(BIT,1));
      self.LASTMASK=bit.bxor(self.LASTMASK,bit.bor(bit.rshift(self.LASTMASK,1),bit.lshift(self.LASTMASK,1)));
      self.ENDBIT=bit.rshift(self.ENDBIT,1);
      self.BOUND2=self.BOUND2-1;
    end
  end
  --
  function NQueen:rbits(byte,sz)
    local score=0;
    for i=sz,0,-1 do
      if bit.band(bit.arshift(byte,i), 1) ==0 then
        score=score+2^i;
      end
    end
    return score;
  end
  --
  function NQueen:NQueen()
    local max=17;
    print(" N:            Total       Unique    hh:mm:ss");
    for si=2,max,1 do
      self.size=si;
      self.TOTAL=0;
      self.UNIQUE=0;
      self.COUNT2=0;
      self.COUNT4=0;
      self.COUNT8=0;
      for k=0,self.size-1,1 do self.board[k]=k; end --テーブルの初期化
      self.MASK=bit.lshift(1,self.size)-1;    
      s=os.time();
      self:NQueens(0);
      self.TOTAL=self.COUNT2*2+self.COUNT4*4+self.COUNT8*8;
      self.UNIQUE=self.COUNT2+self.COUNT4+self.COUNT8;
      print(string.format("%2d:%17d%13d%12s",si,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  return setmetatable( this,{__index=NQueen} );
end
  --
NQueen.new():NQueen();
