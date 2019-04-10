#!/usr/bin/env luajit

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
 １１．バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化＋対称解除法のビッ トマップ化＋クイーンの位置による振り分け（BOUND1+BOUND2)＋枝刈り

	実行結果


 ]]--

Info={}; Info.new=function()
  local this={
    nTotal=0;nUniq=0; nextCol=0; limit=0;
    starttime=os.time();
  };
  function Info:resetCount(size)
    self.nTotal,self.nUniq=0,0;
    self.limit=1;
  end
  function Info:nextJob(nS,nU)
    self.nTotal=self.nTotal+nS;
    self.nUniq=self.nUniq+nU;
    if self.nextCol<self.limit then
      self.nextCol=self.nextCol+1;
    else self.nextCol=-1; end
    return self.nextCol;
  end
  function Info:getTotal() return self.nTotal; end
  function Info:getUnique() return self.nUniq; end
  function Info:getTime() 
    return self:secstotime(os.difftime(os.time(),self.starttime)); 
  end
  function Info:secstotime(secs)
    sec=math.floor(secs)
    if(sec>59) then
      local hour=math.floor(sec*0.000277777778);
      local minute=math.floor(sec*0.0166666667)-hour*60;
      sec=sec-hour*3600-minute*60
      if(sec<10)then sec="0"..sec; end
      if(hour<10)then hour="0"..hour; end
      if(minute<10)then minute="0"..minute; end
      return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec;
  end 
  return setmetatable(this,{__index=Info});
end

Thread={}; Thread.new=function()
  local this={
    size=2;
    nTotal=0;nUniq=0;
    C2=0;C4=0;C8=0;
    aB={};--array board
    B=0; --BIT
    M=0; --MASK
    SE=0;--SIZEE
    TB=0;--TOPBIT
    SM=0;--SIDEMASK
    LM=0;--LASTMAK
    EB=0;--ENDBIT
    B1=0;--BOUND1
    B2=0;--BOUND2
  };
  function Thread:Thread(size,info)
    self.size=size;
    self.info=info;
    info:resetCount();
  end
  function Thread:run()
    local nextCol;
    local size=self.size;
    while(true) do
      nextCol=info:nextJob(self.nTotal,self.nUniq);
      if nextCol<0 then break; end
      self.nTotal,self.nUniq=0,0;
      for y=0,size-1,1 do self.aB[y]=y; end --テーブルの初期化
      self:BM_rotate(size);
    end
  end

  function Thread:Check(bsize)
    --90度回転
    local SE=self.SE; --SIZEE
    local aB=self.aB; --array board[]
    local TB=self.TB; --TOPBIT
    local EB=self.EB; --ENDBIT
    local B1=self.B1; --BOUND1
    local B2=self.B2; --BOUND2
		if aB[B2]==1 then 
			local own=1; 
      local ptn=2; 
      while own<=SE do
        self.B=1; 
        local you=SE; 
        while aB[you]~=ptn and aB[own]>=self.B do
          --self.B=(self.B<<1);
          self.B=bit.lshift(self.B,1);
          you=you-1;
        end
        if aB[own]>self.B then return; end
        if aB[own]<self.B then break; end
        own=own+1;
        --ptn=(ptn<<1);
        ptn= bit.lshift(ptn,1);
      end
		--//90度回転して同型なら180度/270度回転も同型である
      if own>SE then
        self.C2=self.C2+1;
        return;
      end
    end
    --//180度回転
    if aB[SE]==EB then
      local own=1; 
      local you=SE-1;
      while own<=SE do
        self.B=1; 
        local ptn=TB;
        while ptn~=aB[you] and aB[own]>=self.B do
          --self.B=(self.B<<1);
          self.B=bit.lshift(self.B,1);
          --ptn=(ptn>>1);
          ptn=bit.rshift(ptn,1);
        end
        if aB[own]>self.B then return; end
        if aB[own]<self.B then break; end
        own=own+1;
        you=you-1;
      end
    --	//90度回転が同型でなくても180度回転が同型である事もある
      if own>SE then
        self.C4=self.C4+1;
        return;
      end
    end
    --	//270度回転
    if aB[B1]==TB then
      local own=1; 
      --local ptn=(TB>>1); 
      local ptn=bit.rshift(self.TB,1);
      while own<=SE do
        self.B=1; 
        local you=0;
        while aB[you]~=ptn and aB[own]>=self.B do
          --self.B=(self.B<<1);
          self.B=bit.lshift(self.B,1);
          you=you+1;
        end
        if aB[own]>self.B then return; end
        if aB[own]<self.B then break; end
        own=own+1;
        --ptn=(ptn>>1);
        ptn=bit.rshift(ptn,1);
      end
    end
    self.C8=self.C8+1;
  end   
  --ビット反転させるメソッド・・・
  function Thread:rbits(byte,sz)
    local score=0;
    for i=sz,0,-1 do
    --io.write(bit.bnot(bit.band(bit.arshift(byte,i), 1)))
      if bit.band(bit.arshift(byte,i), 1) ==0 then
        score=score+2^i;
      end
    end
    return score;
  end
	--* 最上段のクイーンが角以外にある場合の探索
  function Thread:backTrack2(y,l,d,r)
    --local BM=(self.M&~(l|d|r)); -- BM:bitmap
    local BM=bit.band(self.M,self:rbits(bit.bor(l,d,r),self.size-1));
    local SE=self.SE; --SEZIE
    local LM=self.LM; --LASTmASK
    local SM=self.SM; --SIDEMASK
    local B1=self.B1; --BOUND1
    local B2=self.B2; --BOUND2
    if y==SE then
      if BM~=0 then 
        --if (BM&LM)==0 then
        if bit.band(BM,LM)==0 then
          self.aB[y]=BM;
          self:Check(BM);
        end
      end
    else
      if y<B1 then
        --BM=(BM|SM);
        BM=bit.bor(BM,SM);
        --BM=(BM~SM);
        BM=bit.bxor(BM,SM);
      elseif y==B2 then
        --if(d&SM)==0 then return; end 
        if(bit.band(d,SM)==0) then return; end
        --if(d&SM)~=SM then BM=(BM&SM); end
        if(bit.band(d,SM)~=SM) then BM=bit.band(BM,SM); end
      end
      while BM~=0 do
        --self.aB[y],self.B=(-BM&BM),(-BM&BM);
        self.aB[y],self.B=bit.band(-BM,BM),bit.band(-BM,BM);
        --BM=(BM~self.aB[y]);
        BM=bit.bxor(BM,self.aB[y]);
        --self:backTrack2(y+1,(l|self.B)<<1,(d|self.B),((r|self.B)>>1));
        self:backTrack2(y+1,bit.lshift(bit.bor(l,self.B),1),bit.bor(d,self.B),bit.rshift(bit.bor(r,self.B),1));
      end
    end
  end
  -- * 最上段のクイーンが角にある場合の探索
  function Thread:backTrack1(y,l,d,r)
    --local BM=(self.M&~(l|d|r));
    local BM=bit.band(self.M,self:rbits(bit.bor(l,d,r),self.size-1));
    local SE=self.SE; --SEZIE
    local B1=self.B1; --BOUND1
    if y==SE then
      if BM~=0 then
          self.aB[y]=BM;
          self.C8=self.C8+1;
      end
    else
      if y<B1 then
        --BM=BM|2;
        BM=bit.bor(BM,2);
        --BM=BM~2;
        BM=bit.bxor(BM,2);
      end
      while BM~=0 do
        --self.aB[y],self.B=(-BM&BM),(-BM&BM)
        self.aB[y],self.B=bit.band(-BM,BM),bit.band(-BM,BM);
        --BM=(BM~self.aB[y]);
        BM=bit.bxor(BM,self.aB[y]);
        --self:backTrack1(y+1,(l|self.B)<<1,(d|self.B),(r|self.B)>>1);
        self:backTrack1(y+1,bit.lshift(bit.bor(l,self.B),1),bit.bor(d,self.B),bit.rshift(bit.bor(r,self.B),1));
      end
    end
  end
  function Thread:BM_rotate(size)
    self.SE=size-1;
		local SE=self.SE;
    --self.TB=(1<<SE);
		self.TB=bit.lshift(1,SE);
    --self.M=(1<<size)-1;    
    self.M=bit.lshift(1,size)-1;    
    self.aB[0]=1;
    self.B1=2;
    while self.B1>1 and self.B1<SE do
      --self.aB[1],self.B=(1<<self.B1),(1<<self.B1);
			self.aB[1],self.B=bit.lshift(1,self.B1),bit.lshift(1,self.B1);
      --self:backTrack1(2,(2|self.B)<<1,(1|self.B),(self.B>>1));
      self:backTrack1(2,bit.lshift(bit.bor(2,self.B),1),bit.bor(1,self.B),bit.rshift(self.B,1));
      self.B1=self.B1+1;
    end
    --self.SM,self.LM=(self.TB|1),(self.TB|1);
		self.SM,self.LM=bit.bor(self.TB,1),bit.bor(self.TB,1);
    --self.EB=(self.TB>>1);
    self.EB=bit.rshift(self.TB,1);
    self.B1=1;
    self.B2=size-2;
    while self.B1>0 and self.B2<SE and self.B1<self.B2 do
      --self.aB[0],self.B=(1<<self.B1),(1<<self.B1);
      self.aB[0],self.B=bit.lshift(1,self.B1),bit.lshift(1,self.B1);
      --self:backTrack2(1,self.B<<1,self.B,self.B>>1);
      self:backTrack2(1,bit.lshift(self.B,1),self.B,bit.rshift(self.B,1));
      --self.LM=(self.LM|self.LM>>1|self.LM<<1);
      self.LM=bit.bor(self.LM,bit.rshift(self.LM,1),bit.lshift(self.LM,1));
      --self.EB=(self.EB>>1);
      self.EB=bit.rshift(self.EB,1);
      self.B1=self.B1+1;
      self.B2=self.B2-1;
    end
    self.nUniq=self.C8+self.C4+self.C2;
    self.nTotal=(self.C8*8)+(self.C4*4)+(self.C2*2);
  end
  return setmetatable(this,{__index=Thread});
end

NQueen={}; NQueen.new=function()
  local this={};
  function NQueen:NQueen()
    local max=17;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,max,1 do
      info=Info.new(); 
      thread=Thread.new();
      thread:Thread(size,info);   
      thread:run();
      print(string.format("%2d:%17d%13d%12s", 
      size,info:getTotal(),info:getUnique(),info:getTime())); 
    end
  end
  return setmetatable(this,{__index=NQueen} );
end

NQueen.new():NQueen();

