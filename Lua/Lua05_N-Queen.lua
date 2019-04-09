#!/usr/bin/env lua

--[[
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 *
 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。
 *

 実行結果 Lua
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
14:           365596        45752    00:00:05
15:          2279184       285053    00:00:29
16:         14772512      1846955    00:03:14
17:         95815104     11977939    00:22:57
]]--

NQueen={}; NQueen.new=function()
	--
  local this={
    SIZE=0;
    nTOTAL=0;
    nUNIQUE=0;
    nEquiv=0;
    colChk={};
    diagChk={};
    antiChk={};
    board={};
  };
	--
  function NQueen:secstotime(secs)
    sec=math.floor(secs);
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
	--
  function NQueen:intncmp(lt,rt,n)
    local k,rtn=0,0; 
    for k=0,n-1,1 do
      rtn=lt[k]-rt[k];
      if (rtn~=0)then break; end
    end
    return rtn;
  end
	--
  function NQueen:rotate(trial,scratch,n,neg)
    local k=0;
	  local incr=0;
    if neg then k=0; else k=n-1; end
    if neg then incr=1; else incr=-1; end 
	  local j=0;
	  while j<n do
      scratch[j]=trial[k];
		  k=k+incr;
      j=j+1;
    end
    if neg then k=n-1; else k=0; end 
	  local j=0;
	  while j<n do
      trial[scratch[j]]=k;
      j=j+1;
	    k=k-incr;
    end
  end
	--
  function NQueen:vMirror(check,n)
    for j=0,n-1,1 do
      check[j]=(n-1)-check[j];
    end
  end
	--
  function NQueen:symmetryOps()
	  local trial={};   --//回転・反転の対称チェック
	  local scratch={};
    self.nEquiv=0; 	  
    for k=0,self.size-1,1 do trial[k]=self.board[k]; end
    --//時計回りに90度回転
    self:rotate(trial,scratch,self.size,nil);
    local k=self:intncmp(self.board,trial,self.size);
    if(k>0)then return 0; end
    if(k==0)then self.nEquiv=1; else
      --//時計回りに180度回転
      self:rotate(trial,scratch,self.size,nil);
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      if(k==0)then self.nEquiv=2; else
        --//時計回りに270度回転
        self:rotate(trial,scratch,self.size,nil);
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
        self.nEquiv=4;
      end
    end  
    for k=0,self.size-1,1 do trial[k]=self.board[k]; end
    --//垂直反転 
    self:vMirror(trial,self.size);	
    k=self:intncmp(self.board,trial,self.size);
    if(k>0) then return 0; end
    --// 4回転とは異なる場合
    if (self.nEquiv>1) then
			-- 90度回転 対角鏡と同等
      self:rotate(trial,scratch,self.size,true);
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      --// 2回転とは異なる場合
      if(self.nEquiv>2)then
			  -- 180度回転 水平鏡像と同等
        self:rotate(trial,scratch,self.size,true);
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
			  -- 270度回転 反対角鏡と同等
        self:rotate(trial,scratch,self.size,true);
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
      end 
    end
    return self.nEquiv * 2;
  end
	--
  function NQueen:NQueens(row)
		--枝刈り
    if row==self.size then
	    local tst=self:symmetryOps();--//回転・反転・対称の解析
	    if(tst~=0) then
	      self.nUnique=self.nUnique+1;
	      self.nTotal=self.nTotal+tst;
	    end
    else
		  if(row~=0) then lim=self.size; else lim=(self.size+1)/2; end
      for col=0,self.size-1,1 do
        self.board[row]=col; 
        if	self.colChk[col]==nil 
			  and self.antiChk[row+col]==nil 
			  and self.diagChk[row-col+(self.size-1)]==nil then
          self.colChk[col],
          self.antiChk[row+col],
          self.diagChk[row-col+(self.size-1)]=true,true,true;
          self:NQueens(row+1);
          self.colChk[col],
          self.antiChk[row+col],
          self.diagChk[row-col+(self.size-1)]=nil,nil,nil;
        end
      end
    end
  end
	--
  function NQueen:NQueen()
	  local max=17 ; -- //最大N
    print(" N:            Total       Unique    hh:mm:ss");
    for si=2,max,1 do
      self.size=si; --初期化
		  self.nUnique,self.nTotal=0,0;
		  self.board,self.colChk,self.diagChk,self.antiChk={},{},{},{};
      s=os.time();  --計測開始
      for k=0,self.size-1,1 do
        self.board[k]=k;
      end
      self:NQueens(0);
      print(string.format("%2d:%17d%13d%12s",
												si,self.nTotal,self.nUnique,
												self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen();

