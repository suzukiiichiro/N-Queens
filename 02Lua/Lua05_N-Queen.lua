#!/usr/bin/env luajit

--[[
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 *
 * ５．対称解除法＋枝刈り
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。
 *

 実行結果 Luajit
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
14:           365596        45752    00:00:03
15:          2279184       285053    00:00:21
16:         14772512      1846955    00:02:09
17:         95815104     11977939    00:16:38
]]--

NQueen={}; NQueen.new=function()
	--
  local this={
    TOTAL=0;
    UNIQUE=0;
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
  function NQueen:intncmp(board,trial,size)
    local rtn=0; 
    for k=0,size,1 do
      rtn=board[k]-trial[k];
      if (rtn~=0)then 
        break; 
      end
    end
    return rtn;
  end
	--
  function NQueen:rotate(trial,scratch,size,neg)
    local k;
	  local incr;
    if neg then k=0; else k=size-1; end
    if neg then incr=1; else incr=-1; end 

	  local j=0;
	  while j<size do
      scratch[j]=trial[k];
		  k=k+incr;
      j=j+1;
    end

    if neg then k=size-1; else k=0; end 

	  local j=0;
	  while j<size do
      trial[scratch[j]]=k;
	    k=k-incr;
      j=j+1;
    end
  end
	--
  function NQueen:vMirror(trial,size)
    for j=0,size,1 do
      trial[j]=(size-1)-trial[j];
    end
  end
	--
  function NQueen:symmetryOps(board,trial,scratch,size)
    local nEquiv=0; 	  
    --初期化
    for i=0,size,1 do 
      trial[i]=board[i]; 
    end
    --//時計回りに90度回転
    self:rotate(trial,scratch,size,0);
    local k=self:intncmp(board,trial,size);
    if(k>0)then return 0; end
    if(k==0)then 
      nEquiv=1; 
    else
      --//時計回りに180度回転
      self:rotate(trial,scratch,size,0);
      k=self:intncmp(board,trial,size);
      if(k>0) then return 0; end
      if(k==0)then 
        nEquiv=2; 
      else
        --//時計回りに270度回転
        self:rotate(trial,scratch,size,0);
        k=self:intncmp(board,trial,size);
        if(k>0) then 
          return 0; 
        end
        nEquiv=4;
      end
    end  
    --初期化
    for i=0,size,1 do 
      trial[i]=board[i]; 
    end
    --//垂直反転 
    self:vMirror(trial,size);	
    k=self:intncmp(board,trial,size);
    if(k>0) then 
      return 0; 
    end
    --// 4回転とは異なる場合
    if (nEquiv>1) then
			-- 90度回転 対角鏡と同等
      self:rotate(trial,scratch,size,1);
      k=self:intncmp(board,trial,size);
      if(k>0) then 
        return 0; 
      end
      --// 2回転とは異なる場合
      if(nEquiv>2)then
			  -- 180度回転 水平鏡像と同等
        self:rotate(trial,scratch,size,1);
        k=self:intncmp(board,trial,size);
        if(k>0) then 
          return 0; 
        end
			  -- 270度回転 反対角鏡と同等
        self:rotate(trial,scratch,size,1);
        k=self:intncmp(board,trial,size);
        if(k>0) then 
          return 0; 
        end
      end 
    end
    return nEquiv * 2;
  end
	--
  function NQueen:NQueens(board,row,size)
    local tmp;
    local trial={};
    local scratch={};
    --枝刈り
    -- if row==size then
    if row==size-1 then
      --枝刈り
      if(self.diagChk[row-board[row]+size-1]==true
        or self.antiChk[row+board[row]]==true) then
        return;
      end
	    local tst=self:symmetryOps(board,trial,scratch,size);--//回転・反転・対称の解析
	    if(tst~=0) then
	      self.UNIQUE=self.UNIQUE+1;
        self.TOTAL=self.TOTAL+tst;
	    end
    else
      --枝刈り
      local lim ;
		  if row~=0 then lim=size; else lim=(size+1)/2; end
      --for i=0,size-1,1 do
        for i=row,lim-1,1 do
        --board[row]=i; 
          -- 交換
          tmp=board[i];
          board[i]=board[row];
          board[row]=tmp;
          if	self.antiChk[row+board[row]]==nil
          and self.diagChk[row-board[row]+(size-1)]==nil then
            self.antiChk[row+board[row]],
            self.diagChk[row-board[row]+(size-1)]=true,true;
            self:NQueens(board,row+1,size);
            self.antiChk[row+board[row]],
            self.diagChk[row-board[row]+(size-1)]=nil,nil;
          end
        end
        -- 交換
        tmp=board[row];
        for i=row+1,size,1 do
          board[i-1]=board[i];
        end
        board[size-1]=tmp;
      end
  end
	--
  function NQueen:NQueen()
	  local max=17 ; -- //最大N
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,max,1 do
		  self.UNIQUE=0;
      self.TOTAL=0;
      s=os.time();  --計測開始
      for j=0,size,1 do
        self.board[j]=j;
      end
      self:NQueens(self.board,0,size);
      print(string.format("%2d:%17d%13d%12s",
												size,self.TOTAL,self.UNIQUE,
												self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen();

