#!/usr/bin/env luajit

--[[
/**
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  ６．スレッド
  
  　クイーンが上段角にある場合とそうではない場合の二つにスレッドを分割し並行処理
  さらに高速化するならば、rowひとつずつにスレッドを割り当てる方法もある。
  　backTrack1とbackTrack2を以下で囲んでスレッド処理するとよい。
  　ただしスレッド数を管理する必要がある。

	実行結果

]]--


Info={}; Info.new=function()
  local this={
    nTotal=0;nUniq=0; nextCol=0; limit=1;
    starttime=os.time();
  };
  function Info:resetCount()
    self.nTotal,self.nUniq=0,0;
    --self.limit=size;
    --self.limit=1;
  end
  function Info:nextJob(nS,nU)
    self.nTotal=self.nTotal+nS;
    self.nUniq=self.nUniq+nU;
    if self.nextCol<self.limit then
      self.nextCol=self.nextCol+1;
    else
      self.nextCol=-1;
    end
    return self.nextCol;
  end
  function Info:getTotal() return self.nTotal; end
  function Info:getUnique() return self.nUniq; end
  function Info:getTime() return self:secstotime(os.difftime(os.time(),self.starttime)); end
  function Info:secstotime(secs)
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
  return setmetatable(this,{__index=Info});
end


Thread={}; Thread.new=function()
  local this={
    size=2; nTotal=0;nUniq=0;nEquiv=0; COUNT2=0;COUNT4=0;COUNT8=0;
    board={};colChk={};diagChk={};antiChk={};
  };
  function Thread:Thread(size,info)
    self.size=size; self.info=info;
  end
  function Thread:run()
    local nextCol;
    while(true) do
      nextCol=info:nextJob(self.nTotal,self.nUniq);
      if nextCol<0 then break; end
      self.nTotal,self.nUniq=0,0;
      for row=0,self.size-1,1 do self.board[row]=row; end --テーブルの初期化
      thread:NQueens(0);
    end
  end
  function Thread:NQueens(row)
		local lim=0;
		local vTemp=0;
		if(row<self.size-1) then
		    if( self.diagChk[row-self.board[row]+self.size-1] 
          or self.antiChk[row+self.board[row]])  then
        else
	        self.diagChk[row-self.board[row]+self.size-1],
          self.antiChk[row+self.board[row]]=true,true;
          self:NQueens(row+1);
	        self.diagChk[row-self.board[row]+self.size-1],
          self.antiChk[row+self.board[row]]=false,false;
		    end	
			if(row~=0) then lim=self.size; else lim=(self.size+1)/2; end
			for k=row+1,lim-1,1 do
				vTemp=self.board[k];
				self.board[k]=self.board[row];
				self.board[row]=vTemp;
				if ( self.diagChk[row-self.board[row]+self.size-1] 
          or self.antiChk[row+self.board[row]]) then
        else
		      self.diagChk[row-self.board[row]+self.size-1],
          self.antiChk[row+self.board[row]]=true,true;
          self:NQueens(row+1);
		      self.diagChk[row-self.board[row]+self.size-1],
          self.antiChk[row+self.board[row]]=false,false;
        end
      end
			vTemp=self.board[row];
			for k=row+1,self.size-1,1 do
				self.board[k-1]=self.board[k];
      end
			self.board[self.size-1]=vTemp;
		else
	    if (self.diagChk[row-self.board[row]+self.size-1] 
        or self.antiChk[row+self.board[row]]) then
				return;
      end
			k=self:symmetryOps();
			if (k~=0) then
        self.nTotal=self.nTotal+k;
        self.nUniq=self.nUniq+1;
      end
    end
		return;
  end

  function Thread:intncmp(lt,rt,n)
    local k,rtn=0,0; 
    for k=0,n-1,1 do
      rtn=lt[k]-rt[k];
      if (rtn~=0)then break; end
    end
    return rtn;
  end
  function Thread:rotate(trial,scratch,n,neg) --回転
    local k=0; local incr=0;
    if neg then k=0; else k=n-1; end
    if neg then incr=1; else incr=-1; end 
    local j=0;
    while j<n do scratch[j]=trial[k]; k=k+incr; j=j+1; end
    if neg then k=n-1; else k=0; end 
    local j=0;
    while j<n do trial[scratch[j]]=k; k=k-incr; j=j+1; end
  end
  function Thread:vMirror(check,n) -- 反転
    for j=0,n-1,1 do
      check[j]=(n-1)-check[j];
    end
  end
  function Thread:symmetryOps()
    local trial={}; 	local scratch={}; local k=0;
    for k=0,self.size-1,1 do trial[k]=self.board[k]; end --テーブルの初期化
    --回転
    self:rotate(trial,scratch,self.size,nil);-- 時計回りに90度回転 self.nEquiv=1
    k=self:intncmp(self.board,trial,self.size);
    if(k>0)then return 0; end
    if(k==0)then self.nEquiv=1; else 
      self:rotate(trial,scratch,self.size,nil);-- 時計回りに180度回転 self.nEquiv=2
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      if(k==0)then self.nEquiv=2; else 
        self:rotate(trial,scratch,self.size,nil);-- 時計回りに270度回転 self.nEquiv=4
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
        self.nEquiv=4;
      end
    end  
    -- 反転
    self:vMirror(trial,self.size);	
    k=self:intncmp(self.board,trial,self.size);
    if(k>0) then return 0; end
    if (self.nEquiv>1) then 		
      self:rotate(trial,scratch,self.size,true);-- 90度反転 対角鏡と同等 return 0				
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      if(self.nEquiv>2)then    
        self:rotate(trial,scratch,self.size,true);-- 180度反転 水平鏡像と同等 return 0
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
        self:rotate(trial,scratch,self.size,true);-- 270度反転 反対角鏡と同等 return 0			
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
      end 
    end
    return self.nEquiv * 2;
  end
  return setmetatable(this,{__index=Thread});
end


NQueen={}; NQueen.new=function()
  local this={ 
    max=17; size=2; 
  };
  function NQueen:NQueen()
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,self.max,1 do
      info=Info.new(); 
      info:resetCount();
      thread=Thread.new();
      thread:Thread(size,info);   
      thread:run();
      print(string.format("%2d:%17d%13d%12s", size,info:getTotal(),info:getUnique(),info:getTime())); 
    end
  end
  return setmetatable(this,{__index=NQueen} );
end

NQueen.new():NQueen();

