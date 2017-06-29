#!/usr/bin/env luajit

--[[
/**
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 *
 * エイト・クイーンについて
 * https://ja.wikipedia.org/wiki/エイト・クイーン
 *
 * N-Queens問題とは
 *    Nクイーン問題とは、「8列×8行のチェスボードに8個のクイーンを、互いに効きが
 *    当たらないように並べよ」という８クイーン問題のクイーン(N)を、どこまで大き
 *    なNまで解を求めることができるかという問題。
 *    クイーンとは、チェスで使われているクイーンを指し、チェス盤の中で、縦、横、
 *    斜めにどこまでも進むことができる駒で、日本の将棋でいう「飛車と角」を合わ
 *    せた動きとなる。８列×８行で構成される一般的なチェスボードにおける8-Queens
 *    問題の解は、解の総数は92個である。比較的単純な問題なので、学部レベルの演
 *    習問題として取り上げられることが多い。
 *    8-Queens問題程度であれば、人力またはプログラムによる「力まかせ探索」でも
 *    解を求めることができるが、Nが大きくなると解が一気に爆発し、実用的な時間で
 *    は解けなくなる。
 *    現在すべての解が判明しているものは、2004年に電気通信大学で264CPU×20日をか
 *    けてn=24を解決し世界一に、その後2005 年にニッツァ大学でn=25、2016年にドレ
 *    スデン工科大学でn=27の解を求めることに成功している。
 *
 * 目次
 *  Nクイーン問題
 *  １．ブルートフォース（力まかせ探索） NQueen1() 
 *  ２．配置フラグ（制約テスト高速化）   NQueen3()
 *  ３．バックトラック                   NQueen2()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *<>５．枝刈りと最適化                   NQueen5()
 *  ６．スレッド                         NQueen6()
 *  ７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド                   NQueen8()
 *
 *
 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。
 *
 * 実行結果 Bash
   N:            Total       Unique   hh:mm:ss
   2:                0            0          0
   3:                0            0          0
   4:                2            0          0
   5:               10            0          0
   6:                4            0          0
   7:               40            0          0
   8:               92            0          0
   9:              352            0          1
  10:              724            0          3
  11:             2680            0         14
  12:            14200            0         71
  13:            73712            0        392
  
 実行結果 Java
   N:            Total       Unique    hh:mm:ss
   2:                0            0   00:00:00
   3:                0            0   00:00:00
   4:                2            1   00:00:00
   5:               10            2   00:00:00
   6:                4            1   00:00:00
   7:               40            6   00:00:00
   8:               92           12   00:00:00
   9:              352           46   00:00:00
  10:              724           92   00:00:00
  11:             2680          341   00:00:00
  12:            14200         1787   00:00:00
  13:            73712         9233   00:00:00
  14:           365596        45752   00:00:00
  15:          2279184       285053   00:00:03
  16:         14772512      1846955   00:00:24
  */

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
13:            73712         9233    00:00:00
14:           365596        45752    00:00:02
15:          2279184       285053    00:00:16
]]--

NQueen={}; NQueen.new=function()

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

  function NQueen:intncmp(lt,rt,n)
    local k,rtn=0,0; 
    for k=0,n-1,1 do
      rtn=lt[k]-rt[k];
      if (rtn~=0)then break; end
    end
    return rtn;
  end

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

  function NQueen:vMirror(check,n)
    for j=0,n-1,1 do
      check[j]=(n-1)-check[j];
    end
  end

  function NQueen:symmetryOps()
	  local trial={}; 	--//回転・反転の対称チェック
	  local scratch={};
    self.nEquiv=0; 				--グローバル初期化
    for k=0,self.size-1,1 do trial[k]=self.board[k]; end
    self:rotate(trial,scratch,self.size,nil);--//時計回りに90度回転
    local k=self:intncmp(self.board,trial,self.size);
    if(k>0)then return 0; end
    if(k==0)then self.nEquiv=1; else
      self:rotate(trial,scratch,self.size,nil);--//時計回りに180度回転
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      if(k==0)then self.nEquiv=2; else
        self:rotate(trial,scratch,self.size,nil);--//時計回りに270度回転
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
        self.nEquiv=4;
      end
    end  
    for k=0,self.size-1,1 do trial[k]=self.board[k]; end
    self:vMirror(trial,self.size);	--//垂直反転
    k=self:intncmp(self.board,trial,self.size);
    if(k>0) then return 0; end
    if (self.nEquiv>1) then 		--// 4回転とは異なる場合
      self:rotate(trial,scratch,self.size,true);				-- 90度回転 対角鏡と同等
      k=self:intncmp(self.board,trial,self.size);
      if(k>0) then return 0; end
      if(self.nEquiv>2)then    --// 2回転とは異なる場合
        self:rotate(trial,scratch,self.size,true);			-- 180度回転 水平鏡像と同等
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
        self:rotate(trial,scratch,self.size,true);			-- 270度回転 反対角鏡と同等
        k=self:intncmp(self.board,trial,self.size);
        if(k>0) then return 0; end
      end 
    end
    return self.nEquiv * 2;
  end

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

  function NQueen:NQueens(row)
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
				self.nUnique=self.nUnique+1;
				self.nTotal=self.nTotal+k;
      end
    end
		return;
  end

  function NQueen:NQueen()
	  local max=27 ; -- //最大N
    print(" N:            Total       Unique    hh:mm:ss");
  for si=2,max-1,1 do
      self.size=si;
		  self.nUnique,self.nTotal=0,0;
		  self.board,self.colChk,self.diagChk,self.antiChk={},{},{},{};
	    for k=0,self.size-1,1 do self.board[k]=k ; end
      for k=0,self.size-1,1 do
        self.board[k]=k;
      end
      s=os.time();
      self:NQueens(0);
      print(string.format("%2d:%17d%13d%12s",
												si,self.nTotal,self.nUnique,
												self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  return setmetatable( this,{__index=NQueen} );
end

NQueen.new():NQueen();

