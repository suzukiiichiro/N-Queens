#!/usr/bin/env luajit

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
 ７．バックトラック＋ビットマップ＋対称解除法

     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 
 ■ユニーク解の判定方法
   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 ことを意味しています。
 
   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 わりません。
 
   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 下の例では要素数が２個のものと８個のものがあります。


 Ｎ＝５の全解は１０、ユニーク解は２なのです。
 
 グループ１: ユニーク解１つ目
 - - - Q -   - Q - - -
 Q - - - -   - - - - Q
 - - Q - -   - - Q - -
 - - - - Q   Q - - - -
 - Q - - -   - - - Q -
 
 グループ２: ユニーク解２つ目
 - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -

 
   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 と呼ぶことにします。
 
 - - - - Q   0
 - - Q - -   2
 Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 - - - Q -   1
 - Q - - -   3
 
 
   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 変換を試みるだけでユニーク解の判定が可能になります。
  
 
 ■ユニーク解の個数を求める
   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 央になければならず、それは互いの効き筋にあたるので有り得ません。

 ■ユニーク解から全解への展開
   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 いう考察が必要になってきます。
 
   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 ず８個(＝２×４)になります。
 
   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 ために次の事柄を確認します。

 TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
    ジナルと同型になる。  

    COUNT2 * 2
 
   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
    り得る。 

    COUNT4 * 4
 
   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 
    COUNT8 * 8 

   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 りを外す必要がなくなったというわけです。 
 
   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)

 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 ですが、今回の処理を行うことによって、さらに、処理スピードが飛躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。



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

