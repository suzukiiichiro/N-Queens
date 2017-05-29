#!/usr/bin/env luajit

--[[
/**
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 * Java版 N-Queen
 * https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
 * Bash版 N-Queen
 * https://github.com/suzukiiichiro/AI_Algorithm_Bash
 * Lua版  N-Queen
 * https://github.com/suzukiiichiro/AI_Algorithm_Lua
 * 
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
 *  ５．枝刈りと最適化                   NQueen5()
 *  ６．スレッド                         NQueen6()
 *<>７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド                   NQueen8()
*/

/**
 * ７．ビットマップ
 *
 *   ビット演算を使って高速化 状態をビットマップにパックし、処理する
 *   単純なバックトラックよりも２０〜３０倍高速
 * 
 * 　ビットマップであれば、シフトにより高速にデータを移動できる。
 *  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
 *  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
 *  分。
 *
 * 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
 * 　バックトラックよりも２０−３０倍高速。
 * 
 * ===================
 * 考え方 1
 * ===================
 *
 * 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
 * つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
 * する)
 * 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
 * ずれかのビット位置をひとつだけONにして進めていきます。
 *
 * 
 *- - - - - Q - -    00000100 0番目のビットフィールド
 *- - - Q - - - -    00010000 1番目のビットフィールド
 *- - - - - - Q -    00000010 2番目のビットフィールド
 *Q - - - - - - -    10000000 3番目のビットフィールド
 *- - - - - - - Q    00000001 4番目のビットフィールド
 *- Q - - - - - -    01000000 5番目のビットフィールド
 *- - - - Q - - -    00001000 6番目のビットフィールド
 *- - Q - - - - -    00100000 7番目のビットフィールド
 *
 *
 * ===================
 * 考え方 2
 * ===================
 *
 * 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。
 *
 * 1. 左下に効き筋が進むもの: l 
 * 2. 真下に効き筋が進むもの: d
 * 3. 右下に効き筋が進むもの: r
 *
 *次に、斜めの利き筋を考えます。
 * 上図の場合、
 * 1列目の右斜め上の利き筋は 3 番目 (0x08)
 * 2列目の右斜め上の利き筋は 2 番目 (0x04) になります。
 * この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
 * ことができます。
 * また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目 (0x40)
 *になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。
 *
 *つまり、右シフトの利き筋を r、左シフトの利き筋を l で表すことで、クイー
 *ンの効き筋はrとlを1 ビットシフトするだけで求めることができるわけです。
 *
 *  *-------------
 *  | . . . . . .
 *  | . . . -3. .  0x02 -|
 *  | . . -2. . .  0x04  |(1 B 右シフト r)
 *  | . -1. . . .  0x08 -|
 *  | Q . . . . .  0x10 ←(Q の位置は 4   d)
 *  | . +1. . . .  0x20 -| 
 *  | . . +2. . .  0x40  |(1 B 左シフト l)  
 *  | . . . +3. .  0x80 -|
 *  *-------------
 *  図：斜めの利き筋のチェック
 *
 * n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
 * の３つのビットフィールドとn番目のビットフィールド(B)とのOR演算をそれぞれ行
 * います。lは左にひとつシフトし、dはそのまま、rは右にひとつシフトして
 * n+1番目のビットフィールド探索に渡してやります。
 *
 * l : (l |B)<<1
 * r: (r|B)>>1
 * d :   d|B
 *
 *
 * ===================
 * 考え方 3
 * ===================
 *
 *   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
 * ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
 * できない位置ということになります。次にその３つのビットフィールドをORしたビッ
 * トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
 * ルド」に変換します。そしてこの配置可能なビットフィールドを BM と呼ぶとして、
 * 次の演算を行なってみます。
 * 
 * B = -BM & BM; //一番右のビットを取り出す
 * 
 *   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
 * に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
 * と次のようになります。
 * 
 *  00000011   3
 *  00000010   2
 *  00000001   1
 *  00000000   0
 *  11111111  -1
 *  11111110  -2
 *  11111101  -3
 * 
 *   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
 * 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
 * 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
 * によって1ビット抽出を実現させていることが重要です。
 * 
 *      00010110   22
 *  AND 11101010  -22
 * ------------------
 *      00000010
 * 
 *   さて、そこで下のようなwhile文を書けば、このループは BM のONビットの数の
 * 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
 * 生成されることになります。
 * 
 * while (BM) {
 *     B = -BM & BM;
 *     BM ^= B;
 *     //ここでは配置可能なパターンがひとつずつ生成される(B) 
 * }
 */

   /**
    * 実行結果 Java版
   N:            Total       Unique    hh:mm:ss
   2:                0            0  00:00:00
   3:                0            0  00:00:00
   4:                2            1  00:00:00
   5:               10            2  00:00:00
   6:                4            1  00:00:00
   7:               40            6  00:00:00
   8:               92           12  00:00:00
   9:              352           46  00:00:00
  10:              724           92  00:00:00
  11:             2680          341  00:00:00
  12:            14200         1787  00:00:00
  13:            73712         9233  00:00:00
  14:           365596        45752  00:00:00
  15:          2279184       285053  00:00:00
  16:         14772512      1846955  00:00:02
	17:         95815104     11977939  00:00:15
	18:        666090624     83263591  00:01:49
	19:       4968057848    621012754  00:13:55
	20:      39029188884   4878666808  01:50:42
	21:     314666222712  39333324973  15:34:05
	22:    2691008701644 336376244042 136:08:43
  */

    * 実行結果 Lua版
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
14:           365596        45752    00:00:00
15:          2279184       285053    00:00:03
16:         14772512      1846955    00:00:20
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
    local max=16;
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

