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


NQueen={}; NQueen.new=function()
  -- 
  local this={
    size=0;
    TOTAL=0;
    UNIQUE=0;
    MASK=0;
    --nTotal=0;nUniq=0;nEquiv=0; 
    nEquiv=0;
    COUNT2=0;COUNT4=0;COUNT8=0;
    board={};colChk={};diagChk={};antiChk={};
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
  function NQueen:intncmp(lt,rt,n)
    local k,rtn=0,0; 
    for k=0,n-1,1 do
      rtn=lt[k]-rt[k];
      if (rtn~=0)then break; end
    end
    return rtn;
  end
  --
  function NQueen:rotate(trial,scratch,n,neg) --回転
    local k=0; local incr=0;
    if neg then k=0; else k=n-1; end
    if neg then incr=1; else incr=-1; end 
    local j=0;
    while j<n do scratch[j]=trial[k]; k=k+incr; j=j+1; end
    if neg then k=n-1; else k=0; end 
    local j=0;
    while j<n do trial[scratch[j]]=k; k=k-incr; j=j+1; end
  end
  --
  function NQueen:vMirror(check,n) -- 反転
    for j=0,n-1,1 do
      check[j]=(n-1)-check[j];
    end
  end
  --
  function NQueen:symmetryOps()
    local trial={};local scratch={}; local k=0;
    for k=0,self.size-1,1 do 
      trial[k]=self.board[k]; 
      scratch[k]=self.board[k];
    end --テーブルの初期化
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
  --
  function NQueen:NQueens(min,left,down,right) 
    local bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right ),self.size-1));
    local BIT=0;
    local k=0;
    if min==self.size then
      self.TOTAL=self.TOTAL+1 ;
       --[[
      -- if bitmap~=0 then
        self.board[min]=bitmap;
        k=self:symmetryOps();
        if (k~=0) then
          self.TOTAL=self.TOTAL+k;
          self.UNIQUE=self.UNIQUE+1;
        end
      -- end
       ]]--
    else
      bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right),self.size-1));
      while bitmap~=0 do
        BIT=bit.band(-bitmap,bitmap);
        bitmap=bit.bxor(bitmap,BIT);
        self:NQueens(min+1,bit.lshift(bit.bor(left,BIT),1),bit.bor(down,BIT),bit.rshift(bit.bor(right,BIT),1));
      end
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
    local max=24;
    print(" N:            Total       Unique    hh:mm:ss");
    for si=2,max,1 do
      self.size=si;
      self.TOTAL=0;
      self.UNIQUE=0;
      self.nEquiv=0;
      for k=0,self.size-1,1 do self.board[k]=k; end --テーブルの初期化
      self.MASK=bit.lshift(1,self.size)-1;    
      s=os.time();
      self:NQueens(0,0,0,0);
      print(string.format("%2d:%17d%13d%12s",si,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  return setmetatable( this,{__index=NQueen} );
end
  --
NQueen.new():NQueen();
