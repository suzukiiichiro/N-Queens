#!/usr/bin/env luajit

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  ６．バックトラック＋ビットマップ

   ビット演算を使って高速化 状態をビットマップにパックし、処理する
   単純なバックトラックよりも２０〜３０倍高速
 
 　ビットマップであれば、シフトにより高速にデータを移動できる。
  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
  分。

 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
 　バックトラックよりも２０−３０倍高速。
 
 ===================
 考え方 1
 ===================

 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
 つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
 する)
 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
 ずれかのビット位置をひとつだけONにして進めていきます。

 
- - - - - Q - -    00000100 0番目のビットフィールド
- - - Q - - - -    00010000 1番目のビットフィールド
- - - - - - Q -    00000010 2番目のビットフィールド
Q - - - - - - -    10000000 3番目のビットフィールド
- - - - - - - Q    00000001 4番目のビットフィールド
- Q - - - - - -    01000000 5番目のビットフィールド
- - - - Q - - -    00001000 6番目のビットフィールド
- - Q - - - - -    00100000 7番目のビットフィールド


 ===================
 考え方 2
 ===================

 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。

 1. 左下に効き筋が進むもの: left 
 2. 真下に効き筋が進むもの: down
 3. 右下に効き筋が進むもの: right

次に、斜めの利き筋を考えます。
 上図の場合、
 1列目の右斜め上の利き筋は 3 番目 (0x08)
 2列目の右斜め上の利き筋は 2 番目 (0x04) になります。
 この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
 ことができます。
 また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目 (0x40)
になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。

つまり、右シフトの利き筋を right、左シフトの利き筋を left で表すことで、クイー
ンの効き筋はrightとleftを1 ビットシフトするだけで求めることができるわけです。

  *-------------
  | . . . . . .
  | . . . -3. .  0x02 -|
  | . . -2. . .  0x04  |(1 bit 右シフト right)
  | . -1. . . .  0x08 -|
  | Q . . . . .  0x10 ←(Q の位置は 4   down)
  | . +1. . . .  0x20 -| 
  | . . +2. . .  0x40  |(1 bit 左シフト left)  
  | . . . +3. .  0x80 -|
  *-------------
  図：斜めの利き筋のチェック

 n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
 の３つのビットフィールドとn番目のビットフィールド(bit)とのOR演算をそれぞれ行
 います。leftは左にひとつシフトし、downはそのまま、rightは右にひとつシフトして
 n+1番目のビットフィールド探索に渡してやります。

 left : (left |bit)<<1
 right: (right|bit)>>1
 down :   down|bit


 ===================
 考え方 3
 ===================

   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
 ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
 できない位置ということになります。次にその３つのビットフィールドをORしたビッ
 トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
 ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
 次の演算を行なってみます。
 
 bit = -bitmap & bitmap; //一番右のビットを取り出す
 
   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
 に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
 と次のようになります。
 
  00000011   3
  00000010   2
  00000001   1
  00000000   0
  11111111  -1
  11111110  -2
  11111101  -3
 
   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
 によって1ビット抽出を実現させていることが重要です。
 
      00010110   22
  AND 11101010  -22
 ------------------
      00000010
 
   さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
 生成されることになります。
 
 while (bitmap) {
     bit = -bitmap & bitmap;
     bitmap ^= bit;
     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
 }
 
   * 実行結果 luajit
 N:            Total       Unique    hh:mm:ss
 2:                0            0    00:00:00
 3:                0            0    00:00:00
 4:                2            0    00:00:00
 5:               10            0    00:00:00
 6:                4            0    00:00:00
 7:               40            0    00:00:00
 8:               92            0    00:00:00
 9:              352            0    00:00:00
10:              724            0    00:00:00
11:             2680            0    00:00:00
12:            14200            0    00:00:00
13:            73712            0    00:00:01
14:           365596            0    00:00:02
15:          2279184            0    00:00:11
16:         14772512            0    00:01:17
17:         95815104            0    00:09:05
]]--
--
NQueen={}; NQueen.new=function()
  -- 
  local this={
    size=0;
    TOTAL=0;
    UNIQUE=0;
    MASK=0;
  };
  --ビット反転させるメソッド・・・
  function NQueen:rbits(byte,sz)
    local score=0;
    for i=sz,0,-1 do
    --io.write(bit.bnot(bit.band(bit.arshift(byte,i), 1)))
      if bit.band(bit.arshift(byte,i), 1) ==0 then
        score=score+2^i;
      end
    end
    return score;
  end
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
  function NQueen:NQueens(min,left,down,right) 
    local bitmap=0;
    local BIT=0;
    if min==self.size then
      self.TOTAL=self.TOTAL+1 ;
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
  function NQueen:NQueen()
    local max=24;
    print(" N:            Total       Unique    hh:mm:ss");
    for si=2,max,1 do
      self.size=si;
      self.TOTAL=0;
      self.UNIQUE=0;
      self.MASK=bit.lshift(1,self.size)-1;    
      s=os.time();
      self:NQueens(0,0,0,0);
      print(string.format("%2d:%17d%13d%12s",si,self.TOTAL,0,self:secstotime(os.difftime(os.time(),s)))); 
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen();

