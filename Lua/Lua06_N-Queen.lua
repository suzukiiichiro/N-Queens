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
 *<>３．バックトラック                   NQueen2()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *  ５．枝刈りと最適化                   NQueen5()
 *  ６．スレッド                         NQueen6()
 *  ７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド                   NQueen8()
*/

/**
 * ３．バックトラック
 *  　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 *  さない事が明らかな場合は、それ以降のパターン生成を行わない。
 *  　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 *  　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 *  は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 *  けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 *  を降りても解がないと判明した時点で木を引き返すということができます。
 */

  /**
   * 実行結果 Bash
	 N:        Total       Unique        hh:mm
	 2:            0            0            0
	 3:            0            0            0
	 4:            2            0            0
	 5:           10            0            0
	 6:            4            0            0
	 7:           40            0            0
	 8:           92            0            1
	 9:          352            0            1
	10:          724            0            7
	11:         2680            0           33
	12:        14200            0          183

   * 実行結果 java
   N:            Total       Unique    hh:mm:ss
   2:                0            0  00:00:00
   3:                0            0  00:00:00
   4:                2            0  00:00:00
   5:               10            0  00:00:00
   6:                4            0  00:00:00
   7:               40            0  00:00:00
   8:               92            0  00:00:00
   9:              352            0  00:00:00
  10:              724            0  00:00:00
  11:             2680            0  00:00:00
  12:            14200            0  00:00:00
  13:            73712            0  00:00:00
  14:           365596            0  00:00:02
  15:          2279184            0  00:00:14
  16:         14772512            0  00:01:35
  */

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
13:            73712            0    00:00:00
14:           365596            0    00:00:04
]]--

NQueen={}; NQueen.new=function()
  
  local this={
    size=0;
    TOTAL=0;
    UNIQUE=0;
    MASK=0;
    min=0;
    left=0;
    down=0;
    right=0;
  };

  function NQueen:NQueens(row,left,down,right) 
    local bitmap=0;
    local BIT=0;
    if row==self.size then
      self.TOTAL=self.TOTAL+1 ;
    else
      bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right),self.size-1));
      --print(string.format("bitmap:%d",bitmap)); 
      while bitmap>0 do
        BIT=bit.band(-bitmap,bitmap);
        --print(string.format("bitmap:%d",bitmap)); 
        bitmap=bit.bxor(bitmap,BIT);
        self:NQueens(row+1,bit.lshift(bit.bor(left,BIT),1),bit.bor(down,BIT),bit.rshift(bit.bor(right,BIT),1));
      end
    end
  end

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
  return setmetatable( this,{__index=NQueen} );
end
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

NQueen.new():NQueen();

