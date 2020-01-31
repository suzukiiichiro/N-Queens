#!/usr/bin/env luajit

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 
  ３．バックトラック
   　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
   さない事が明らかな場合は、それ以降のパターン生成を行わない。
   　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
   　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
   は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
   けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
   を降りても解がないと判明した時点で木を引き返すということができます。


  N-Queen の データ配列について
  =============================

  総当たり
  結局全部のケースをやってみる（完全解）

  バックトラック
  とりあえずやってみる。ダメなら戻って別の道を探る


  N-Queen: クイーンの効き筋
  =========================
  　　　 ___________________
        |-*-|---|---|-*-|---|
        +-------------------+
        |---|-*-|---|-*-|---|
        +-------------------+
        |---|---|-*-|-*-|-*-|
        +-------------------+
        |-*-|-*-|-*-|-Q-|-*-|
        +-------------------+
        |---|---|-*-|-*-|-*-|
        +-------------------+


  N-Queen: 盤面上で互いのクイーンが効き筋にならないように配置
  ===========================================================
  　　　 ___________________
        |-Q-|---|---|---|---|
        +-------------------+
        |---|---|---|-Q-|---|
        +-------------------+
        |---|-Q-|---|---|---|
        +-------------------+
        |---|---|---|---|-Q-|
        +-------------------+
        |---|---|-Q-|---|---|
        +-------------------+


  盤面をデータ構造で表す
  ======================

     row = { 0, 3, 1, 4, 2 };

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________
    0   |-Q-|---|---|---|---|
        +-------------------+
    1   |---|---|---|-Q-|---|
        +-------------------+
    2   |---|-Q-|---|---|---|
        +-------------------+
    3   |---|---|---|---|-Q-|
        +-------------------+
    4   |---|---|-Q-|---|---|
        +-------------------+


  効き筋の表現
  ============

  　すでに効き筋：FALSE(盤面ではF）
  　配置可能    ：TRUE

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________
    0   |---|-F-|---|-F-|---|
        +-------------------+
    1   |-F-|-F-|-F-|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-|
        +-------------------+
    3   |-F-|-F-|-F-|---|---|
        +-------------------+
    4   |---|-F-|---|-F-|---|
        +-------------------+


  効き筋を三つの配列で表現
  ========================

  ■配列1：q_row[row]

  そのrow(行)にQueenがいる場合はFALSE
                      いない場合はTRUE

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________
    0   |---|---|---|---|---|
        +-------------------+
    1   |---|---|---|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-|   q_row[row]==false
        +-------------------+
    3   |---|---|---|---|---|
        +-------------------+
    4   |---|---|---|---|---|
        +-------------------+

  ■配列２：q_se[col-row+N-1]

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________
    0   |---|---|---|---|---|
        +-------------------+
    1   |-F-|---|---|---|---|
        +-------------------+
    2   |---|-Q-|---|---|---|
        +-------------------+
    3   |---|---|-F-|---|---|
        +-------------------+
    4   |---|---|---|-F-|---|
        +-------------------+
                      q_se[col-row+N-1]==F

  ■配列3：q_sw[col+row]

  　　       column(列)
          0   1   2   3   4
  row(行)
                      q_sw[col+row]==F
         ___________________
    0   |---|---|---|-F-|---|
        +-------------------+
    1   |---|---|-F-|---|---|
        +-------------------+
    2   |---|-Q-|---|---|---|
        +-------------------+
    3   |-F-|---|---|---|---|
        +-------------------+
    4   |---|---|---|---|---|
        +-------------------+

  ■aBoard[row]=col
   クイーンの位置

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________
    0   |---|---|---|---|---|
        +-------------------+
    1   |---|---|---|---|---|
        +-------------------+
    2   |---|-Q-|---|---|---| aBoard[row]=col
        +-------------------+
    3   |---|---|---|---|---|
        +-------------------+
    4   |---|---|---|---|---|
        +-------------------+


  考え方：１
  ==========
  row=0, col=0 にクイーンを配置してみます。

  aBoard[row]=col
     ↓
  aBoard[0]=0;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  配列構造では
    0   |-Q-|---|---|---|---|  aBoard[]={0,,,,}
        +-------------------+
    1   |---|---|---|---|---|
        +-------------------+
    2   |---|---|---|---|---| 
        +-------------------+
    3   |---|---|---|---|---|
        +-------------------+
    4   |---|---|---|---|---|
        +-------------------+


  考え方：２
  ==========
  効き筋を埋めます

  aBoard[row]=col
     ↓
  aBoard[0]=0;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  配列構造で 
    0   |-Q-|-F-|-F-|-F-|-F-|  aBoard[]={0,,,,}  
        +-------------------+
    1   |---|-F-|---|---|---|
        +-------------------+
    2   |---|---|-F-|---|---| 
        +-------------------+
    3   |---|---|---|-F-|---|
        +-------------------+
    4   |---|---|---|---|-F-|
        +-------------------+


  考え方：３
  ==========
  次の候補は以下のＡ，Ｂ，Ｃとなります

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=;
  aBoard[3]=;
  aBoard[4]=;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  配列構造で 
    0   |-Q-|-F-|-F-|-F-|-F-|  aBoard[]={0,,,,}  
        +-------------------+
    1   |---|-F-|---|---|---|
        +-------------------+
    2   |---|-A-|-F-|---|---| 
        +-------------------+
    3   |---|-B-|---|-F-|---|
        +-------------------+
    4   |---|-C-|---|---|-F-|
        +-------------------+

  考え方：４
  ==========
  Ａにおいてみます。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  配列構造で 
    0   |-Q-|-F-|-F-|-F-|-F-|  aBoard[]={0,,1,,}  
        +-------------------+
    1   |---|-F-|---|---|---|
        +-------------------+
    2   |---|-Q-|-F-|---|---| 
        +-------------------+
    3   |---|---|---|-F-|---|
        +-------------------+
    4   |---|---|---|---|-F-|
        +-------------------+

  考え方：５
  ==========
  効き筋は以下の通りです。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  配列構造で 
    0   |-Q-|-F-|-F-|-F-|-F-|  aBoard[]={0,,1,,}  
        +-------------------+
    1   |-F-|-F-|-F-|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+
    4   |---|---|---|-F-|-F-|
        +-------------------+

  考え方：６
  ==========
  次の候補はＡとなります

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+
    1   |-F-|-F-|-F-|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+ 配列構造で 
    4   |---|---|-A-|-F-|-F-|  aBoard[]={0,,1,,2} 
        +-------------------+

  考え方：７
  ==========
  効き筋は以下の通りです。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+
    1   |-F-|-F-|-F-|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+ 配列構造で 
    4   |---|---|-Q-|-F-|-F-|  aBoard[]={0,,1,,2} 
        +-------------------+

  考え方：８
  ==========
  効き筋は以下の通りです。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+
    1   |-F-|-F-|-F-|---|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+ 配列構造で 
    4   |---|---|-Q-|-F-|-F-|  aBoard[]={0,,1,,2} 
        +-------------------+

  考え方：９
  ==========
  次の候補はＡとなります

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=3;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+
    1   |-F-|-F-|-F-|-A-|---|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+ 配列構造で 
    4   |---|---|-Q-|-F-|-F-|  aBoard[]={0,,1,,2} 
        +-------------------+

  考え方：１０
  ==========
  今回は、うまくいっていますが、
  次の候補がなければ、キャンセルして、
  前のコマを次の候補にコマを移動し、
  処理を継続します。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=3;
  aBoard[2]=1;
  aBoard[3]=;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+ 配列構造で 
    1   |-F-|-F-|-F-|-Q-|-F-|  aBoard[]={0,3,1,,2}
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|---|
        +-------------------+ 
    4   |---|---|-Q-|-F-|-F-|  
        +-------------------+


  考え方：１１
  ==========
  最後のクイーンをおきます

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=3;
  aBoard[2]=1;
  aBoard[3]=4;
  aBoard[4]=2;

  　　       column(列)
          0   1   2   3   4
  row(行)
         ___________________  
    0   |-Q-|-F-|-F-|-F-|-F-|  
        +-------------------+
    1   |-F-|-F-|-F-|-Q-|-F-|
        +-------------------+
    2   |-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
    3   |-F-|---|-F-|-F-|-Q-|
        +-------------------+ 配列構造で 
    4   |---|---|-Q-|-F-|-F-|  aBoard[]={0,3,1,4,2} 
        +-------------------+


  考え方：１２
  ==========
  rowの脇にcolの位置を示します。

  aBoard[row]=col
     ↓
  aBoard[0]=0;
  aBoard[1]=3;
  aBoard[2]=1;
  aBoard[3]=4;
  aBoard[4]=2;

  　　       column(列)
           0   1   2   3   4
  row(行)
          ___________________  
    0[0] |-Q-|-F-|-F-|-F-|-F-|  
         +-------------------+
    1[3] |-F-|-F-|-F-|-Q-|-F-|
         +-------------------+
    2[1] |-F-|-Q-|-F-|-F-|-F-| 
         +-------------------+
    3[4] |-F-|---|-F-|-F-|-Q-|
         +-------------------+ 配列構造で 
    4[2] |---|---|-Q-|-F-|-F-|  aBoard[]={0,3,1,4,2} 
         +-------------------+


 実行結果
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
14:           365596            0    00:00:04
15:          2279184            0    00:00:26
16:         14772512            0    00:02:55
17:         95815104            0    00:20:48
]]--

NQueen={}; NQueen.new=function()
  -- 
  local this={                        --クラス変数
    SIZE=0;
    TOTAL=0;
    UNIQUE=0;
    colChk={};
    diagChk={};
    antiChk={};
    board={};
  };
  --
  function NQueen:NQueens(row)        --メインロジックメソッド
    if row==self.SIZE then
      self.TOTAL=self.TOTAL+1 ;
    else
      for col=0,self.SIZE-1,1 do
        self.board[row]=col; 
        if	self.colChk[col]==nil 
          and self.antiChk[row+col]==nil 
          and self.diagChk[row-col+(self.SIZE-1)]==nil then
          self.colChk[col],
          self.diagChk[row-self.board[row]+self.SIZE-1],
          self.antiChk[row+self.board[row]]=true,true,true;
          self:NQueens(row+1);
          self.colChk[col],
          self.diagChk[row-self.board[row]+self.SIZE-1],
          self.antiChk[row+self.board[row]]=nil,nil,nil;
        end
      end
    end
  end
  --
  function NQueen:secstotime(secs)    --計測時間処理メソッド
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
  function NQueen:NQueen()
    local max=27;
    print(" N:            Total       Unique    hh:mm:ss");
    for si=2,max,1 do
      self.SIZE=si;                 --初期化
      self.TOTAL=0;
      self.UNIQUE=0;
      s=os.time();                  --計測開始
      self:NQueens(0);
      print(string.format("%2d:%17d%13d%12s",si,self.TOTAL,0,self:secstotime(os.difftime(os.time(),s))));            --計測終了とともに出力 
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen();              --実行

