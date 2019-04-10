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

