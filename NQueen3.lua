#!/bin/lua

--[[import org.apache.commons.lang3.time.DurationFormatUtils;

/**
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
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
 *  １．ブルートフォース（力まかせ探索） NQueen1() * N 8: 00:04:15
 *  ２．バックトラック                   NQueen2() * N 8: 00:00:01
 *<>３．配置フラグ（制約テスト高速化）   NQueen3() * N16: 00:01:35
 *  ４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 *  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 *  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 *  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 *  ８．マルチスレッド2                  NQueen8() * N16: 00:00:00
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
   * 実行結果
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
]]--
function NQueen()
	print(" N:            Total       Unique    hh:mm:ss");
  for size=2,max-1,1 do
      print(size);
			TOTAL=0;
			for k=0,size-1,1 do
        board[k]=k ;
      end
      s=os.clock();
			nQueens(0);
      e=os.clock();
      print(e-s);
	end
end
function nQueens(row)
		if(row==size)then
			TOTAL=TOTAL+1 ;
		else
			for col=0,size-1,1 do
				board[row]=col; 
				if colChk[col]==false and antiChk[row+col]==false and diagChk[row-col+(size-1)]==false then
					colChk[col]=true;
          diagChk[row-board[row]+size-1] =true;
          antiChk[row+board[row]] = true;
					nQueens(row+1);
					colChk[col]=false;
          diagChk[row-board[row]+size-1] =false;
          antiChk[row+board[row]] =false;
        end
      end
    end
end
function main()
  NQueen();
end
board={};
TOTAL=nil;
size=nil;
max=nil;
colChk={};    --// セル
diagChk={};   --// 対角線
antiChk={};   --// 反対角線
max=8 ;
main();
