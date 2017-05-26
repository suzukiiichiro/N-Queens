#!/usr/bin/env luajit

--[[ /**
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
 *<>１．ブルートフォース（力まかせ探索） NQueen1() 
 *  ２．バックトラック                   NQueen2()
 *  ３．配置フラグ（制約テスト高速化）   NQueen3()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *  ５．枝刈りと最適化                   NQueen5()
 *  ６．スレッド                         NQueen6()
 *  ７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド                   NQueen8()
*/

/** 
 * １．ブルートフォース（力まかせ探索）
 *　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
 *  なるかをチェックする方法
 *  (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
 *  けであって8王妃問題を解いているわけではありません
  :
  :
  7 7 7 7 7 7 6 7 : 16777208
  7 7 7 7 7 7 7 0 : 16777209
  7 7 7 7 7 7 7 1 : 16777210
  7 7 7 7 7 7 7 2 : 16777211
  7 7 7 7 7 7 7 3 : 16777212
  7 7 7 7 7 7 7 4 : 16777213
  7 7 7 7 7 7 7 5 : 16777214
  7 7 7 7 7 7 7 6 : 16777215
  7 7 7 7 7 7 7 7 : 16777216
  */
]]--

NQueen={}; NQueen.new=function()

  local this={
    board={};
    size=8;
    count=1;
  };

  function NQueen:display()
      for col=0,self.size-1,1 do
        io.write(string.format('%2d', self.board[col]));
      end
      print(" : "..self.count);
      self.count=self.count+1;
  end

  function NQueen:NQueen(row) 
      if row==self.size then --全列に配置完了 最後の列で出力
        self:display();
      else
        for col=0,self.size-1,1 do -- 各列にひとつのクイーンを配置する
          self.board[row]=col;
          self:NQueen(row+1); -- 次の列に王妃を配置
        end
      end
  end
  return setmetatable( this,{__index=NQueen} );
end

NQueen.new():NQueen(0); -- ０列目に王妃を配置してスタート

