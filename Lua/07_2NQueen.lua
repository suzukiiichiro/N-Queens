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
 *<>２．配置フラグ（制約テスト高速化）   NQueen3()
 *  ３．バックトラック                   NQueen2()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *  ５．枝刈りと最適化                   NQueen5()
 *  ６．スレッド                         NQueen6()
 *  ７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド                   NQueen8()
*/

/** 
 * ２．配置フラグ（制約テスト高速化）
 *  パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
 *  い事が明らかな場合は、それ以降のパターン生成を行わない。
 * 「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
 * 事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
 * (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
 * み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
  :
  :
  7 6 5 4 2 0 3 1 : 40310
  7 6 5 4 2 1 0 3 : 40311
  7 6 5 4 2 1 3 0 : 40312
  7 6 5 4 2 3 0 1 : 40313
  7 6 5 4 2 3 1 0 : 40314
  7 6 5 4 3 0 1 2 : 40315
  7 6 5 4 3 0 2 1 : 40316
  7 6 5 4 3 1 0 2 : 40317
  7 6 5 4 3 1 2 0 : 40318
  7 6 5 4 3 2 0 1 : 40319
  7 6 5 4 3 2 1 0 : 40320
                         N16: 00:00:01
  */
]]--

NQueen={}; NQueen.new=function()

  local this={
    board={};
    flag={};
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
        if self.flag[col] then -- i行には王妃は未配置
        else
          self.board[row]=col; -- 王妃をi行に配置
          self.flag[col]=true; -- i行に王妃を配置したらtrueに
          self:NQueen(row+1); -- 次の列に王妃を配置
          self.flag[col]=false; --戻ってきたら王妃を取り除く
        end
      end
    end
  end
  return setmetatable( this,{__index=NQueen} );
end

NQueen.new():NQueen(0);

