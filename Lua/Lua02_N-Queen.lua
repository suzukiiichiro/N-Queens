#!/usr/bin/env lua

--[[
  Luaで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  ２．配置フラグ（制約テスト高速化）
   パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
   い事が明らかな場合は、それ以降のパターン生成を行わない。
  「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
  事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
  (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
  み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。

  実行結果
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
]]--

NQueen={}; NQueen.new=function()
  --
  local this={
    board={};
    flag={};
    size=8;
    count=1;
  };
  --
  function NQueen:display()       --出力用メソッド
    for col=0,self.size-1,1 do
      io.write(string.format('%2d', self.board[col]));
    end
    print(" : "..self.count);
    self.count=self.count+1;      --インクリメント
  end
  --
  function NQueen:NQueen(row)     --メインロジックメソッド 
    if row==self.size then        --全列に配置完了 最後の列で出力
      self:display();             --出力
    else
      for col=0,self.size-1,1 do  --各列にひとつのクイーンを配置する
        if self.flag[col] then    --i行には王妃は未配置
        else
          self.board[row]=col;    --王妃をi行に配置
          self.flag[col]=true;    --i行に王妃を配置したらtrueに
          self:NQueen(row+1);     --次の列に王妃を配置
          self.flag[col]=false;   --戻ってきたら王妃を取り除く
        end
      end
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen(0);           -- ０列目に王妃を配置してスタート
