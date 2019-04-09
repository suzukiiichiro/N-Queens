#!/usr/bin/env lua

--[[ 
  Luaで学ぶ「アルゴリズムとデータ構造」
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  １．ブルートフォース（力まかせ探索）
 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
   なるかをチェックする方法
   (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
   けであって8王妃問題を解いているわけではありません

  実行には luajiit が必要です。
  Macの場合は

  # インストール
  $ brew install luajit ;

  # 確認
  $ which lulajit
  $ /usr/local/bin/luajit

  #実行
  $ luajit Lua01_N-Queen.lua
  または
  $ ./Lua01_N-Queen.lua
  :
  :
  実行結果

  7 7 7 7 7 7 6 7 : 16777208
  7 7 7 7 7 7 7 0 : 16777209
  7 7 7 7 7 7 7 1 : 16777210
  7 7 7 7 7 7 7 2 : 16777211
  7 7 7 7 7 7 7 3 : 16777212
  7 7 7 7 7 7 7 4 : 16777213
  7 7 7 7 7 7 7 5 : 16777214
  7 7 7 7 7 7 7 6 : 16777215
  7 7 7 7 7 7 7 7 : 16777216
]]--

NQueen={}; NQueen.new=function()
  local this={                    --NQueenクラスのローカルメソッド
    board={};
    size=8;
    count=1;
  };
  -- コメントはハイフンをふたつ繋ぎます
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
      self:display();
    else
      for col=0,self.size-1,1 do  -- 各列にひとつのクイーンを配置する
        self.board[row]=col;
        self:NQueen(row+1);       -- 次の列に王妃を配置
      end
    end
  end
  --
  return setmetatable( this,{__index=NQueen} );
end
--
NQueen.new():NQueen(0);           -- ０列目に王妃を配置してスタート

