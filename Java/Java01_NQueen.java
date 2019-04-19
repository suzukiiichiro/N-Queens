/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 
 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

  コンパイル
  javac -cp .:commons-lang3-3.4.jar Java01_NQueen.java ;

  実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java01_NQueen  ;

  １．ブルートフォース　力任せ探索
   各縦列に１個の王妃を配置する組み合わせを再帰的に列挙
   　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
   なるかをチェックする方法(※)各行に１個の王妃を配置する組み合わせを再帰的に列
   挙組み合わせを生成するだけであって8王妃問題を解いているわけではない

   実行結果
      :
      :
      16777208 : 7 7 7 7 7 7 6 7
      16777209 : 7 7 7 7 7 7 7 0
      16777210 : 7 7 7 7 7 7 7 1
      16777211 : 7 7 7 7 7 7 7 2
      16777212 : 7 7 7 7 7 7 7 3
      16777213 : 7 7 7 7 7 7 7 4
      16777214 : 7 7 7 7 7 7 7 5
      16777215 : 7 7 7 7 7 7 7 6
      16777216 : 7 7 7 7 7 7 7 7
*/

import org.apache.commons.lang3.time.DurationFormatUtils;

class Java01_NQueen{
  //グローバル変数
	private int SIZE;
	private int COUNT;
	private int[]	board;
  // コンストラクタ
	public Java01_NQueen(int size){
		this.SIZE=size;
		this.COUNT=1;             // 解数は1からカウント
		board=new int[size];
		nQueens(0);               // ０列目に王妃を配置してスタート
	}
  //ロジックメソッド
	private void nQueens(int row){
		if(row==this.SIZE){       // 全列に配置完了 最後の列で出力
			print();
		}else{
			for(int col=0;col<this.SIZE;col++){
				board[row]=col;       // 各列にひとつのクイーンを配置する
				nQueens(row+1);       // 次の列に王妃を配置
			}
		}
	}
  //出力メソッド
	private void print(){
		System.out.print(this.COUNT++ + " : ");
		for(int col=0;col<this.SIZE;col++){
			System.out.printf("%2d",board[col]);
		}
    System.out.println("");
	}
  //メインメソッド
	public static void main(String[] args){
	  new Java01_NQueen(8);
	}
}
