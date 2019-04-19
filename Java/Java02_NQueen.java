/**

 Javaで学ぶアルゴリズムとデータ構造  
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

  コンパイル
  javac -cp .:commons-lang3-3.4.jar Java02_NQueen.java ;

  実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java02_NQueen  ;
 

　 ２．配置フラグ（制約テスト高速化）
   各縦横列に１個の王妃を配置する組み合わせを再帰的に列挙
   
   実行結果 
   :
   :
   6 1 5 2 0 3 7 4 : 83
   6 2 0 5 7 4 1 3 : 84
   6 2 7 1 4 0 5 3 : 85
   6 3 1 4 7 0 2 5 : 86
   6 3 1 7 5 0 2 4 : 87
   6 4 2 0 5 7 1 3 : 88
   7 1 3 0 6 4 2 5 : 89
   7 1 4 2 0 6 3 5 : 90
   7 2 0 5 1 4 6 3 : 91
   7 3 0 2 5 1 6 4 : 92
*/

import org.apache.commons.lang3.time.DurationFormatUtils;

class Java02_NQueen{
  //グローバル変数
	private int		SIZE;
	private int		COUNT;
	private int[]	board;
	public Java02_NQueen(int size){
		this.SIZE=size;
		this.COUNT=1;              // 解数は1からカウント
		board=new int[size];
		nQueens(0);               // ０列目に王妃を配置してスタート
	}
	// 出力
	private void print(){
		System.out.println(count++ + " : ");
		for(int col=0;col<size;col++){
			System.out.printf("%2d",board[col]);
		}
    System.out.println("");
	}
	private boolean Valid(int row){
		for(int Idx=0;Idx<row;Idx++){
			if(board[Idx]==board[row]||Math.abs(board[row]-board[Idx])==(row-Idx)){
				return false;
			}
		}
		return true;
	}
	private void nQueens(int row){
		if(row==this.SIZE-1){
			print();            // 全列に配置完了 最後の列で出力
		}else{
			for(int col=0;col<this.SIZE;col++){
				board[row]=col;   // 各列にひとつのクイーンを配置する
				if(Valid(row)){
					nQueens(row+1); // 次の列に王妃を配置
				}
			}
		}
	}
	public static void main(String[] args){
		new Java02_NQueen(8);
	}
}
