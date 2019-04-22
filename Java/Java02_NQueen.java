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
	40312 :  7 6 5 4 2 1 3 0
	40313 :  7 6 5 4 2 3 0 0
	40314 :  7 6 5 4 2 3 1 0
	40315 :  7 6 5 4 3 0 1 0
	40316 :  7 6 5 4 3 0 2 0
	40317 :  7 6 5 4 3 1 0 0
	40318 :  7 6 5 4 3 1 2 0
	40319 :  7 6 5 4 3 2 0 0
	40320 :  7 6 5 4 3 2 1 0
*/

import org.apache.commons.lang3.time.DurationFormatUtils;

class Java02_NQueen{
  //グローバル変数
	private int		SIZE;
	private int		COUNT;
	private int[]	board;
  private int[] fA;
	public Java02_NQueen(int size){
		this.SIZE=size;
		this.COUNT=1;              // 解数は1からカウント
		board=new int[size];
    fA=new int[size];
		nQueens(0);               // ０列目に王妃を配置してスタート
	}
	// 出力
	private void print(){
		System.out.print(this.COUNT++ + " : ");
		for(int i=0;i<SIZE;i++){
			System.out.printf("%2d",board[i]);
		}
    System.out.println("");
	}
  //ロジックメソッド
	private void nQueens(int row){
		if(row==SIZE-1){
			print();            // 全列に配置完了 最後の列で出力
		}else{
			for(int i=0;i<SIZE;i++){
				board[row]=i;   // 各列にひとつのクイーンを配置する
				if(fA[i]==0){
          fA[i]=1;
					nQueens(row+1); // 次の列に王妃を配置
          fA[i]=0;
				}
			}
		}
	}
  //メインメソッド
	public static void main(String[] args){
		new Java02_NQueen(8);
	}
}
