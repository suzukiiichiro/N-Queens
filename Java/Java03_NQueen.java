/**

 Javaで学ぶアルゴリズムとデータ構造  
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java03_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java03_NQueen  ;


	３．バックトラック
    各縦横列に加え斜め１個の王妃を配置する組み合わせの配置フラグ各列、対角線
上に クイーンがあるかどうかのフラグを用意し、途中で制約を満たさない事が明らか
な場 合は、それ以降のパターン生成を行わない。各列、対角線上にクイーンがあるか
どう かのフラグを用意することで高速化を図る。これまでは行方向と列方向に重複し
ない 組み合わせを列挙するものですが、王妃は斜め方向のコマをとることができるの
で、 どの斜めライン上にも王妃をひとつだけしか配置できない制限を加える事によ
り、深 さ優先探索で全ての葉を訪問せず木を降りても解がないと判明した時点で木を
引き返 すということができる。

 実行結果 
 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            0     00:00:00.000
 5:               10            0     00:00:00.000
 6:                4            0     00:00:00.001
 7:               40            0     00:00:00.000
 8:               92            0     00:00:00.001
 9:              352            0     00:00:00.002
10:              724            0     00:00:00.003
11:             2680            0     00:00:00.014
12:            14200            0     00:00:00.064
13:            73712            0     00:00:00.381
14:           365596            0     00:00:02.229
15:          2279184            0     00:00:14.249
16:         14772512            0     00:01:38.375
17:         95815104            0     00:12:18.698
*/

import org.apache.commons.lang3.time.DurationFormatUtils;

class Java03_NQueen{
  //グローバル変数
	private int				SIZE;
	private long			TOTAL;
	private int[]			board;
	private boolean[]	fA,fB,fC;
	// コンストラクタ
	public Java03_NQueen(){
		int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(SIZE=4;SIZE<max;SIZE++){
			TOTAL=0;
			board=new int[SIZE];
			fA=new boolean[SIZE];
			fB=new boolean[2*SIZE-1];
			fC=new boolean[2*SIZE-1];
			for(int i=0;i<SIZE;i++){
				board[i]=i;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",SIZE,TOTAL,0,TIME);
		}
	}
	// 再帰関数
	private void nQueens(int row){
		if(row==SIZE){
			TOTAL++;
		}else{
			for(int i=0;i<SIZE;i++){
				board[row]=i; // 各列にひとつのクイーンを配置する
				if(fA[i]==false
          &&fC[row+i]==false
          &&fB[row-i+(SIZE-1)]==false){
					fA[i]=true;
          fB[row-board[row]+SIZE-1]=true;
          fC[row+board[row]]=true;
					nQueens(row+1);
					fA[i]=false;
          fB[row-board[row]+SIZE-1]=false;
          fC[row+board[row]]=false;
				}
			}
		}
	}
  //メインメソッド
	public static void main(String[] args){
		new Java03_NQueen();   //実行はコメントを外して $ ./MAIN.SH を実行
	}
}

