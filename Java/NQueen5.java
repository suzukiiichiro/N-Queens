import org.apache.commons.lang3.time.DurationFormatUtils;

/**
 * Luaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
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
 *  ２．配置フラグ（制約テスト高速化）   NQueen2()
 *  ３．バックトラック                   NQueen3()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *<>５．枝刈りと最適化                   NQueen5()
 *  ６．マルチスレッド1                  NQueen6()
 *  ７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド2                  NQueen8()
*/

/**
 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

   N:            Total       Unique    hh:mm:ss
   2:                0            0  00:00:00
   3:                0            0  00:00:00
   4:                2            1  00:00:00
   5:               10            2  00:00:00
   6:                4            1  00:00:00
   7:               40            6  00:00:00
   8:               92           12  00:00:00
   9:              352           46  00:00:00
  10:              724           92  00:00:00
  11:             2680          341  00:00:00
  12:            14200         1787  00:00:00
  13:            73712         9233  00:00:00
  14:           365596        45752  00:00:00
  15:          2279184       285053  00:00:03
  16:         14772512      1846955  00:00:24
  */

class NQueen5 {
  public static void main(String[] args){
    // javac -cp .:commons-lang3-3.4.jar: NQueen5.java ;
    // java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen5  ;
    new NQueen(); //最適化
  }
}

class NQueen{
	private int[] board ;
	private long nUnique ;
	private long nTotal ;
	private int size;
	private int max ;
//	private boolean[] colChk;
  private boolean[] diagChk;
  private boolean[] antiChk;
	private int[] trial ;
	private int[] scratch ;
  // コンストラクタ
	public NQueen(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.size=2; size<max; size++){
			board=new int[size];
			trial     = new int[size];
			scratch   = new int[size];
//			colChk    = new boolean[size];
			diagChk   = new boolean[2*size-1];
			antiChk   = new boolean[2*size-1];
			nTotal=0; 
			nUnique=0;
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			nQueens(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,getTotal(),getUnique(),TIME); 
		}
	}
	void nQueens(int row) {
		int k, lim, vTemp;
		if (row < size-1) {
		    if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
				nQueens(row + 1);
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
			}
			lim = (row != 0) ? size : (size + 1) / 2;
			for (k=row+1; k<lim; k++) {
				vTemp = board[k];
				board[k] = board[row];
				board[row] = vTemp;
				if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
		            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
					nQueens(row + 1);
		            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
				}
			}
			vTemp = board[row];
			for (k = row + 1; k < size; k++){
				board[k - 1] = board[k];
			}
			board[k - 1] = vTemp;
		} else { 
	        if ( (diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
				return;
			}
			k = symmetryOps();
			if (k != 0) {
				nUnique++;
				nTotal += k;
			}
		}
		return;
	}
	private long getUnique(){ return nUnique ; }
	private long getTotal(){ return nTotal ; }
	//
	//以下は以降のステップで使い回します
	private int symmetryOps() {
      int     k;
      int     nEquiv;
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){
    	  trial[k] = board[k];
      }
      //時計回りに90度回転
      rotate (trial, scratch, size, false);
      k = intncmp (board, trial, size);
      if (k > 0) { 
    	  return 0;
      }
      if ( k == 0 ){
         nEquiv = 1;
      } else {
    	 //時計回りに180度回転
         rotate (trial, scratch, size, false);
         k = intncmp (board, trial, size);
         if (k > 0) { 
        	 return 0;
         }
         if ( k == 0 ){
            nEquiv = 2;
         } else {
        	 /* 270 degrees */
        	//時計回りに270度回転
            rotate (trial, scratch, size, false);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            nEquiv = 4;
         }
      }
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ 
    	  trial[k] = board[k];
      }
      //垂直反転
      vMirror (trial, size);
      k = intncmp (board, trial, size);
      if (k > 0) {
    	  return 0;
      }
      if (nEquiv > 1) {        // 4回転とは異なる場合
    	 // -90度回転 対角鏡と同等
         rotate (trial, scratch, size, true);
         k = intncmp (board, trial, size);
         if (k > 0) {
        	 return 0;
         }
         if (nEquiv > 2){     // 2回転とは異なる場合
        	// -180度回転 水平鏡像と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            // -270度回転 反対角鏡と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
         }
      }
      return nEquiv * 2;
   }
	private int intncmp (int[] lt, int[] rt, int n) {
      int k, rtn = 0;
      for (k = 0; k < n; k++) {
    	  rtn = lt[k]-rt[k];
    	  if ( rtn != 0 ){ 
    		  break;
    	  }
      }
      return rtn;
	}
	private void rotate(int[] check, int[] scr, int n, boolean neg) {
      int j, k;
      int incr;
      k = neg ? 0 : n-1;
      incr = (neg ? +1 : -1);
      for (j = 0; j < n; k += incr ){ 
    	  scr[j++] = check[k];
      }
      k = neg ? n-1 : 0;
      for (j = 0; j < n; k -= incr ) { 
    	 check[scr[j++]] = k;
      }
	}
	private void vMirror(int[] check, int n) {
      int j;
      for (j = 0; j < n; j++) { 
    	  check[j] = (n-1) - check[j];
      }
      return;
	}
}

