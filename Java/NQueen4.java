import org.apache.commons.lang3.time.DurationFormatUtils;

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
 *  ３．配置フラグ（制約テスト高速化）   NQueen3() * N16: 00:01:35
 *<>４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 *  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 *  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 *  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 *  ８．マルチスレッド2                  NQueen8() * N16: 00:00:00
 *
 *
 * ４．対称解除法
 *     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 *     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 * 
 * ■ユニーク解の判定方法
 *   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 * ことを意味しています。
 * 
 *   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 * わりません。
 * 
 *   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 * 下の例では要素数が２個のものと８個のものがあります。
 *
 *
 * Ｎ＝５の全解は１０、ユニーク解は２なのです。
 * 
 * グループ１: ユニーク解１つ目
 * - - - Q -   - Q - - -
 * Q - - - -   - - - - Q
 * - - Q - -   - - Q - -
 * - - - - Q   Q - - - -
 * - Q - - -   - - - Q -
 * 
 * グループ２: ユニーク解２つ目
 * - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 * - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 * Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 * - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 * - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
 *
 * 
 *   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 * ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 * の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 * をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 * と呼ぶことにします。
 * 
 * - - - - Q   0
 * - - Q - -   2
 * Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 * - - - Q -   1
 * - Q - - -   3
 * 
 * 
 *   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 * 変換を試みるだけでユニーク解の判定が可能になります。
 *  
 * 
 * ■ユニーク解の個数を求める
 *   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 * ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 * 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 * うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 * れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 * 央になければならず、それは互いの効き筋にあたるので有り得ません。
 *
 *
 * ***********************************************************************
 * 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
 * ***********************************************************************
 * 
 *   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
 * の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
 * を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
 * ます。突き詰めれば、
 * 
 * [上から２行目のクイーンの位置が右から何番目にあるか]
 * [右から２列目のクイーンの位置が上から何番目にあるか]
 * 
 *
 * を比較するだけで判定することができます。この２つの値が同じになることはないからです。
 * 
 *       3 0
 *       ↓↓
 * - - - - Q ←0
 * - Q - - - ←3
 * - - - - -         上から２行目のクイーンの位置が右から４番目にある。
 * - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
 * - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
 * 
 *   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 * られる解は総てユニーク解であることが保証されます。
 * 
 * - - - - X Q
 * - Q - - X -
 * - - - - X -
 * - - - - X -
 * - - - - - -
 * - - - - - -
 * 
 *   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 * ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 * ます。
 * 
 * X X - - - Q X X
 * X - - - - - - X
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * X - - - - - - X
 * X X - - - - X X
 * 
 *   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 * 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 * て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 * になります。
 * 
 * X X x x x Q X X
 * X - - - x x x X
 * C - - x - x - x
 * - - x - - x - -
 * - x - - - x - -
 * x - - - - x - A
 * X - - - - x - X
 * X X B - - x X X
 *
 *
 * ■ユニーク解から全解への展開
 *   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 * ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 * 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 * 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 * を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 * れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 * いう考察が必要になってきます。
 * 
 *   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 * オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 * 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 * ず８個(＝２×４)になります。
 * 
 *   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 * ために次の事柄を確認します。
 *
 * TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
 *   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
 *    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
 *    ジナルと同型になる。  
 *
 *    COUNT2 * 2
 * 
 *   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
 *    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
 *    り得る。 
 *
 *    COUNT4 * 4
 * 
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 * 
 *    COUNT8 * 8 
 *
 *   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 * りを外す必要がなくなったというわけです。 
 * 
 *   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
 *   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 *
 * 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 * ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
 * 躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
 *
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
  14:           365596        45752  00:00:02
  15:          2279184       285053  00:00:16
  16:         14772512      1846955  00:01:50
*/

class NQueen4 {
  public static void main(String[] args){
    // javac -cp .:commons-lang3-3.4.jar: NQueen4.java ;
    // java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen4  ;
    new NQueen();   //回転・反転・対称
  }
}

class NQueen{
	private int[] board ;
	private long nUnique ; //ユニーク解
	private long nTotal ;  //合計解
	private int size;
	private int max ;
	private boolean[] colChk;
  private boolean[] diagChk;
  private boolean[] antiChk;
	private int[] trial ; //回転・反転の対称チェック
	private int[] scratch ;
  // コンストラクタ
	public NQueen(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.size=2; size<max; size++){
			board=new int[size];
			colChk    = new boolean[size];
			diagChk   = new boolean[2*size-1];
			antiChk   = new boolean[2*size-1];
			trial     = new int[size]; //回転・反転の対称チェック
			scratch   = new int[size];
			nUnique=0;	//ユニーク解
			nTotal=0;	//合計解 
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			nQueens(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,getTotal(),getUnique(),TIME); 
		}
	}
	private void nQueens(int row){
		if(row==size){
			//回転・反転・対称の解析
	         int tst = symmetryOps ();
	         if (tst != 0) {
	            nUnique++;
	            nTotal += tst;
	         }
		}else{
			for(int col=0; col<size; col++){
				board[row]=col; 
				if(colChk[col]==false && antiChk[row+col]==false && diagChk[row-col+(size-1)]==false){
					colChk[col]=antiChk[row+col]=diagChk[row-col+(size-1)]=true;
					nQueens(row+1);
					colChk[col]=antiChk[row+col]=diagChk[row-col+(size-1)]=false;
				}
			}
		}
	}
	private long getUnique(){ 
		return nUnique ; 
	}
	private long getTotal(){ return nTotal ; }
	//以下は以降のステップで使い回します
	private int symmetryOps() {
      int     k;
      int     nEquiv;
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ trial[k] = board[k];}
      //時計回りに90度回転
      rotate (trial, scratch, size, false);
      k = intncmp (board, trial, size);
      if (k > 0) return 0;
      if ( k == 0 )
         nEquiv = 1;
      else {
    	 //時計回りに180度回転
         rotate (trial, scratch, size, false);
         k = intncmp (board, trial, size);
         if (k > 0) return 0;
         if ( k == 0 )
            nEquiv = 2;
         else {
        	//時計回りに270度回転
            rotate (trial, scratch, size, false);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
            nEquiv = 4;
         }
      }
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ trial[k] = board[k];}
      //垂直反転
      vMirror (trial, size);
      k = intncmp (board, trial, size);
      if (k > 0) return 0;
      if (nEquiv > 1) {        // 4回転とは異なる場合
    	 // -90度回転 対角鏡と同等
         rotate (trial, scratch, size, true);
         k = intncmp (board, trial, size);
         if (k > 0) return 0;
         if (nEquiv > 2){     // 2回転とは異なる場合
        	// -180度回転 水平鏡像と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
            // -270度回転 反対角鏡と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
         }
      }
      return nEquiv * 2;
   }
	private int intncmp (int[] lt, int[] rt, int n) {
      int k, rtn = 0;
      for (k = 0; k < n; k++) {
    	  rtn = lt[k]-rt[k];
    	  if ( rtn != 0 ){ break;}
      }
      return rtn;
	}
	private void rotate(int[] check, int[] scr, int n, boolean neg) {
      int j, k;
      int incr;
      k = neg ? 0 : n-1;
      incr = (neg ? +1 : -1);
      for (j = 0; j < n; k += incr ){ scr[j++] = check[k];}
      k = neg ? n-1 : 0;
      for (j = 0; j < n; k -= incr ){ check[scr[j++]] = k;}
	}
	private void vMirror(int[] check, int n) {
      int j;
      for (j = 0; j < n; j++){ check[j] = (n-1) - check[j];}
      return;
	}
}

