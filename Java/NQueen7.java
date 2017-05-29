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
 *  １．ブルートフォース（力まかせ探索） NQueen1()
 *  ２．配置フラグ（制約テスト高速化）   NQueen2()
 *  ３．バックトラック                   NQueen3()
 *  ４．対称解除法(回転と斜軸）          NQueen4()
 *  ５．枝刈りと最適化                   NQueen5()
 *  ６．マルチスレッド1                  NQueen6()
 *<>７．ビットマップ                     NQueen7()
 *  ８．マルチスレッド2                  NQueen8()
*/

/**
 * ７．ビットマップ
 *
 *   ビット演算を使って高速化 状態をビットマップにパックし、処理する
 *   単純なバックトラックよりも２０〜３０倍高速
 * 
 * 　ビットマップであれば、シフトにより高速にデータを移動できる。
 *  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
 *  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
 *  分。
 *
 * 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
 * 　バックトラックよりも２０−３０倍高速。
 * 
 * ===================
 * 考え方 1
 * ===================
 *
 * 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
 * つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
 * する)
 * 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
 * ずれかのビット位置をひとつだけONにして進めていきます。
 *
 * 
 *- - - - - Q - -    00000100 0番目のビットフィールド
 *- - - Q - - - -    00010000 1番目のビットフィールド
 *- - - - - - Q -    00000010 2番目のビットフィールド
 *Q - - - - - - -    10000000 3番目のビットフィールド
 *- - - - - - - Q    00000001 4番目のビットフィールド
 *- Q - - - - - -    01000000 5番目のビットフィールド
 *- - - - Q - - -    00001000 6番目のビットフィールド
 *- - Q - - - - -    00100000 7番目のビットフィールド
 *
 *
 * ===================
 * 考え方 2
 * ===================
 *
 * 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。
 *
 * 1. 左下に効き筋が進むもの: left 
 * 2. 真下に効き筋が進むもの: down
 * 3. 右下に効き筋が進むもの: right
 *
 *次に、斜めの利き筋を考えます。
 * 上図の場合、
 * 1列目の右斜め上の利き筋は 3 番目 (0x08)
 * 2列目の右斜め上の利き筋は 2 番目 (0x04) になります。
 * この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
 * ことができます。
 * また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目 (0x40)
 *になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。
 *
 *つまり、右シフトの利き筋を right、左シフトの利き筋を left で表すことで、クイー
 *ンの効き筋はrightとleftを1 ビットシフトするだけで求めることができるわけです。
 *
 *  *-------------
 *  | . . . . . .
 *  | . . . -3. .  0x02 -|
 *  | . . -2. . .  0x04  |(1 bit 右シフト right)
 *  | . -1. . . .  0x08 -|
 *  | Q . . . . .  0x10 ←(Q の位置は 4   down)
 *  | . +1. . . .  0x20 -| 
 *  | . . +2. . .  0x40  |(1 bit 左シフト left)  
 *  | . . . +3. .  0x80 -|
 *  *-------------
 *  図：斜めの利き筋のチェック
 *
 * n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
 * の３つのビットフィールドとn番目のビットフィールド(bit)とのOR演算をそれぞれ行
 * います。leftは左にひとつシフトし、downはそのまま、rightは右にひとつシフトして
 * n+1番目のビットフィールド探索に渡してやります。
 *
 * left : (left |bit)<<1
 * right: (right|bit)>>1
 * down :   down|bit
 *
 *
 * ===================
 * 考え方 3
 * ===================
 *
 *   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
 * ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
 * できない位置ということになります。次にその３つのビットフィールドをORしたビッ
 * トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
 * ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
 * 次の演算を行なってみます。
 * 
 * bit = -bitmap & bitmap; //一番右のビットを取り出す
 * 
 *   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
 * に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
 * と次のようになります。
 * 
 *  00000011   3
 *  00000010   2
 *  00000001   1
 *  00000000   0
 *  11111111  -1
 *  11111110  -2
 *  11111101  -3
 * 
 *   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
 * 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
 * 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
 * によって1ビット抽出を実現させていることが重要です。
 * 
 *      00010110   22
 *  AND 11101010  -22
 * ------------------
 *      00000010
 * 
 *   さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
 * 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
 * 生成されることになります。
 * 
 * while (bitmap) {
 *     bit = -bitmap & bitmap;
 *     bitmap ^= bit;
 *     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
 * }
 */

   /**
    * 実行結果
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
  15:          2279184       285053  00:00:00
  16:         14772512      1846955  00:00:02
	17:         95815104     11977939  00:00:15
	18:        666090624     83263591  00:01:49
	19:       4968057848    621012754  00:13:55
	20:      39029188884   4878666808  01:50:42
	21:     314666222712  39333324973  15:34:05
	22:    2691008701644 336376244042 136:08:43
  */

public class NQueen7 {
  public static void main(String[] args){
    // javac -cp .:commons-lang3-3.4.jar: NQueen7.java ;
    // java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen7  ;
    new NQueen() ;    // シンプルな対称解除法＋ビットマップ
  }
}

class NQueen {
	private int bit;
	private int MASK;
	private int SIZEE;
	private int[] BOARD;
	private int TOPBIT;
	private int SIDEMASK;
	private int LASTMASK;
	private int ENDBIT;
	private int BOUND1;
	private int BOUND2;
	private long UNIQUE ;
	private long TOTAL ;
	private long COUNT8;
	private long COUNT4;
	private long COUNT2;
	private void Check(int bsize) {
		int _BOARD =0;
		int _BOARD1=BOUND1;
		int _BOARD2=BOUND2;
		int _BOARDE=SIZEE;
		//90度回転
		if (BOARD[_BOARD2] == 1) {
			int own = _BOARD+1;
			for (int ptn=2 ; own<=_BOARDE; own++, ptn<<=1) {
				bit=1;
				int bown = BOARD[own];
				for (int you=_BOARDE; (BOARD[you] != ptn) && (bown >= bit); you--)
					bit<<=1;
				if (bown>bit) { return; }
				if (bown<bit) { break; }
			}
			//90度回転して同型なら180度/270度回転も同型である
			if (own>_BOARDE) {
				COUNT2++;
				return;
			}
		}
		//180度回転
		if (bsize==ENDBIT) {
			int own = _BOARD+1;
			for (int you=_BOARDE-1; own<=_BOARDE; own++, you--) {
				bit = 1;
				for (int ptn=TOPBIT; (ptn!=BOARD[you])&&(BOARD[own]>=bit); ptn>>=1)
					bit<<=1;
				if (BOARD[own] > bit) { return; }
				if (BOARD[own] < bit) { break; }
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if (own>_BOARDE) {
				COUNT4++;
				return;
			}
		}
		//270度回転
		if (BOARD[_BOARD1]==TOPBIT) {
			int own=_BOARD+1;
			for (int ptn=TOPBIT>>1; own<=_BOARDE; own++, ptn>>=1) {
				bit=1;
				for (int you=_BOARD; BOARD[you]!=ptn && BOARD[own]>=bit; you++) {
					bit<<=1;
				}
				if (BOARD[own]>bit) { return; }
				if (BOARD[own]<bit) { break; }
			}
		}
		COUNT8++;
	}
	/**
	 * 最上段のクイーンが角以外にある場合の探索
	 */
	private void backTrack2(int y, int left, int down, int right){
		int bitmap= ( MASK & ~(left|down|right)) ;
		if(y==SIZEE){
			if(bitmap!=0){
				if( (bitmap & LASTMASK)==0){ //最下段枝刈り
					BOARD[y]=bitmap;
					Check(bitmap);
				}
			}
		}else{
			if(y<BOUND1){ //上部サイド枝刈り
				bitmap|=SIDEMASK ;
				bitmap^=SIDEMASK;
			}else if(y==BOUND2){ //下部サイド枝刈り
				if( (down&SIDEMASK) == 0) return ;
				if( (down&SIDEMASK) !=SIDEMASK) bitmap&=SIDEMASK;
			}
			while(bitmap!=0){
				bitmap^=BOARD[y]=bit=-bitmap&bitmap;
				backTrack2((y+1), (left|bit)<<1, (down|bit), (right|bit)>>1 )	;	
			}
		}
	}
	/**
	 * 最上段のクイーンが角にある場合の探索
	 */
	private void backTrack1(int y, int left, int down, int right){
		int bitmap=( MASK & ~(left|down|right) );
		if(y==SIZEE){
			if(bitmap!=0){
				BOARD[y]=bitmap;
				COUNT8++;
			}
		}else{
			if(y<BOUND1){//斜軸反転解の排除
				bitmap|=2 ;
				bitmap^=2;
			}
			while(bitmap!=0){
				bitmap^=BOARD[y]=bit=(-bitmap&bitmap);
				backTrack1( y+1, (left|bit)<<1, down|bit, (right|bit)>>1 );
			}
		}
	}
	private void bitmap_rotate(int SIZE) {
		SIZEE = SIZE-1;
		TOPBIT = 1<<SIZEE;
		MASK=(1<<SIZE)-1;
		COUNT8 = COUNT4 = COUNT2 = 0;
		BOARD = new int[SIZE];
		BOARD[0] = 1;
		for (BOUND1=2; BOUND1<SIZEE; BOUND1++) {
			BOARD[1]=bit=(1<< BOUND1);
			backTrack1(2, (2|bit)<<1, (1|bit), (bit>>1));
		}
		SIDEMASK=LASTMASK=(TOPBIT|1);
		ENDBIT = (TOPBIT>>1);
		for (BOUND1=1, BOUND2=SIZE-2; BOUND1<BOUND2; BOUND1++, BOUND2--) {
			BOARD[0]=bit=(1<<BOUND1);
			backTrack2(1, bit<<1, bit, bit>>1);
			LASTMASK|=LASTMASK>>1|LASTMASK<<1;
			ENDBIT>>=1;
		}
		UNIQUE = COUNT8 + COUNT4 + COUNT2;
		TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
	}
  // コンストラクタ
	public NQueen(){
	int max=27;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(int size=2; size<max+1; size++){
			long start = System.currentTimeMillis() ;
			bitmap_rotate(size);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,TOTAL,UNIQUE,TIME); 
		}
	}
}

