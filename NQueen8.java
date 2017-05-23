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
 *  ４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 *  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 *  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 *  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 *<>８．マルチスレッド2                  NQueen8() * N16: 00:00:00
*/

/**
 *   ８．マルチスレッド2
 * 
 * ここまでの処理は、一つのスレッドが順番にＡ行の１列目から順を追って処理判定をし
 * てきました。この節では、Ａ行の列それぞれに別々のスレッドを割り当て、全てのス
 * レッドを同時に処理判定させます。Ａ行それぞれの列の処理判定結果はBoardクラスで
 * 管理し、処理完了とともに結果を出力します。スレッドはWorkEngineクラスがNの数だ
 * け生成されます。WorkEngineクラスは自身の持ち場のＡ行＊列の処理だけを担当しま
 * す。これらはマルチスレッド処理と言い、並列処理のための同期、排他、ロック、集計
 * など複雑な処理を理解する知識が必要です。そして処理の最後に合計値を算出する方法
 * をマルチスレッド処理と言います。
 * １Ｘ１，２Ｘ２，３Ｘ３，４Ｘ４，５Ｘ５，６Ｘ６，７ｘ７、８Ｘ８のボートごとの計
 * 算をスレッドに割り当てる手法がちまたでは多く見受けられます。これらの手法は、
 * 実装は簡単ですが、Ｎが７の計算をしながら別スレッドでＮが８の計算を並列処理する
 * といった矛盾が原因で、Ｎが大きくなるとむしろ処理時間がかかります。
 *   ここでは理想的なアルゴリズムとして前者の手法でプログラミングします。
 */

  /**
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
  16:         14772512      1846955  00:00:00
  17:         95815104     11977939  00:00:04
  18:        666090624     83263591  00:00:34
  19:       4968057848    621012754  00:04:18
  20:      39029188884   4878666808  00:35:07
  21:     314666222712  39333324973  04:41:36
  22:    2691008701644 336376244042  39:14:59
   */
class NQueen8 {
  public static void main(String[] args){
    // javac -cp .:commons-lang3-3.4.jar: NQueen8.java ;
    // java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen8  ;
    new NQueen() ;    // シンプルな対称解除法＋ビットマップ＋マルチスレッド
  }
}

class NQueen8_Board {
	private long COUNT8 ; private long COUNT4 ; private long COUNT2 ;
	private int limit ; private int nextCol ; private int nextID ;
	public NQueen8_Board(int size) {
		COUNT8=COUNT4=COUNT2=0;
		nextCol=0; nextID=0;
		limit=(size+1)/2 ;
		this.size=size ;
	}	
	private int size ;
	private int ENDBIT ;
	private int LASTMASK ;
	private int SIDEMASK ;
	private int MASK ;
	private int TOPBIT ;
	public void setMASK(int MASK){ this.MASK=MASK ; }
	public int getMASK(){ return this.MASK ; }
	public void setTOPBIT(int TOPBIT){ this.TOPBIT=TOPBIT ; }
	public int getTOPBIT(){ return this.TOPBIT ; }
	public void setSIDEMASK(int SIDEMASK){ this.SIDEMASK=SIDEMASK ; }
	public int getSIDEMASK(){ return this.SIDEMASK ; }
	public void setENDBIT(int ENDBIT){ this.ENDBIT=ENDBIT; }
	
	
	public int getENDBIT(){ return ENDBIT ; }
	public void setLASTMASK(int LASTMASK){ this.LASTMASK=LASTMASK ; }
	public int getLASTMASK(){ return LASTMASK; }
	public long  getTotal() { return COUNT8 * 8 + COUNT4 * 4 + COUNT2 * 2 ; }
	public long getUnique() { return COUNT8 + COUNT4 + COUNT2 ; }
	public synchronized int threadID(){ return nextID; }
	public synchronized int nextJob(){ return nextCol < limit ? nextCol++ : -1; }
	public void resetCount(){ COUNT8=COUNT4=COUNT2=0; }
	public synchronized void setCount(long COUNT8, long COUNT4, long COUNT2){
		this.COUNT8+=COUNT8 ; this.COUNT4+=COUNT4 ; this.COUNT2+=COUNT2 ;
	}
}

class NQueen8_WorkEngine extends Thread{
	private int SIZEE;
	private int TOPBIT;
	private int MASK ;
	private int[] BOARD;
	private int BOUND1;
	private int BOUND2;
	private int SIDEMASK;
	private int LASTMASK;
	private int ENDBIT;
	private NQueen8_WorkEngine child  ;
	private NQueen8_Board info;
	private int size ;
	private int B1, B2;
	private int nMore ;
//		boolean bThread=false ; //スレッド処理をするか (Yes=true / No=false) 
		boolean bThread=true ; //スレッド処理をするか (Yes=true / No=false) 
	// コンストラクタ
	public NQueen8_WorkEngine(int size, int nMore, NQueen8_Board info, int B1, int B2){
		this.size=size;
		this.info=info ;
		this.nMore=nMore ;
		this.B1=B1; 
		this.B2=B2;
		BOARD=new int[size];
		if(nMore>0){
			try{
				if(bThread){
					child = new NQueen8_WorkEngine(size, nMore-1, info, B1-1, B2+1);
					child.start();
//					child.join();
				}
			}catch(Exception e){
				System.out.println(e);
			}
		}else{
			child=null ;
		}
	}
	//シングルスレッド
	public void run() {
		if(child==null){
			if(nMore>0){
				// 最上段のクイーンが角以外にある場合の探索
				B1=2; 
				BOARD[0]=1;
				SIZEE=size-1;
				MASK=(1<<size)-1;
				TOPBIT=1<<SIZEE;
				while(B1>1 && B1<SIZEE) {
					BOUND1(B1);
					B1++;
				}
				SIDEMASK=LASTMASK=(TOPBIT|1);
				ENDBIT=(TOPBIT>>1);
				// 最上段のクイーンが角にある場合の探索
				B1=1; 
				B2=size-2;
				while(B1>0 && B2<size-1 && B1<B2){
					BOUND2(B1, B2);
					B1++ ;B2--;
					ENDBIT>>=1;
					LASTMASK|=LASTMASK>>1|LASTMASK<<1;
				}
			}
		}
		//マルチスレッド
		if(child!=null){
			// 最上段のクイーンが角以外にある場合の探索
			BOARD[0]=1;
			SIZEE=size-1;
			MASK=(1<<size)-1;
			TOPBIT=1<<SIZEE;

			if(B1>1 && B1<SIZEE) { 
				BOUND1(B1); 
			}

			// SIDEMASK=LASTMASK=(TOPBIT|1);
			// ENDBIT=(TOPBIT>>1);
			ENDBIT=(TOPBIT>>B1);
			SIDEMASK=(TOPBIT|1);
			LASTMASK=(TOPBIT|1);
			// 最上段のクイーンが角にある場合の探索
			if(B1>0 && B2<size-1 && B1<B2){ 
				// LASTMASK|=LASTMASK>>1|LASTMASK<<1;
				for(int i=1; i<B1; i++){
					LASTMASK=LASTMASK|LASTMASK>>1|LASTMASK<<1;
				}
				BOUND2(B1, B2); 
				// ENDBIT>>=1;
				ENDBIT>>=nMore;
			}
			try{
				child.join();
			} catch (Exception e){
				System.out.println(e);
			}
		}
	}
	// 最上段のクイーンが角以外にある場合の探索
	private void BOUND1(int B1){
		int bit ;
		BOUND1=B1 ;
		if(BOUND1<SIZEE) {
			BOARD[1]=bit=(1<<BOUND1);
			backTrack1(2, (2|bit)<<1, (1|bit), (bit>>1));
		}
	}
	// 最上段のクイーンが角にある場合の探索
	private void BOUND2(int B1, int B2){
		int bit ;
		BOUND1=B1 ;
		BOUND2=B2;
		if(BOUND1<BOUND2) {
			BOARD[0]=bit=(1<<BOUND1);
			backTrack2(1, bit<<1, bit, bit>>1);
//			System.out.println("B1:" + B1 + " B2:" + B2 + " nMore:" + nMore + " TOPBIT:" + TOPBIT + " ENDBIT:" + ENDBIT + " MASK:" + MASK + " SIDEMASK:"+ SIDEMASK + " LASTMASK:" + LASTMASK );
		}
	}
	// 最上段のクイーンが角以外にある場合の探索
	private void backTrack2(int y, int left, int down, int right){
		int bit ;
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
				backTrack2((y+1), (left|bit)<<1, (down|bit), (right|bit)>>1); 
			}
		}
	}
	// 最上段のクイーンが角にある場合の探索
	private void backTrack1(int y, int left, int down, int right){
		int bit ;
		int bitmap=( MASK & ~(left|down|right) );
		if(y==SIZEE){
			if(bitmap!=0){
				BOARD[y]=bitmap;
				// COUNT8++;
				info.setCount(1, 0, 0);
			}
		}else{
			if(y<BOUND1){//斜軸反転解の排除
				bitmap|=2 ;
				bitmap^=2;
			}
			while(bitmap!=0){
				bitmap^=BOARD[y]=bit=(-bitmap&bitmap);
				backTrack1( y+1, (left|bit)<<1, (down|bit), (right|bit)>>1) ;
			}
		}
	}
	//対称解除法
	synchronized private void Check(int bsize) {
		//90度回転
		if (BOARD[BOUND2] == 1) {
			int own=1;
			for (int ptn=2 ; own<=SIZEE; own++, ptn<<=1) {
				int bit=1;
				int bown = BOARD[own];
				for (int you=SIZEE; (BOARD[you] != ptn) && (bown >= bit); you--)
					bit<<=1;
				if (bown>bit) { return; }
				if (bown<bit) { break; }
			}
			//90度回転して同型なら180度/270度回転も同型である
			if (own>SIZEE) {
				// COUNT2++;
				info.setCount(0, 0, 1);
				return;
			}
		}
		//180度回転
		if (bsize==ENDBIT) {
			int own=1;
			for (int you=SIZEE-1; own<=SIZEE; own++, you--) {
				int bit = 1;
				for (int ptn=TOPBIT; (ptn!=BOARD[you])&&(BOARD[own]>=bit); ptn>>=1)
					bit<<=1;
				if (BOARD[own] > bit) { return; }
				if (BOARD[own] < bit) { break; }
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if (own>SIZEE) {
				// COUNT4++;
				info.setCount(0, 1, 0);
				return;
			}
		}
		//270度回転
		if (BOARD[BOUND1]==TOPBIT) {
			int own=1;
			for (int ptn=TOPBIT>>1; own<=SIZEE; own++, ptn>>=1) {
				int bit=1;
				for (int you=0; BOARD[you]!=ptn && BOARD[own]>=bit; you++) {
					bit<<=1;
				}
				if (BOARD[own]>bit) { return; }
				if (BOARD[own]<bit) { break; }
			}
		}
		//COUNT8++;
		info.setCount(1, 0, 0);
	}
}
class NQueen{
  // コンストラクタ
	public NQueen(){
		int max=28;
		NQueen8_Board info ;
		NQueen8_WorkEngine child ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(int size=2; size<max+1; size++){
			int nThreads=size ;
			info = new NQueen8_Board(size);
			long start = System.currentTimeMillis() ;
			try{
				child = new NQueen8_WorkEngine(size, nThreads, info, size-1, 0);
				child.start();
				child.join();
			}catch(Exception e){
				System.out.println(e);
			}
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,info.getTotal(),info.getUnique(),TIME);
		}
	}
}

