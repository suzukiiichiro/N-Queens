import org.apache.commons.lang3.time.DurationFormatUtils;

class Algorithm {
	public static void main(String[] args){
		new NQueen8_();
	}
}
class NQueen8_Board_ {
	private long COUNT8 ; private long COUNT4 ; private long COUNT2 ;
	private int limit ; private int nextCol ; private int nextID ;
	public NQueen8_Board_(int size) {
		COUNT8=COUNT4=COUNT2=0;
		nextCol=0; nextID=0;
		limit=(size+1)/2 ;
	}
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
	int SIZEE;
	int TOPBIT;
	int MASK ;
	int[] BOARD;
	int BOUND1;
	int BOUND2;
	int bit ;
	int SIDEMASK;
	int LASTMASK;
	int ENDBIT;
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
				// COUNT2++;
				info.setCount(0, 0, 1);
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
				// COUNT4++;
				info.setCount(0, 1, 0);
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
		//COUNT8++;
		info.setCount(1, 0, 0);
	}
	// 最上段のクイーンが角以外にある場合の探索
	public void backTrack2(int y, int left, int down, int right){
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
	public void backTrack1(int y, int left, int down, int right){
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
				backTrack1( y+1, (left|bit)<<1, down|bit, (right|bit)>>1) ;
			}
		}
	}
	private NQueen8_WorkEngine child  ;
	private NQueen8_Board_ info;
	private int size ;
	private int B1, B2;
	private int nMore ;
	public NQueen8_WorkEngine(int size, int nMore, NQueen8_Board_ info, int B1, int B2){
		this.size=size;
		this.info=info ;
		this.nMore=nMore ;
		BOARD = new int[size];
		BOARD[0] = 1;
		MASK=(1<<size)-1;
		SIZEE=size-1;
		TOPBIT = 1<<SIZEE;
		SIDEMASK=LASTMASK=(TOPBIT|1);
		ENDBIT = (TOPBIT>>1);
		if(nMore>0){
			try{
				this.B1=B1; 
				this.B2=B2;
//				child = new NQueen8_(size, nMore-1, info, B1+1, B2-1);
//				child.start();
			}catch(Exception e){
				System.out.println(e);
			}
		}else{
			child=null ;
		}
	}
	public void run() {
		if(child == null){
			B1=2; 
			while(B1>1 && B1<SIZEE) {
				BOUND1(B1);
				B1++;
			}
			B1=1; B2=size-2;
			while(B1>0 && B2<size-1 && B1<B2){
				BOUND2(B1, B2);
				B1++ ;B2--;
			}
		}
		if(child!=null){
			try{
				child.join();
			}catch(Exception e){
				System.out.println(e);
			}
		}
	}
	public void BOUND1(int B1){
		BOUND1=B1 ;
		if(BOUND1<SIZEE) {
			BOARD[1]=bit=(1<< BOUND1);
			backTrack1(2, (2|bit)<<1, (1|bit), (bit>>1));
		}
	}
	public void BOUND2(int B1, int B2){
		BOUND1=B1 ;
		BOUND2=B2;
		if(BOUND1<BOUND2) {
			BOARD[0]=bit=(1<<BOUND1);
			backTrack2(1, bit<<1, bit, bit>>1);
			LASTMASK|=LASTMASK>>1|LASTMASK<<1;
			ENDBIT>>=1;
		}
	}
}
class NQueen8_{
	public NQueen8_(){
		int max=27;
		NQueen8_Board_ info ;
		NQueen8_WorkEngine child ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(int size=2; size<max+1; size++){
			int nThreads=size ;
			info = new NQueen8_Board_(size);
			long start = System.currentTimeMillis() ;
			try{
				child = new NQueen8_WorkEngine(size, nThreads, info, 1, size-2);
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
