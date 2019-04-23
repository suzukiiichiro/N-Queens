/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java13b_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java13b_NQueen  ;


 １３ｂ．マルチスレッド（準備２）
 並列処理　シングルスレッド threadの実装

実行結果

 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.001
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.001
 7:               40            6     00:00:00.000
 8:               92           12     00:00:00.000
 9:              352           46     00:00:00.000
10:              724           92     00:00:00.001
11:             2680          341     00:00:00.001
12:            14200         1787     00:00:00.004
13:            73712         9233     00:00:00.010
14:           365596        45752     00:00:00.055
15:          2279184       285053     00:00:00.323
16:         14772512      1846955     00:00:02.089
17:         95815104     11977939     00:00:14.495

 */
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Board{
	private long COUNT8,COUNT4,COUNT2;
  //
	public Board(){
		COUNT8=COUNT4=COUNT2=0;
	}
  //
	public long getTotal(){
		return COUNT8*8+COUNT4*4+COUNT2*2;
	}
  //
	public long getUnique(){
		return COUNT8+COUNT4+COUNT2;
	}
  //
	public synchronized void setCount(long COUNT8,long COUNT4,long COUNT2){
		this.COUNT8+=COUNT8;
		this.COUNT4+=COUNT4;
		this.COUNT2+=COUNT2;
	}
}
//
class WorkingEngine extends Thread{
	private int[]			board;
	private int				MASK;
	private int				size,sizeE;
	private int				TOPBIT,ENDBIT,SIDEMASK,LASTMASK,BOUND1,BOUND2,B1,B2;
	private WorkingEngine	child;
	private Board			info;
	private int				nMore;
	private boolean		bThread	=false;
  //
	public WorkingEngine(int size,int nMore,Board info,int B1,int B2){
		this.size=size;
		this.info=info;
		this.nMore=nMore;
		board=new int[size];
		for(int k=0;k<size;k++){
			board[k]=k;
		}
		if(nMore>0){
			try{
				if(bThread){
					child=new WorkingEngine(size,nMore-1,info,B1-1,B2+1);
					child.start();
					//child.join();
				}
			}catch(Exception e){
				System.out.println(e);
			}
		}else{
			child=null;
		}
	}
  //
	private void BOUND2(int B1,int B2){
		int bit;
		BOUND1=B1;
		BOUND2=B2;
		board[0]=bit=(1<<BOUND1);
		backTrack2(1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
  //
	private void BOUND1(int B1){
		int bit;
		BOUND1=B1;
		board[1]=bit=(1<<BOUND1);
		backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
	}
  //
	public void run(){
		if(child==null){
			if(nMore>0){
				board[0]=1;
				sizeE=size-1;
				MASK=(1<<size)-1;
				TOPBIT=1<<sizeE;
				B1=2;
				while(B1>1&&B1<sizeE){
					BOUND1(B1);
					B1++;
				}
				SIDEMASK=LASTMASK=(TOPBIT|1);
				ENDBIT=(TOPBIT>>1);
				B1=1;
				B2=size-2;
				while(B1>0&&B2<size-1&&B1<B2){
					BOUND2(B1,B2);
					B1++;
					B2--;
				}
			}
		}
	}
  //
	private void symmetryOps(int bitmap){
		int own,you,ptn,bit;
		//90度回転
		if(board[BOUND2]==1){
			own=1;
			for(ptn=2;own<=sizeE;own++,ptn<<=1){
				bit=1;
				int bown=board[own];
				for(you=sizeE;(board[you]!=ptn)&&(bown>=bit);you--){ bit<<=1; }
				if(bown>bit){ return; }
				if(bown<bit){ break; }
			}
			//90度回転して同型なら180度/270度回転も同型である
			if(own>sizeE){
				//				COUNT2++;
				info.setCount(0,0,1);
				return;
			}
		}
		//180度回転
		if(bitmap==ENDBIT){
			own=1;
			for(you=sizeE-1;own<=sizeE;own++,you--){
				bit=1;
				for(ptn=TOPBIT;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1){ bit<<=1;}
				if(board[own]>bit){ return; }
				if(board[own]<bit){ break; }
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if(own>sizeE){
				//				COUNT4++;
				info.setCount(0,1,0);
				return;
			}
		}
		//270度回転
		if(board[BOUND1]==TOPBIT){
			own=1;
			for(ptn=TOPBIT>>1;own<=sizeE;own++,ptn>>=1){
				bit=1;
				for(you=0;board[you]!=ptn&&board[own]>=bit;you++){ bit<<=1; }
				if(board[own]>bit){ return; }
				if(board[own]<bit){ break; }
			}
		}
		//		COUNT8++;
		info.setCount(1,0,0);
	}
  //
	private void backTrack2(int row,int left,int down,int right){
		int bit;
		int bitmap=MASK&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				if((bitmap&LASTMASK)==0){
					board[row]=bitmap;
					symmetryOps(bitmap);
				}
			}
		}else{
			if(row<BOUND1){ bitmap&=~SIDEMASK; }
      else if(row==BOUND2){
				if((down&SIDEMASK)==0){ return; }
				if((down&SIDEMASK)!=SIDEMASK){
					bitmap&=SIDEMASK;
				}
			}
			while(bitmap>0){
        //最も下位の１ビットを抽出
				bitmap^=board[row]=bit=(-bitmap&bitmap); 
				backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
  //
	private void backTrack1(int row,int left,int down,int right){
		int bit;
		int bitmap=MASK&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				board[row]=bitmap;
				//				COUNT8++;
				info.setCount(1,0,0);
			}
		}else{
			if(row<BOUND1){ bitmap&=~2; }
			while(bitmap>0){
        //最も下位の１ビットを抽出
				bitmap^=board[row]=bit=(-bitmap&bitmap); 
				backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
}
//
class Java13b_NQueen{
  //
	public Java13b_NQueen(){
		Board info;
		WorkingEngine child;
		int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(int size=4;size<max;size++){
			int nThreads=size;
			long start=System.currentTimeMillis();
			info=new Board();
			try{
				child=new WorkingEngine(size,nThreads,info,size-1,0);
				child.start();
				child.join();
			}catch(Exception e){
				System.out.println(e);
			}
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,info.getTotal(),info.getUnique(),TIME);
		}
	}
  //
  public static void main(String[] args){
    new Java13b_NQueen();
  }
}
