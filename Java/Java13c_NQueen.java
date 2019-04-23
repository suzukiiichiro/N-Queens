/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java13c_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java13c_NQueen  ;


 １３ｃ．マルチスレッド（完成）
 並列処理　マルチスレッド マルチスレッドの実装

　ここまでの処理は、一つのスレッドが順番にＡ行の１列目から順を追って処理判定をし
てきました。この節では、Ａ行の列それぞれに別々のスレッドを割り当て、全てのスレッ
ドを同時に処理判定させます。Ａ行それぞれの列の処理判定結果はBoardクラスで管理し、
処理完了とともに結果を出力します。スレッドはWorkEngineクラスがNの数だけ生成され
ます。WorkEngineクラスは自身の持ち場のＡ行＊列の処理だけを担当します。これらはマ
ルチスレッド処理と言い、並列処理のための同期、排他、ロック、集計など複雑な処理を
理解する知識が必要です。そして処理の最後に合計値を算出する方法をマルチスレッド処
理と言います。
　１Ｘ１，２Ｘ２，３Ｘ３，４Ｘ４，５Ｘ５，６Ｘ６，７ｘ７、８Ｘ８のボートごとの計
算をスレッドに割り当てる手法がちまたでは多く見受けられます。これらの手法は、実装
は簡単ですが、Ｎが７の計算をしながら別スレッドでＮが８の計算を並列処理するといっ
た矛盾が原因で、Ｎが大きくなるとむしろ処理時間がかかります。ここでは理想的なアル
ゴリズムとして前者の手法でプログラミングします。

実行結果

 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.001
 5:               10            2     00:00:00.001
 6:                4            1     00:00:00.000
 7:               40            6     00:00:00.001
 8:               92           12     00:00:00.001
 9:              352           46     00:00:00.001
10:              724           92     00:00:00.001
11:             2680          341     00:00:00.003
12:            14200         1787     00:00:00.002
13:            73712         9233     00:00:00.005
14:           365596        45752     00:00:00.021
15:          2279184       285053     00:00:00.102
16:         14772512      1846955     00:00:00.631
17:         95815104     11977939     00:00:04.253

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
  //マルチスレッド　ＯＮ＝true ＯＦＦ＝false
	private boolean		bThread	=true;
  //
	public WorkingEngine(int size,int nMore,Board info,int B1,int B2){
		this.size=size;
		this.info=info;
		this.nMore=nMore;
    //追加
		this.B1=B1;
		this.B2=B2;
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
		// シングルスレッド
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
		}else{
			//マルチスレッド
			board[0]=1;
			sizeE=size-1;
			MASK=(1<<size)-1;
			TOPBIT=1<<sizeE;
			if(B1>1&&B1<sizeE){
				BOUND1(B1);
			}
			ENDBIT=(TOPBIT>>B1);
			SIDEMASK=LASTMASK=(TOPBIT|1);
			if(B1>0&&B2<size-1&&B1<B2){
				for(int i=1;i<B1;i++){
					LASTMASK=LASTMASK|LASTMASK>>1|LASTMASK<<1;
				}
				BOUND2(B1,B2);
				ENDBIT>>=nMore;
			}
			try{
				child.join();
			}catch(Exception e){
				System.out.println(e);
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
class Java13c_NQueen{
  //
	public Java13c_NQueen(){
		Board info;
		WorkingEngine child;
		int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(int size=4;size<=max;size++){
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
      //追加
			info=null;
			child=null;
			System.gc();
		}
	}
  //
  public static void main(String[] args){
    new Java13c_NQueen();
  }
}
