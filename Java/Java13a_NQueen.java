/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java13a_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java13a_NQueen  ;


 １３a．マルチスレッド（準備１）

 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.000
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.000
 7:               40            6     00:00:00.000
 8:               92           12     00:00:00.000
 9:              352           46     00:00:00.000
10:              724           92     00:00:00.000
11:             2680          341     00:00:00.001
12:            14200         1787     00:00:00.003
13:            73712         9233     00:00:00.010
14:           365596        45752     00:00:00.055
15:          2279184       285053     00:00:00.327
16:         14772512      1846955     00:00:02.127
17:         95815104     11977939     00:00:14.962

 */
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Java13a_NQueen{
  //グローバル変数
  private int size;
  private int sizeE;
  private int board[];
  private int MASK;
  private int COUNT2,COUNT4,COUNT8;
  private int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  //
  public long getUnique(){
    return COUNT2+COUNT4+COUNT8;
  }
  //
  public long getTotal(){
    return COUNT2*2+COUNT4*4+COUNT8*8;
  }
  //新バージョンのsymmetryOps();
  public void symmetryOps(int bitmap){
    int own,ptn,you,bit;
    //90度回転
    if(board[BOUND2]==1){ own=1; ptn=2;
      while(own<=size-1){ bit=1; you=size-1;
        while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; you--; }
        if(board[own]>bit){ return; }
        if(board[own]<bit){ break; }
        own++; ptn<<=1;
      }
      /** 90度回転して同型なら180度/270度回転も同型である */
      if(own>size-1){ COUNT2++; return; }
    }
    //180度回転
    if(board[size-1]==ENDBIT){ own=1; you=size-1-1;
      while(own<=size-1){ bit=1; ptn=TOPBIT;
        while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; ptn>>=1; }
        if(board[own]>bit){ return; } 
        if(board[own]<bit){ break; }
        own++; you--;
      }
      /** 90度回転が同型でなくても180度回転が同型である事もある */
      if(own>size-1){ COUNT4++; return; }
    }
    //270度回転
    if(board[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
      while(own<=size-1){ bit=1; you=0;
        while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; you++; }
        if(board[own]>bit){ return; } 
        if(board[own]<bit){ break; }
        own++; ptn>>=1;
      }
    }
    COUNT8++;
  }
  //
  public void backTrack2(int row,int left,int down,int right){
    int bit;
    int bitmap=MASK&~(left|down|right); /* 配置可能フィールド */
    // 【枝刈り】
    //if(row==size){
    if(row==sizeE){
      //【枝刈り】 最下段枝刈り
      if(bitmap!=0){ //※【追加】
        if((bitmap&LASTMASK)==0){
          board[row]=bitmap;
          symmetryOps(bitmap);
        }
      }
    }else{
      //【枝刈り】上部サイド枝刈り
      if(row<BOUND1){
        bitmap&=~SIDEMASK;
      //【枝刈り】下部サイド枝刈り
      }else if(row==BOUND2) {
        if((down&SIDEMASK)==0){ return; }
        if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
      }
      while(bitmap>0){
        bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
        backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      }
    }
  }
  //
  public void backTrack1(int row,int left,int down,int right){
    int bit;
    int bitmap=MASK&~(left|down|right); /* 配置可能フィールド */
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    //if(row==size){
    if(row==(sizeE)){
      //if(bitmap==0){
      //枝刈り
      if(bitmap!=0){
        board[row]=bitmap;
        //枝刈り 処理せずにインクリメント
        //symmetryOps_bitmap();
        COUNT8++;
      }
    }else{
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if(row<BOUND1) {
        bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      while(bitmap>0){
        bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
        backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      }
    }
  }
  //新設
	private void BOUND2(int B1,int B2){
		int bit;
		BOUND1=B1;
		BOUND2=B2;
		board[0]=bit=(1<<BOUND1);
		backTrack2(1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
  //新設
	private void BOUND1(int B1){
		int bit;
		BOUND1=B1;
		board[1]=bit=(1<<BOUND1);
		backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
	}
  //
  public void NQueen(){
		board[0]=1;
		sizeE=size-1;
		MASK=(1<<size)-1;
		TOPBIT=1<<sizeE;
		BOUND1=2;
		while(BOUND1>1&&BOUND1<sizeE){
			BOUND1(BOUND1);
			BOUND1++;
		}
		SIDEMASK=LASTMASK=(TOPBIT|1);
		ENDBIT=(TOPBIT>>1);
		BOUND1=1;
		BOUND2=size-2;
		while(BOUND1>0&&BOUND2<size-1&&BOUND1<BOUND2){
			BOUND2(BOUND1,BOUND2);
			BOUND1++;
			BOUND2--;
		}
  //   int bit;
  //   board[0]=1;
  //   sizeE=size-1;
		// MASK=(1<<size)-1;
  //   TOPBIT=1<<(size-1);
  //   for(BOUND1=2;BOUND1<sizeE;BOUND1++){
  //     board[1]=bit=(1<<BOUND1);
  //     backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
  //   }
  //   SIDEMASK=LASTMASK=(TOPBIT|1);
  //   ENDBIT=(TOPBIT>>1);
  //   for(BOUND1=1,BOUND2=sizeE-1;BOUND1<BOUND2;BOUND1++,BOUND2--){
  //     board[0]=bit=(1<<BOUND1);
  //     backTrack2(1,bit<<1,bit,bit>>1);
  //     LASTMASK|=LASTMASK>>1|LASTMASK<<1;
  //     ENDBIT>>=1;
  //   }
  }
  //
  public Java13a_NQueen(){
    int min=4;
    int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
    for(int i=min;i<=max;i++){
      COUNT2=COUNT4=COUNT8=0;
      MASK=(1<<i)-1;
      board=new int[i+1];
      for(int j=0;j<=i;j++){
        board[j]=j;
      }
      size=i;
			long start=System.currentTimeMillis();
			NQueen(); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,getTotal(),getUnique(),TIME);
    }
  }
  //
	public static void main(String[] args){
		new Java13a_NQueen();
	}
}

