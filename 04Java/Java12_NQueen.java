/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java12_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java12_NQueen  ;


 １２．最適化


  実行結果
【旧バージョン】
 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.000
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.000
 7:               40            6     00:00:00.001
 8:               92           12     00:00:00.000
 9:              352           46     00:00:00.001
10:              724           92     00:00:00.002
11:             2680          341     00:00:00.002
12:            14200         1787     00:00:00.008
13:            73712         9233     00:00:00.025
14:           365596        45752     00:00:00.133
15:          2279184       285053     00:00:00.875
16:         14772512      1846955     00:00:06.017
17:         95815104     11977939     00:00:44.417

【新バージョン】
 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.000
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.000
 7:               40            6     00:00:00.000
 8:               92           12     00:00:00.000
 9:              352           46     00:00:00.000
10:              724           92     00:00:00.001
11:             2680          341     00:00:00.001
12:            14200         1787     00:00:00.002
13:            73712         9233     00:00:00.011
14:           365596        45752     00:00:00.055
15:          2279184       285053     00:00:00.324
16:         14772512      1846955     00:00:02.089
17:         95815104     11977939     00:00:14.524

 */
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Java12_NQueen{
  //グローバル変数
  private int size;
  private int sizeE;
  private int board[];
  private int MASK;
  private int aT[];
  private int aS[];
  private int bit;
  private int COUNT2,COUNT4,COUNT8;
  private int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  //
  public int rh(int a,int sz){
    int tmp=0;
    for(int i=0;i<=sz;i++){
      if( (a&(1<<i))!=0){ return tmp|=(1<<(sz-i)); }
    }
    return tmp;
  }
  //
  public void vMirror_bitmap(int bf[],int af[],int si){
    int score ;
    for(int i=0;i<si;i++) {
      score=bf[i];
      af[i]=rh(score,si-1);
    }
  }
  //
  public void rotate_bitmap(int bf[],int af[],int si){
    for(int i=0;i<si;i++){
      int t=0;
      for(int j=0;j<si;j++){
        t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
      }
      af[i]=t;                        // y[i] の j ビット目にする
    }
  }
  //
  public int intncmp(int lt[],int rt[],int n){
    int rtn=0;
    for(int k=0;k<n;k++){
      rtn=lt[k]-rt[k];
      if(rtn!=0){
        break;
      }
    }
    return rtn;
  }
  //
  public long getUnique(){
    return COUNT2+COUNT4+COUNT8;
  }
  //
  public long getTotal(){
    return COUNT2*2+COUNT4*4+COUNT8*8;
  }
  //
  //旧バージョンのsymmetryOps();
  public void symmetryOps_bitmap(){
    int nEquiv;
    // 回転・反転・対称チェックのためにboard配列をコピー
    for(int i=0;i<size;i++){ aT[i]=board[i];}
    rotate_bitmap(aT,aS,size);    //時計回りに90度回転
    int k=intncmp(board,aS,size);
    if(k>0)return;
    if(k==0){ nEquiv=2;}else{
      rotate_bitmap(aS,aT,size);  //時計回りに180度回転
      k=intncmp(board,aT,size);
      if(k>0)return;
      if(k==0){ nEquiv=4;}else{
        rotate_bitmap(aT,aS,size);//時計回りに270度回転
        k=intncmp(board,aS,size);
        if(k>0){ return;}
        nEquiv=8;
      }
    }
    // 回転・反転・対称チェックのためにboard配列をコピー
    for(int i=0;i<size;i++){ aS[i]=board[i];}
    vMirror_bitmap(aS,aT,size);   //垂直反転
    k=intncmp(board,aT,size);
    if(k>0){ return; }
    if(nEquiv>2){             //-90度回転 対角鏡と同等
      rotate_bitmap(aT,aS,size);
      k=intncmp(board,aS,size);
      if(k>0){return;}
      if(nEquiv>4){           //-180度回転 水平鏡像と同等
        rotate_bitmap(aS,aT,size);
        k=intncmp(board,aT,size);
        if(k>0){ return;}       //-270度回転 反対角鏡と同等
        rotate_bitmap(aT,aS,size);
        k=intncmp(board,aS,size);
        if(k>0){ return;}
      }
    }
    if(nEquiv==2){COUNT2++;}
    if(nEquiv==4){COUNT4++;}
    if(nEquiv==8){COUNT8++;}
  }
  //
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
          //
          //旧バージョン
          //symmetryOps_bitmap();
          //
          //新バージョン
          symmetryOps(bitmap);
          //
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
  //
  public void NQueen(){
    //int bit;
    board[0]=1;
    sizeE=size-1;
		MASK=(1<<size)-1;
    TOPBIT=1<<(size-1);
    for(BOUND1=2;BOUND1<sizeE;BOUND1++){
      board[1]=bit=(1<<BOUND1);
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
    }
    SIDEMASK=LASTMASK=(TOPBIT|1);
    ENDBIT=(TOPBIT>>1);
    for(BOUND1=1,BOUND2=sizeE-1;BOUND1<BOUND2;BOUND1++,BOUND2--){
      board[0]=bit=(1<<BOUND1);
      backTrack2(1,bit<<1,bit,bit>>1);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
    }
  }
  //
  public Java12_NQueen(){
    int min=4;
    int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
    for(int i=min;i<=max;i++){
      COUNT2=COUNT4=COUNT8=0;
      MASK=(1<<i)-1;
      board=new int[i+1];
      aT=new int[i+1];
      aS=new int[i+1];
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
		new Java12_NQueen();
	}
}

