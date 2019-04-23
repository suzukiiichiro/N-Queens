/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

コンパイル
javac -cp .:commons-lang3-3.4.jar Java11_NQueen.java ;

実行
java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java11_NQueen  ;


 １１．枝刈り

  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない


  1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
    １行目かどうかの条件判断はループ外に出してもよい
    処理時間的に有意な差はないので、分かりやすいコードを示した
  2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
    １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    鏡像についても、主対角線鏡像のみを判定すればよい
    ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい

  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
  置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる
    ☓☓・・・Ｑ☓☓
    ☓・・・／｜＼☓
    ｃ・・／・｜・rt
    ・・／・・｜・・
    ・／・・・｜・・
    lt・・・・｜・ａ
    ☓・・・・｜・☓
    ☓☓ｂ・・dn☓☓
    
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む


  実行結果
 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.000
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.001
 7:               40            6     00:00:00.000
 8:               92           12     00:00:00.001
 9:              352           46     00:00:00.001
10:              724           92     00:00:00.001
11:             2680          341     00:00:00.005
12:            14200         1787     00:00:00.009
13:            73712         9233     00:00:00.048
14:           365596        45752     00:00:00.267
15:          2279184       285053     00:00:01.754
16:         14772512      1846955     00:00:12.398
17:         95815104     11977939     00:01:34.783
 */
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Java11_NQueen{
  //グローバル変数
  private int size;
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
  public void backTrack2(int row,int left,int down,int right){
    int bit;
    int bitmap=MASK&~(left|down|right); /* 配置可能フィールド */
    // 【枝刈り】
    //if(row==size){
    if(row==size-1){
      //【枝刈り】 最下段枝刈り
      //if(bitmap==0){
      if((bitmap&LASTMASK)==0){
        board[row]=bitmap;
        symmetryOps_bitmap();
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
    if(row==size-1){
      //if(bitmap==0){
      if(bitmap>0){
        board[row]=bitmap;
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
    int bit;
    TOPBIT=1<<(size-1);
    board[0]=1;
    for(BOUND1=2;BOUND1<size-1;BOUND1++){
      board[1]=bit=(1<<BOUND1);
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
    }
    SIDEMASK=LASTMASK=(TOPBIT|1);
    ENDBIT=(TOPBIT>>1);
    for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
      board[0]=bit=(1<<BOUND1);
      backTrack2(1,bit<<1,bit,bit>>1);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
    }
  }
  //
  public Java11_NQueen(){
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
		new Java11_NQueen();
	}
}

