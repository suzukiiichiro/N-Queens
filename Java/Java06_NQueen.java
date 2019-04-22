/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

  コンパイル
  javac -cp .:commons-lang3-3.4.jar Java06_NQueen.java ;

  実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java06_NQueen  ;

 ６．ビットマップ(symmetryOps()以外の対応）

   ビット演算を使って高速化 状態をビットマップにパックし、処理する
   単純なバックトラックよりも２０〜３０倍高速
 
 　ビットマップであれば、シフトにより高速にデータを移動できる。
  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
  分。

 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
 　バックトラックよりも２０−３０倍高速。
 
 ===================
 考え方 1
 ===================

 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
 つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
 する)
 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
 ずれかのビット位置をひとつだけONにして進めていきます。

 
  - - - - - Q - -    00000100 0番目のビットフィールド
  - - - Q - - - -    00010000 1番目のビットフィールド
  - - - - - - Q -    00000010 2番目のビットフィールド
  Q - - - - - - -    10000000 3番目のビットフィールド
  - - - - - - - Q    00000001 4番目のビットフィールド
  - Q - - - - - -    01000000 5番目のビットフィールド
  - - - - Q - - -    00001000 6番目のビットフィールド
  - - Q - - - - -    00100000 7番目のビットフィールド


 ===================
 考え方 2
 ===================

 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。

 1. 左下に効き筋が進むもの: left 
 2. 真下に効き筋が進むもの: down
 3. 右下に効き筋が進むもの: right

次に、斜めの利き筋を考えます。
 上図の場合、
 1列目の右斜め上の利き筋は 3 番目 (0x08)
 2列目の右斜め上の利き筋は 2 番目 (0x04) になります。
 この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
 ことができます。
 また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目 (0x40)
になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。

つまり、右シフトの利き筋を right、左シフトの利き筋を left で表すことで、クイー
ンの効き筋はrightとleftを1 ビットシフトするだけで求めることができるわけです。

  *-------------
  | . . . . . .
  | . . . -3. .  0x02 -|
  | . . -2. . .  0x04  |(1 bit 右シフト right)
  | . -1. . . .  0x08 -|
  | Q . . . . .  0x10 ←(Q の位置は 4   down)
  | . +1. . . .  0x20 -| 
  | . . +2. . .  0x40  |(1 bit 左シフト left)  
  | . . . +3. .  0x80 -|
  *-------------
  図：斜めの利き筋のチェック

 n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
 の３つのビットフィールドとn番目のビットフィールド(bit)とのOR演算をそれぞれ行
 います。leftは左にひとつシフトし、downはそのまま、rightは右にひとつシフトして
 n+1番目のビットフィールド探索に渡してやります。

 left : (left |bit)<<1
 right: (right|bit)>>1
 down :   down|bit


 ===================
 考え方 3
 ===================

   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
 ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
 できない位置ということになります。次にその３つのビットフィールドをORしたビッ
 トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
 ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
 次の演算を行なってみます。
 
 bit = -bitmap & bitmap; //一番右のビットを取り出す
 
   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
 に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
 と次のようになります。
 
  00000011   3
  00000010   2
  00000001   1
  00000000   0
  11111111  -1
  11111110  -2
  11111101  -3
 
   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
 によって1ビット抽出を実現させていることが重要です。
 
      00010110   22
  AND 11101010  -22
 ------------------
      00000010
 
   さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
 生成されることになります。
 
 while (bitmap) {
     bit = -bitmap & bitmap;
     bitmap ^= bit;
     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
 }



実行結果

*/
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Java06_NQueen{
  //グローバル変数
	private int[]			board,trial,scratch;
	private int				size;
	private int[]	fA,fC,fB;
  private int bit;
  private int COUNT2,COUNT4,COUNT8;
  //
	private int intncmp(int[] lt,int[] rt,int n){
		int k,rtn=0;
		for(k=0;k<n;k++){
			rtn=lt[k]-rt[k];
			if(rtn!=0)
				break;
		}
		return rtn;
	}
	/* rotate +90 or -90: */
	private void rotate(int[] check,int[] scr,int n,boolean neg){
		int j,k;
		int incr;
		k=neg ? 0 : n-1;
		incr=(neg ? +1 : -1);
		for(j=0;j<n;k+=incr)
			scr[j++]=check[k];
		k=neg ? n-1 : 0;
		for(j=0;j<n;k-=incr)
			check[scr[j++]]=k;
	}
  //
	private void vMirror(int[] check,int n){
		int j;
		for(j=0;j<n;j++)
			check[j]=(n-1)-check[j];
		return;
	}
  //
  long getUnique(){
    return COUNT2+COUNT4+COUNT8;
  }
  //
  long getTotal(){
    return COUNT2*2+COUNT4*4+COUNT8*8;
  }
  //
	private void symmetryOps(){
		int nEquiv;
	  // 回転・反転・対称チェックのためにboard配列をコピー
		for(int k=0;k<size;k++){ trial[k]=board[k];}
    //時計回りに90度回転
		rotate(trial,scratch,size,false);
		int k=intncmp(board,trial,size);
		if(k>0) return ;
		if(k==0) nEquiv=2;
		else{
      //時計回りに180度回転
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0) return ;
			if(k==0) nEquiv=4;
			else{
        //時計回りに270度回転
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0) return ;
				nEquiv=8;
			}
		}
    //垂直反転
		for(k=0;k<size;k++) { trial[k]=board[k];}
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0) return ;
    if(nEquiv>2){
      //-90度回転 対角鏡と同等
      rotate(trial,scratch,size,true);
      k=intncmp(board,trial,size);
      if(k>0) return ;
      if(nEquiv>4){
        //-180度回転 水平鏡像と同等
        rotate(trial,scratch,size,true);
        k=intncmp(board,trial,size);
        if(k>0) return ;
          //-270度回転 反対角鏡と同等
          rotate(trial,scratch,size,true);
          k=intncmp(board,trial,size);
          if(k>0) return ;
      }
    }
    if(nEquiv==2){COUNT2++;}
    if(nEquiv==4){COUNT4++;}
    if(nEquiv==8){COUNT8++;}
	}
	// 再帰関数
	private void NQueen(int row,
                      int left,int down,int right){
		int mask=(1<<size)-1;
    int bitmap=mask&~(left|down|right);
    int tmp=0;
    if(row==size){
      if(bitmap!=1){
        board[row-1]=bitmap;
        /** symmetryOps() はまだ未改修のため以下の記述**/
        /** 次のステップで改修します */
        int lim=(row!=0)?size:(size+1)/2;
        for(int i=0;i<lim;i++){
          tmp=board[i];
          board[i]=size-1-((int)Math.log(board[i]));
          System.out.println("board : " + board[i]);
        }
        symmetryOps();
        for(int i=0;i<lim;i++){
          board[i]=tmp;
        }
      }
    }else{
      while(bitmap>0){
        bitmap^=board[row]=bit=(-bitmap&bitmap);
        NQueen(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      }
    }
	}
	// コンストラクタ
	public Java06_NQueen(){
		int max=17;
    int min=4;
    int mask=0;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=min;size<=max;size++){
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
      COUNT2=COUNT4=COUNT8=0;
			// fA=new int[i];
			// fC=new int[2*i-1];
			// fB=new int[2*i-1];
			for(int j=0;j<size;j++){
				board[j]=j;
			}
			long start=System.currentTimeMillis();
			NQueen(0,0,0,0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,getTotal(),getUnique(),TIME);
		}
	}
  //メインメソッド
	public static void main(String[] args){
		 new Java06_NQueen();
	}
}
