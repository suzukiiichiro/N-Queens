/**

 Javaで学ぶ「アルゴリズムとデータ構造」
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版
 https://github.com/suzukiiichiro/N-Queen 
 			

  コンパイル
  javac -cp .:commons-lang3-3.4.jar Java05_NQueen.java ;

  実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Java05_NQueen  ;

 ４．バックトラック＋対称解除法＋枝刈り


実行結果
 N:            Total       Unique     hh:mm:ss.SSS
 4:                2            1     00:00:00.000
 5:               10            2     00:00:00.000
 6:                4            1     00:00:00.000
 7:               40            6     00:00:00.000
 8:               92           12     00:00:00.001
 9:              352           46     00:00:00.002
10:              724           92     00:00:00.002
11:             2680          341     00:00:00.006
12:            14200         1787     00:00:00.020
13:            73712         9233     00:00:00.107
14:           365596        45752     00:00:00.557
15:          2279184       285053     00:00:03.845
16:         14772512      1846955     00:00:24.166
17:         95815104     11977939     00:03:01.539

*/
//
import org.apache.commons.lang3.time.DurationFormatUtils;
//
class Java05_NQueen{
  //グローバル変数
	private int[]			board,trial,scratch;
	private int				size,nUnique,nTotal;
	private int[]	fA,fC,fB;
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
	int symmetryOps(){
		int k;
		int nEquiv;
	  // 回転・反転・対称チェックのためにboard配列をコピー
		for(k=0;k<size;k++)
			trial[k]=board[k];
    //時計回りに90度回転
		rotate(trial,scratch,size,false);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k==0)
			nEquiv=1;
		else{
      //時計回りに180度回転
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k==0)
				nEquiv=2;
			else{
        //時計回りに270度回転
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				nEquiv=4;
			}
		}
    //垂直反転
		for(k=0;k<size;k++)
			trial[k]=board[k];
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
    //-90度回転 対角鏡と同等
		rotate(trial,scratch,size,true);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k<0){
      //-180度回転 水平鏡像と同等
			rotate(trial,scratch,size,true);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k<0){
        //-270度回転 反対角鏡と同等
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
			}
		}
		return nEquiv*2;
	}
	// 再帰関数
	private void nQueens(int row){
    int tmp;
    //枝刈り
		//if(row==size){
		if(row==size-1){
      //枝刈り
      if((fB[row-board[row]+(size-1)]==1||fC[row+board[row]]==1)){
        return;
      }
			int tst=symmetryOps();
			if(tst!=0){
				nUnique++;
				nTotal+=tst;
			}
		}else{
      // 枝刈り
      int lim=(row!=0) ? size : (size+1)/2;
      for(int i=row;i<lim;i++){
			//for(int i=0;i<size;i++){
			//	board[row]=i;
			// 交換
			tmp=board[i];
			board[i]=board[row];
			board[row]=tmp;

			if(!(fB[row-board[row]+size-1]==1||fC[row+board[row]]==1)){
				fB[row-board[row]+size-1]=fC[row+board[row]]=1;
				nQueens(row+1); //再帰
				fB[row-board[row]+size-1]=fC[row+board[row]]=0;
			}
			//  if(fA[i]==0 &&fB[row-i+(size-1)]==0&&fC[row+i]==0){
			//	  fA[i]=fB[row-i+(size-1)]=fC[row+i]=1;
			//		nQueens(row+1);
			//	  fA[i]=fB[row-i+(size-1)]=fC[row+i]=0;
      //  }
			}
      // 交換
      tmp=board[row];
      for(int i=row+1;i<size;i++){
        board[i-1]=board[i];
      }
      board[size-1]=tmp;
    }
	}
	// コンストラクタ
	public Java05_NQueen(){
		int max=17;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<=max;size++){
			nUnique=nTotal=0;
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
			fA=new int[size];
			fC=new int[2*size-1];
			fB=new int[2*size-1];
			for(int i=0;i<size;i++){
				board[i]=i;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,nTotal,nUnique,TIME);
		}
	}
  //メインメソッド
	public static void main(String[] args){
		 new Java05_NQueen();
	}
}
