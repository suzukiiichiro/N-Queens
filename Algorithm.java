import org.apache.commons.lang3.time.DurationFormatUtils;


/**
 * Javaで学ぶアルゴリズムとデータ構造  
 * ステップバイステップでＮ−クイーン問題を最適化
 * 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 * 
 * 目次
 * 1. ソートアルゴリズム
 *   バブルソート
 *   選択ソート
 *   挿入ソート
 *   マージソート
 *   シェルソート
 *   クイックソート
 * 
 * 2. 再帰
 *   三角数
 *   階乗
 *   ユークリッドの互除法
 *   ハノイの塔
 * 
 * 3.  Nクイーン問題
 *  １．ブルートフォース（力まかせ探索） NQueen1() * N 8: 00:04:15
 *  ２．バックトラック                   NQueen2() * N 8: 00:00:01
 *  ３．配置フラグ（制約テスト高速化）   NQueen3() * N16: 00:01:35
 *  ４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 *  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 *  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 *  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 *  ８．マルチスレッド2                  NQueen9() * N16: 00:00:00
 */

/**
 N             Total解        Unique解    鈴木維版       QJH版   takaken版  電通大版  JeffSomers版
 15         2,279,184         285,053     00:00:00    00:00:00    00:00:00  00:00:01    00:00:04
 16        14,772,512       1,846,955     00:00:00    00:00:00    00:00:04  00:00:08    00:00:23
 17        95,815,104      11,977,939     00:00:04    00:00:07    00:00:31  00:01:01    00:02:38
 18       666,090,624      83,263,591     00:00:33    00:00:25    00:03:48  00:07:00    00:19:26
 19     4,968,057,848     621,012,754     00:04:19    00:03:17    00:29:22  00:57:16    02:31:24
 20    39,029,188,884   4,878,666,808     00:34:49    00:24:07    03:54:10  07:19:24    20:35:06
 21   314,666,222,712  39,333,324,973     04:41:36    03:05:28 01:09:17:19  　  　
 22 2,691,008,701,644 336,376,244,042  01:15:14:59 01:03:08:20  　  　  　
*/

public class Algorithm {
  public static void main(String[] args){

/************************************************************
 * １．ソート
 */

/**
 * 1. ソート  バブルソート 13404mm
 * https://ja.wikipedia.org/wiki/バブルソート
 * https://www.youtube.com/watch?v=8Kp-8OGwphY
 *   平均計算時間が O(ｎ^2)
 *   安定ソート
 *   比較回数は「  n(n-1)/2  」
 *   交換回数は「  n^2/2  」
 *   派生系としてシェーカーソートやコムソート
 */

class Sort01_BubbleSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort01_BubbleSort(){
		long start=System.currentTimeMillis();
		setArray();
		display();
		bubbleSort(a);
		display();
		long end=System.currentTimeMillis();
		System.out.println("bubble : " + (end-start));
	}
	void bubbleSort(int[] a){
		int out, in;
		for(out=nElems-1; out>1;out--){
			for(in=0; in<out; in++){
				if(a[in]>a[in+1])
					swap(a, in, in+1);
			}
		}
	}
}

  // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
  // new Sort01_BubbleSort();

/**
 * 選択ソート 3294mm
  * https://ja.wikipedia.org/wiki/選択ソート
 * https://www.youtube.com/watch?v=f8hXR_Hvybo
 *   平均計算時間が O(ｎ^2)
 *   安定ソートではない
 *   比較回数は「  n(n-1)/2  」
 *   交換回数は「  n-1  」
 */

class Sort02_SelectionSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort02_SelectionSort(){
		long start=System.currentTimeMillis();
		setArray();
		display();
		selectionSort(a);
		display();
		long end=System.currentTimeMillis();
		System.out.println("select : " + (end-start));
	}
	void selectionSort(int[] a){
		int out, in, min ;
		for(out=0; out<nElems; out++){
			min=out;
			for(in=out+1; in<nElems; in++){
				if(a[in]<a[min])
					min=in;
			}
			swap(a, out, min);
		}
	}
}

   // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
   // new Sort02_SelectionSort();

/**
 * 挿入ソート 3511mm
  * https://ja.wikipedia.org/wiki/挿入ソート
 * https://www.youtube.com/watch?v=DFG-XuyPYUQ
 *   平均計算時間が O(ｎ^2)
 *   安定ソート
 *   比較回数は「  n(n-1)/2以下  」
 *   交換回数は「  約n^2/2以下  」
 */

class Sort03_InsertionSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort03_InsertionSort (){
		long start=System.currentTimeMillis();
		setArray();
		display();
		insertionSort(a);
		display();
		long end=System.currentTimeMillis();
		System.out.println("insert : " + (end-start));
	}
	void insertionSort(int[] a){
		int out, in, tmp;
		for(out=1; out<nElems; out++){
			tmp=a[out];
			for(in=out; in>0 && a[in-1]>=tmp; --in){
				a[in]=a[in-1];
			}
			a[in]=tmp ;
		}
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new Sort03_InsertionSort();

/**
 * マージソート 1085mm
  * https://ja.wikipedia.org/wiki/マージソート
 * https://www.youtube.com/watch?v=EeQ8pwjQxTM
 *   平均計算時間が O(n Log n)
 *   安定ソート
 *   50以下は挿入ソート、5万以下はマージソート、あとはクイックソートがおすすめ。
 *   バブルソート、挿入ソート、選択ソートがO(N^2)の時間を要するのに対し、マージ
 *   ソートはO(N*logN)です。
 *   例えば、N(ソートする項目の数）が10,000ですと、N^2は100,000,000ですが、
 *   n*logNは40,000です。別の言い方をすると、マージソートで４０秒を要するソート
 *   は、挿入ソートでは約２８時間かかります。
 *   マージソートの欠点は、ソートする配列と同サイズの配列をもう一つ必要とする事
 *   です。
 *   元の配列がかろうじてメモリに治まるという大きさだったら、マージソートは使え
 *   ません。
 */

class Sort04_MergeSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort04_MergeSort(){
		long start=System.currentTimeMillis();
		setArray();
		display();
		int[] w = new int[nElems];
		mergeSort(a, w, 0, nElems-1);
		display();
		long end=System.currentTimeMillis();
		System.out.println("merge : " + (end-start));
	}
	void mergeSort(int[] a, int[] w, int low, int up){
		if(low==up)
			return ;
		else{
			int mid=(low+up)/2;
			mergeSort(a, w, low, mid);
			mergeSort(a, w, mid+1, up);
			merge(a, w, low, mid+1, up);
		}
	}
	void merge(int[]a, int[]w, int low, int high, int upB){
		int j=0;
		int lowB=low;
		int mid=high-1;
		int n=upB-lowB+1;
		while(low<=mid && high<=upB)
			if(a[low]<a[high])
				w[j++]=a[low++];
			else
				w[j++]=a[high++];
		while(low<=mid)
			w[j++]=a[low++];
		while(high<=upB)
			w[j++]=a[high++];
		for(j=0;j<n;j++)
			a[lowB+j]=w[j];
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new Sort04_MergeSort();

/**
 * シェルソート 1052mm
  * https://ja.wikipedia.org/wiki/シェルソート
 * https://www.youtube.com/watch?v=M9YCh-ZeC7Y
 *   平均計算時間が O(ｎ^1.25)
 *   安定ソートではない
 *   挿入ソート改造版
 *   ３倍して１を足すという処理を要素を越えるまで行う
 */

class Sort05_ShellSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort05_ShellSort(){
		long start=System.currentTimeMillis();
		setArray();
		display();
		shellSort(a);
		display();
		long end=System.currentTimeMillis();
		System.out.println("shell : " + (end-start));
	}
	void shellSort(int[]a){
		int out, in, tmp ;
		int h=1;
		while(h<=nElems/3)
			h=h*3+1;
		while(h>0){
			for(out=h;out<nElems; out++){
				tmp=a[out];
				in=out;
				while(in>h-1 && a[in-h]>=tmp){
					a[in]=a[in-h];
					in-=h;
				}
				a[in]=tmp;
			}
			h=(h-1)/3;
		}
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new Sort05_ShellSort();

/**
 * クイックソート 1131mm
  * https://ja.wikipedia.org/wiki/クイックソート
 * https://www.youtube.com/watch?v=aQiWF4E8flQ
 *   平均計算時間が O(n Log n)
 *   安定ソートではない
 *   最大計算時間が O(n^2)
 *
 * データ数が 50 以下なら挿入ソート (Insertion Sort)
 * データ数が 5 万以下ならマージソート (Merge Sort)
 * データ数がそれより多いならクイックソート (Quick Sort)
 */

class Sort06_QuickSort{
	int[] a = null ;
	int nElems=0;
	int maxSize=100000;
	void setArray(){
		a = new int[maxSize];
		for(int i=0; i<maxSize; i++){
			a[i]=(int)(Math.random()*1000000);
			nElems++;
		}
	}
	void display(){
		for(int i=0; i<nElems; i++){
			System.out.println(a[i]);
		}
	}
	void swap(int[] a, int one, int two){
		int tmp=a[one];
		a[one]=a[two];
		a[two]=tmp;
	}
	public Sort06_QuickSort(){
		long start=System.currentTimeMillis();
		setArray();
		display();
		quickSort(a, 0, nElems-1);
		insertionSort(a, 0, nElems-1);
		display();
		long end=System.currentTimeMillis();
		System.out.println("quick : " + (end-start));
	}
	void quickSort(int[]a, int left, int right){
		int size=right-left+1;
		if(size<10)
			insertionSort(a, left, right);
		else{
			int median=medianOf3(left, right);
			int part=getPart(left, right, median);
			quickSort(a, left, part-1);
			quickSort(a, part+1, right);
		}
	}
	int medianOf3(int left, int right){
		int center=(left+right)/2;
		if(a[left]>a[center])
			swap(a, left, center);
		if(a[left]>a[right])
			swap(a, left, right);
		if(a[center]>a[right])
			swap(a, center, right);
		swap(a, center, right-1);
		return a[right-1];
	}
	int getPart(int left, int right, int pivot){
		int leftP=left ;
		int rightP=right-1;
		while(true){
			while(a[++leftP]<pivot)
				;
			while(a[--rightP]>pivot)
				;
			if(leftP>=rightP)
				break ;
			else
				swap(a, leftP, rightP);
		}
		swap(a, leftP, right-1);
		return leftP;
	}
	void insertionSort(int[]a, int left, int right){
		int in , out ;
		for(out=left+1; out<=right; out++){
			int tmp=a[out];
			in=out;
			while(in>left && a[in-1]>=tmp){
				a[in]=a[in-1];
				--in;
			}
			a[in]=tmp;
		}
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new Sort06_QuickSort();

/************************************************************
 * ２．再帰
 */

/**
 * 三角数
 * https://ja.wikipedia.org/wiki/三角数
 * 5 + 4 + 3 + 2 + 1 = 15
 */

class Recursive01_Triangle{
	int triangle(int n){
		if(n==1)
			return 1;
		else
			return n+triangle(n-1);
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // System.out.println(new Recursive01_Triangle().triangle(5));

/**
 * 階乗
 * https://ja.wikipedia.org/wiki/階乗
 * 5 * 4 * 3 * 2 * 1 = 120 
 */

class Recursive02_Factorial{
	int factorial(int n){
		if(n==1)
			return 1;
		else 
			return n*factorial(n-1);
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // System.out.println(new Recursive02_Factorial().factorial(5));

/**
 * ユークリッドの互除法
 * https://ja.wikipedia.org/wiki/ユークリッドの互除法
 * (問題) 1071 と 1029 の最大公約数を求める。
 * 1071 を 1029 で割った余りは 42
 * 1029 を 42 で割った余りは 21
 * 42 を 21 で割った余りは 0
 * よって、最大公約数は21である。
 */

class Recursive03_Euclid{
	int euclid(int x, int y){
		if(y==0)
			return x ;
		else
			return euclid(y, x%y);
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // System.out.println(new Recursive03_Euclid().euclid(1071, 1029));

/**
 * ハノイの塔
 * https://ja.wikipedia.org/wiki/ハノイの塔
 *  古代インドの神話では、遠い秘境のお寺で僧侶達が毎日毎晩、
 *  ６４枚の黄金の円盤をダイヤモンドをちりばめた３つの塔の間で移し替え作業をし
 *  ている。その移し替え作業が完了したら世界の終末が訪れるのだそうだ。
 */

 /**
 *   n3: 7
 *   n4: 15 
 *   n5: 31
 *   n6: 63
 *   n7: 127
 *   n8: 255
 *   n9: 511
 *   n10:1,023
 *   n20:1,048,575
 *   n30:1,073,741,823
 *   n40:1,099,511,627,775
 *   n50:1,125,899,906,842,620
 *   n60:1,152,921,504,606,850,000
 *   n70:1,180,591,620,717,410,000,000
 *   n80:1,208,925,819,614,630,000,000,000
 *   n90:1,237,940,039,285,380,000,000,000,000
 *   n100:1,267,650,600,228,230,000,000,000,000,000
 */

class Recursive04_Hanoi{
	int count=1 ;
	void hanoi(int n, char src, char inter, char dest){
		if(n==1)
			System.out.println("work:" +count++ + " disk1"+src+"to"+dest);
		else{
			hanoi(n-1, src, dest, inter);
			System.out.println("work:" +count++ + " disk"+n+"from"+src+"to"+dest);
			hanoi(n-1, inter, src, dest);
		}	
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new Recursive04_Hanoi().hanoi(20, 'A', 'B', 'C');

/************************************************************
 * ３．再帰  N-クイーン問題
 */

/**
 * 再帰  Nクイーン問題
 *
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
 */

/**
 *   ステップ
 * 
 *  １．ブルートフォース（力まかせ探索） NQueen1() * N 8: 00:04:15 ※Nが8です
 *  ２．バックトラック                   NQueen2() * N 8: 00:00:01 ※Nが8です
 *  ３．配置フラグ（制約テスト高速化）   NQueen3() * N16: 00:01:35
 *  ４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 *  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 *  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 *  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 *  ８．マルチスレッド2                  NQueen9() * N16: 00:00:00
*/
    
/** 
 * １．ブルートフォース（力まかせ探索）
 *　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
 *  なるかをチェックする方法
 *  (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
 *  けであって8王妃問題を解いているわけではありません
 */

  /**
  :
  :
  7 7 7 7 7 7 6 7 : 16777208
  7 7 7 7 7 7 7 0 : 16777209
  7 7 7 7 7 7 7 1 : 16777210
  7 7 7 7 7 7 7 2 : 16777211
  7 7 7 7 7 7 7 3 : 16777212
  7 7 7 7 7 7 7 4 : 16777213
  7 7 7 7 7 7 7 5 : 16777214
  7 7 7 7 7 7 7 6 : 16777215
  7 7 7 7 7 7 7 7 : 16777216
  */

class NQueen1{
	// 各列に１個の王妃を配置する組み合わせを再帰的に列挙
	private int[] board ;
	private int count;
	private int size;
	// コンストラクタ
	public NQueen1(int size){
		this.size=size;
		board=new int[size];
		// 解数は1からカウント
		count=1;
		// ０列目に王妃を配置してスタート
		nQueens(0);
	}
	// 再帰関数
	private void nQueens(int row){
		// 全列に配置完了 最後の列で出力
		if(row==size){
			print();
		}else{
			// 各列にひとつのクイーンを配置する
			for(int col=0; col<size; col++){
				board[row]=col; 
				// 次の列に王妃を配置
				nQueens(row+1);
			}
		}
	}
	//出力
	private void print(){
		for(int col=0; col<size; col++){
			System.out.printf("%2d", board[col]);
		}
		System.out.println(" : " + count++);
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen1(8);  //ブルートフォース

/** 
 * ２．バックトラック
 *  パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
 *  い事が明らかな場合は、それ以降のパターン生成を行わない。
 * 「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
 * 事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
 * (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
 * み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
 */

  /**
  :
  :
  7 6 5 4 2 0 3 1 : 40310
  7 6 5 4 2 1 0 3 : 40311
  7 6 5 4 2 1 3 0 : 40312
  7 6 5 4 2 3 0 1 : 40313
  7 6 5 4 2 3 1 0 : 40314
  7 6 5 4 3 0 1 2 : 40315
  7 6 5 4 3 0 2 1 : 40316
  7 6 5 4 3 1 0 2 : 40317
  7 6 5 4 3 1 2 0 : 40318
  7 6 5 4 3 2 0 1 : 40319
  7 6 5 4 3 2 1 0 : 40320
                         N16: 00:00:01
  */

class NQueen2{
	private int[] board ;
	private int count;
	private int size;
	//行の配置フラグ
	private boolean flag[] ;
	public NQueen2(int size){
		this.size=size ;
		// 解数は1からカウント
		count=1;
		board=new int[size];
		//行の配置フラグ
		flag=new boolean[size];
		// ０列目に王妃を配置
		nQueens(0);
	}
	private void nQueens(int row){
		// 全列+各行に一つの王妃を配置完了 最後の列で出力
		if(row==size){
			print();
		}else{
			// 各列にひとつのクイーンを配置する
			for(int col=0; col<size; col++){
				// i行には王妃は未配置 バックトラック
				if(flag[col]==false){
					// 王妃をi行に配置
					board[row]=col; 
					// i行に王妃を配置したらtrueに
					/**
					 * i行に王妃を配置したらtrueに
					 * これはその行に重複して王妃を配置しないようにするため
					 * falseである行に対してのみ王妃を配置します
					 */
					flag[col]=true;
					// 次の列に王妃を配置
					nQueens(row+1);
					/**
					 * 再帰的に呼び出したメソッドset()から戻ってきたときは
					 * flag[i]をfalseに設定することによってi列から王妃を取り除きます。
					 */
					flag[col]=false ;
				}
			}
		}
	}
	private void print(){
		for(int col=0; col<size; col++){
			System.out.printf("%2d", board[col]);
		}
		System.out.println(" : " + count++);
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen2(8);  //バックトラック

/**
 * ３．配置フラグ（制約テスト高速化）
 *  　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 *  さない事が明らかな場合は、それ以降のパターン生成を行わない。
 *  　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 *  　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 *  は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 *  けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 *  を降りても解がないと判明した時点で木を引き返すということができます。
 */

  /**
   * 実行結果
   N:            Total       Unique    hh:mm:ss
   2:                0            0  00:00:00
   3:                0            0  00:00:00
   4:                2            0  00:00:00
   5:               10            0  00:00:00
   6:                4            0  00:00:00
   7:               40            0  00:00:00
   8:               92            0  00:00:00
   9:              352            0  00:00:00
  10:              724            0  00:00:00
  11:             2680            0  00:00:00
  12:            14200            0  00:00:00
  13:            73712            0  00:00:00
  14:           365596            0  00:00:02
  15:          2279184            0  00:00:14
  16:         14772512            0  00:01:35
  */

class NQueen3{
	private int[] board ;
	private long TOTAL ;
	private int size;
	private int max ;
	private boolean[] colChk;    // セル
    private boolean[] diagChk;   // 対角線
    private boolean[] antiChk;   // 反対角線
	public NQueen3(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(size=2; size<max; size++){
			TOTAL=0;
			board=new int[size];
			colChk    = new boolean[size];
			diagChk   = new boolean[2*size-1];
			antiChk   = new boolean[2*size-1];
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			nQueens(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,TOTAL,0,TIME); 
		}
	}
	private void nQueens(int row){
		if(row==size){
			TOTAL++ ;
		}else{
			for(int col=0; col<size; col++){
				board[row]=col; 
				if(	colChk[col]==false && antiChk[row+col]==false && diagChk[row-col+(size-1)]==false){
					colChk[col]=diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
					nQueens(row+1);
					colChk[col]=diagChk[row-board[row]+size-1] = antiChk[row+board[row]] =false;
				}
			}
		}
	}
}

    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen3(); //配置フラグ

/** 
 * ４．対称解除法
 *     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 *     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 * 
 * ■ユニーク解の判定方法
 *   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 * ことを意味しています。
 * 
 *   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 * わりません。
 * 
 *   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 * 下の例では要素数が２個のものと８個のものがあります。
 *
 *
 * Ｎ＝５の全解は１０、ユニーク解は２なのです。
 * 
 * グループ１: ユニーク解１つ目
 * - - - Q -   - Q - - -
 * Q - - - -   - - - - Q
 * - - Q - -   - - Q - -
 * - - - - Q   Q - - - -
 * - Q - - -   - - - Q -
 * 
 * グループ２: ユニーク解２つ目
 * - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 * - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 * Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 * - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 * - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
 *
 * 
 *   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 * ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 * の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 * をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 * と呼ぶことにします。
 * 
 * - - - - Q   0
 * - - Q - -   2
 * Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 * - - - Q -   1
 * - Q - - -   3
 * 
 * 
 *   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 * 変換を試みるだけでユニーク解の判定が可能になります。
 *  
 * 
 * ■ユニーク解の個数を求める
 *   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 * ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 * 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 * うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 * れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 * 央になければならず、それは互いの効き筋にあたるので有り得ません。
 *
 *
 * ***********************************************************************
 * 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
 * ***********************************************************************
 * 
 *   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
 * の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
 * を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
 * ます。突き詰めれば、
 * 
 * [上から２行目のクイーンの位置が右から何番目にあるか]
 * [右から２列目のクイーンの位置が上から何番目にあるか]
 * 
 *
 * を比較するだけで判定することができます。この２つの値が同じになることはないからです。
 * 
 *       3 0
 *       ↓↓
 * - - - - Q ←0
 * - Q - - - ←3
 * - - - - -         上から２行目のクイーンの位置が右から４番目にある。
 * - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
 * - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
 * 
 *   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 * られる解は総てユニーク解であることが保証されます。
 * 
 * - - - - X Q
 * - Q - - X -
 * - - - - X -
 * - - - - X -
 * - - - - - -
 * - - - - - -
 * 
 *   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 * ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 * ます。
 * 
 * X X - - - Q X X
 * X - - - - - - X
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * X - - - - - - X
 * X X - - - - X X
 * 
 *   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 * 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 * て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 * になります。
 * 
 * X X x x x Q X X
 * X - - - x x x X
 * C - - x - x - x
 * - - x - - x - -
 * - x - - - x - -
 * x - - - - x - A
 * X - - - - x - X
 * X X B - - x X X
 *
 *
 * ■ユニーク解から全解への展開
 *   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 * ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 * 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 * 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 * を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 * れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 * いう考察が必要になってきます。
 * 
 *   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 * オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 * 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 * ず８個(＝２×４)になります。
 * 
 *   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 * ために次の事柄を確認します。
 *
 * TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
 *   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
 *    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
 *    ジナルと同型になる。  
 *
 *    COUNT2 * 2
 * 
 *   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
 *    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
 *    り得る。 
 *
 *    COUNT4 * 4
 * 
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 * 
 *    COUNT8 * 8 
 *
 *   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 * りを外す必要がなくなったというわけです。 
 * 
 *   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
 *   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 *
 * 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 * ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
 * 躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
 *
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
  14:           365596        45752  00:00:02
  15:          2279184       285053  00:00:16
  16:         14772512      1846955  00:01:50
   */

class NQueen4{
	private int[] board ;
	private long nUnique ; //ユニーク解
	private long nTotal ;  //合計解
	private int size;
	private int max ;
	private boolean[] colChk;
    private boolean[] diagChk;
    private boolean[] antiChk;
	private int[] trial ; //回転・反転の対称チェック
	private int[] scratch ;
	public NQueen4(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.size=2; size<max; size++){
			board=new int[size];
			colChk    = new boolean[size];
			diagChk   = new boolean[2*size-1];
			antiChk   = new boolean[2*size-1];
			trial     = new int[size]; //回転・反転の対称チェック
			scratch   = new int[size];
			nUnique=0;	//ユニーク解
			nTotal=0;	//合計解 
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			nQueens(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,getTotal(),getUnique(),TIME); 
		}
	}
	private void nQueens(int row){
		if(row==size){
			//回転・反転・対称の解析
	         int tst = symmetryOps ();
	         if (tst != 0) {
	            nUnique++;
	            nTotal += tst;
	         }
		}else{
			for(int col=0; col<size; col++){
				board[row]=col; 
				if(colChk[col]==false && antiChk[row+col]==false && diagChk[row-col+(size-1)]==false){
					colChk[col]=antiChk[row+col]=diagChk[row-col+(size-1)]=true;
					nQueens(row+1);
					colChk[col]=antiChk[row+col]=diagChk[row-col+(size-1)]=false;
				}
			}
		}
	}
	private long getUnique(){ 
		return nUnique ; 
	}
	private long getTotal(){ return nTotal ; }
	//以下は以降のステップで使い回します
	private int symmetryOps() {
      int     k;
      int     nEquiv;
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ trial[k] = board[k];}
      //時計回りに90度回転
      rotate (trial, scratch, size, false);
      k = intncmp (board, trial, size);
      if (k > 0) return 0;
      if ( k == 0 )
         nEquiv = 1;
      else {
    	 //時計回りに180度回転
         rotate (trial, scratch, size, false);
         k = intncmp (board, trial, size);
         if (k > 0) return 0;
         if ( k == 0 )
            nEquiv = 2;
         else {
        	//時計回りに270度回転
            rotate (trial, scratch, size, false);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
            nEquiv = 4;
         }
      }
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ trial[k] = board[k];}
      //垂直反転
      vMirror (trial, size);
      k = intncmp (board, trial, size);
      if (k > 0) return 0;
      if (nEquiv > 1) {        // 4回転とは異なる場合
    	 // -90度回転 対角鏡と同等
         rotate (trial, scratch, size, true);
         k = intncmp (board, trial, size);
         if (k > 0) return 0;
         if (nEquiv > 2){     // 2回転とは異なる場合
        	// -180度回転 水平鏡像と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
            // -270度回転 反対角鏡と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) return 0;
         }
      }
      return nEquiv * 2;
   }
	private int intncmp (int[] lt, int[] rt, int n) {
      int k, rtn = 0;
      for (k = 0; k < n; k++) {
    	  rtn = lt[k]-rt[k];
    	  if ( rtn != 0 ){ break;}
      }
      return rtn;
	}
	private void rotate(int[] check, int[] scr, int n, boolean neg) {
      int j, k;
      int incr;
      k = neg ? 0 : n-1;
      incr = (neg ? +1 : -1);
      for (j = 0; j < n; k += incr ){ scr[j++] = check[k];}
      k = neg ? n-1 : 0;
      for (j = 0; j < n; k -= incr ){ check[scr[j++]] = k;}
	}
	private void vMirror(int[] check, int n) {
      int j;
      for (j = 0; j < n; j++){ check[j] = (n-1) - check[j];}
      return;
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen4();   //回転・反転・対称


/**
 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。
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
  15:          2279184       285053  00:00:03
  16:         14772512      1846955  00:00:24
  */

class NQueen5{
	private int[] board ;
	private long nUnique ;
	private long nTotal ;
	private int size;
	private int max ;
//	private boolean[] colChk;
    private boolean[] diagChk;
    private boolean[] antiChk;
	private int[] trial ;
	private int[] scratch ;
	public NQueen5(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.size=2; size<max; size++){
			board=new int[size];
			trial     = new int[size];
			scratch   = new int[size];
//			colChk    = new boolean[size];
			diagChk   = new boolean[2*size-1];
			antiChk   = new boolean[2*size-1];
			nTotal=0; 
			nUnique=0;
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			nQueens(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,getTotal(),getUnique(),TIME); 
		}
	}
	void nQueens(int row) {
		int k, lim, vTemp;
		if (row < size-1) {
		    if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
				nQueens(row + 1);
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
			}
			lim = (row != 0) ? size : (size + 1) / 2;
			for (k=row+1; k<lim; k++) {
				vTemp = board[k];
				board[k] = board[row];
				board[row] = vTemp;
				if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
		            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
					nQueens(row + 1);
		            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
				}
			}
			vTemp = board[row];
			for (k = row + 1; k < size; k++){
				board[k - 1] = board[k];
			}
			board[k - 1] = vTemp;
		} else { 
	        if ( (diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
				return;
			}
			k = symmetryOps();
			if (k != 0) {
				nUnique++;
				nTotal += k;
			}
		}
		return;
	}
	private long getUnique(){ return nUnique ; }
	private long getTotal(){ return nTotal ; }
	//
	//以下は以降のステップで使い回します
	private int symmetryOps() {
      int     k;
      int     nEquiv;
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){
    	  trial[k] = board[k];
      }
      //時計回りに90度回転
      rotate (trial, scratch, size, false);
      k = intncmp (board, trial, size);
      if (k > 0) { 
    	  return 0;
      }
      if ( k == 0 ){
         nEquiv = 1;
      } else {
    	 //時計回りに180度回転
         rotate (trial, scratch, size, false);
         k = intncmp (board, trial, size);
         if (k > 0) { 
        	 return 0;
         }
         if ( k == 0 ){
            nEquiv = 2;
         } else {
        	 /* 270 degrees */
        	//時計回りに270度回転
            rotate (trial, scratch, size, false);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            nEquiv = 4;
         }
      }
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ 
    	  trial[k] = board[k];
      }
      //垂直反転
      vMirror (trial, size);
      k = intncmp (board, trial, size);
      if (k > 0) {
    	  return 0;
      }
      if (nEquiv > 1) {        // 4回転とは異なる場合
    	 // -90度回転 対角鏡と同等
         rotate (trial, scratch, size, true);
         k = intncmp (board, trial, size);
         if (k > 0) {
        	 return 0;
         }
         if (nEquiv > 2){     // 2回転とは異なる場合
        	// -180度回転 水平鏡像と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            // -270度回転 反対角鏡と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
         }
      }
      return nEquiv * 2;
   }
	private int intncmp (int[] lt, int[] rt, int n) {
      int k, rtn = 0;
      for (k = 0; k < n; k++) {
    	  rtn = lt[k]-rt[k];
    	  if ( rtn != 0 ){ 
    		  break;
    	  }
      }
      return rtn;
	}
	private void rotate(int[] check, int[] scr, int n, boolean neg) {
      int j, k;
      int incr;
      k = neg ? 0 : n-1;
      incr = (neg ? +1 : -1);
      for (j = 0; j < n; k += incr ){ 
    	  scr[j++] = check[k];
      }
      k = neg ? n-1 : 0;
      for (j = 0; j < n; k -= incr ) { 
    	 check[scr[j++]] = k;
      }
	}
	private void vMirror(int[] check, int n) {
      int j;
      for (j = 0; j < n; j++) { 
    	  check[j] = (n-1) - check[j];
      }
      return;
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen5(); //最適化

/**
 * ６．マルチスレッド1
 * 
 * 　クイーンが上段角にある場合とそうではない場合の二つにスレッドを分割し並行処理
 * さらに高速化するならば、rowひとつずつにスレッドを割り当てる方法もある。
 * 　backTrack1とbackTrack2を以下で囲んでスレッド処理するとよい。
 * 　ただしスレッド数を管理する必要がある。
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
  16:         14772512      1846955  00:00:05
  */

class NQueen6_Board {
	private int nSoln = 0; // Total solutions for this board
	private int nUniq = 0; // Unique solutions, rejecting ones equivalent based on rotations.
	private int limit; // Board mid-point
	private int nextCol = 0; // Next position to be computed
	public NQueen6_Board(int size) {
		limit = (size + 1) / 2; // Mirror images done automatically
	}
	public synchronized int nextJob(long nS, long nU) {
		nSoln += nS;
		nUniq += nU;
		// If all columns have been assigned, return the exit flag
		return nextCol < limit ? nextCol++ : -1;
	}
	public int getTotal() { return nSoln; }
	public int getUnique() { return nUniq; }
}
class NQueen6_WorkEngine extends Thread {
	private int[] board; // Current state of the board
	private int[] trial; // Array for symmetry operations
	private int[] scratch; // Scratch space for rotations
	private int size; // Filled in constructor
	private long nUnique; // Default initialization is zero
	private long nTotal; // for both of these.
	private boolean[] diagChk; // Diagonals in use
	private boolean[] antiChk; // Antidiagonals in use
	private NQueen6_WorkEngine child; // Next thread
	private NQueen6_Board info; // Information broker
	public NQueen6_WorkEngine(int size, int nMore, NQueen6_Board info) {
		this.size = size;
		this.info = info;
		board = new int[size];
		trial = new int[size];
		scratch = new int[size];
		diagChk = new boolean[2 * size - 1];
		antiChk = new boolean[2 * size - 1];
		if (nMore > 0){
			try {
				child = new NQueen6_WorkEngine(size, nMore - 1, info);
				child.start();
			} catch (Exception e) {
				System.out.println(e);
			}
		} else {
			child = null;
		}
	}
	public void run() {
		int nextCol;
		while (true) { // Will break out on -1 for column posn.
			int row, col;
			// On the first call, nTotal and nUnique hold zeroes.
			nextCol = info.nextJob(nTotal, nUnique);
			if (nextCol < 0){
				break;
			}
			// Empty out counts from the last board processed
			nTotal = nUnique = 0;
			// Generate the initial permutation vector, given nextCol
			board[0] = nextCol;
			for (row = 1, col = 0; row < size; row++, col++){
				board[row] = col == nextCol ? ++col : col;
			}
			// Empty out the diagChk and antiChk vectors
			for (row = 0; row < 2 * size - 1; row++){
				diagChk[row] = antiChk[row] = false;
			}
			diagChk[size - 1 - nextCol] = antiChk[nextCol] = true;
			// Now compute from row 1 on down.
			nQueens(1);
		}
		if (child != null){
			try {
				child.join();
			} catch (Exception e) {
				System.out.println(e);
			}
		}
	}
	private void nQueens(int row) {
		int k, lim, vTemp;
		if (row < size - 1) {
		    if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
				nQueens(row + 1);
	            diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
			}
			lim = (row != 0) ? size : (size + 1) / 2;
			for (k = row + 1; k < lim; k++) {
				vTemp = board[k];
				board[k] = board[row];
				board[row] = vTemp;
				if ( !(diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
					diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = true;
					nQueens(row + 1);
					diagChk[row-board[row]+size-1] = antiChk[row+board[row]] = false;
				}
			}
			vTemp = board[row];
			for (k = row + 1; k < size; k++){
				board[k - 1] = board[k];
			}
			board[k - 1] = vTemp;
		} else { 
	        if ( (diagChk[row-board[row]+size-1] || antiChk[row+board[row]]) ){
				return;
	        }
			k = symmetryOps();
			if (k != 0) {
				nUnique++;
				nTotal += k;
			}
		}
		return;
	}

	//
	//以下は以降のステップで使い回します
	private int symmetryOps() {
      int     k;
      int     nEquiv;
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){
    	  trial[k] = board[k];
      }
      //時計回りに90度回転
      rotate (trial, scratch, size, false);
      k = intncmp (board, trial, size);
      if (k > 0) { 
    	  return 0;
      }
      if ( k == 0 ){
         nEquiv = 1;
      } else {
    	 //時計回りに180度回転
         rotate (trial, scratch, size, false);
         k = intncmp (board, trial, size);
         if (k > 0) { 
        	 return 0;
         }
         if ( k == 0 ){
            nEquiv = 2;
         } else {
        	 /* 270 degrees */
        	//時計回りに270度回転
            rotate (trial, scratch, size, false);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            nEquiv = 4;
         }
      }
      // 回転・反転・対称チェックのためにboard配列をコピー
      for (k = 0; k < size; k++){ 
    	  trial[k] = board[k];
      }
      //垂直反転
      vMirror (trial, size);
      k = intncmp (board, trial, size);
      if (k > 0) {
    	  return 0;
      }
      if (nEquiv > 1) {        // 4回転とは異なる場合
    	 // -90度回転 対角鏡と同等
         rotate (trial, scratch, size, true);
         k = intncmp (board, trial, size);
         if (k > 0) {
        	 return 0;
         }
         if (nEquiv > 2){     // 2回転とは異なる場合
        	// -180度回転 水平鏡像と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
            // -270度回転 反対角鏡と同等
            rotate (trial, scratch, size, true);
            k = intncmp (board, trial, size);
            if (k > 0) {
            	return 0;
            }
         }
      }
      return nEquiv * 2;
   }
	private int intncmp (int[] lt, int[] rt, int n) {
      int k, rtn = 0;
      for (k = 0; k < n; k++) {
    	  rtn = lt[k]-rt[k];
    	  if ( rtn != 0 ){ 
    		  break;
    	  }
      }
      return rtn;
	}
	private void rotate(int[] check, int[] scr, int n, boolean neg) {
      int j, k;
      int incr;
      k = neg ? 0 : n-1;
      incr = (neg ? +1 : -1);
      for (j = 0; j < n; k += incr ){ 
    	  scr[j++] = check[k];
      }
      k = neg ? n-1 : 0;
      for (j = 0; j < n; k -= incr ) { 
    	 check[scr[j++]] = k;
      }
	}
	private void vMirror(int[] check, int n) {
      int j;
      for (j = 0; j < n; j++) { 
    	  check[j] = (n-1) - check[j];
      }
      return;
	}
}
class NQueen6{
	private int[] board ;
	private int size;
	private int max ;
	private int nThreads;
	private NQueen6_Board info ;
	private NQueen6_WorkEngine child;
	public NQueen6(){
		max=27 ;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.size=2; size<max; size++){
			board=new int[size];
			info = new NQueen6_Board(size);
			child = new NQueen6_WorkEngine(size, nThreads - 1, info);
			nThreads=size+8;
			for(int k=0; k<size; k++){ board[k]=k ; }
			long start = System.currentTimeMillis() ;
			try {
				child.start() ;
				child.join();
			}catch(Exception e){ System.out.println(e); }
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",size,info.getTotal(),info.getUnique(),TIME); 
		}
	}
}
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen6();   // マルチスレッド

    
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

class NQueen7 {
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
	public NQueen7(){
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
    // $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
    // new NQueen7() ;    // シンプルな対称解除法＋ビットマップ


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
class NQueen8{
	public NQueen8(){
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
    // $ javac Algorithm.java && java -Xms4g -Xmx8g Algorithm
    new NQueen8() ;    // シンプルな対称解除法＋ビットマップ＋マルチスレッド

  }
}

