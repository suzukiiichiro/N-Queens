import org.apache.commons.lang3.time.DurationFormatUtils;
/**
 * $ javac Algorighm.java ;
 * $ java -Xms4g -Xmx4g Algorithm ;
 */
public class Algorithm {
	public static void main(String[] args){
		/**
		 * 目次
		 * 1. ソート
		 * 　バブルソート
		 * 　選択ソート
		 * 　挿入ソート
		 * 　マージソート
		 * 　シェルソート
		 * 　クイックソート
		 * 2. 再帰
		 * 　三角数
		 * 　階乗
		 * 　ユークリッドの互除法
		 * 　ハノイの塔
		 * 3.　Nクイーン問題
		 * 　 ブルートフォース
		 * 　 バックトラック
		 * 　 配置フラグによる制約テスト高速化
		 * 　 ビットマップによるビット処理高速化
		 * 　 回転と斜軸
		 */

		
		/**
		 * ソート
		 * 
		 * 
 		 * バブルソート 13404mm
 		 * https://ja.wikipedia.org/wiki/バブルソート
 		 * https://www.youtube.com/watch?v=8Kp-8OGwphY
 		 * 　平均計算時間が O(ｎ^2)
		 * 　安定ソート
		 * 　比較回数は「　n(n-1)/2　」
		 * 　交換回数は「　n^2/2　」
		 * 　派生系としてシェーカーソートやコムソート
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort01_BubbleSort();
		/**
		 * 選択ソート 3294mm
 		 * https://ja.wikipedia.org/wiki/選択ソート
		 * https://www.youtube.com/watch?v=f8hXR_Hvybo
		 * 　平均計算時間が O(ｎ^2)
		 * 　安定ソートではない
		 * 　比較回数は「　n(n-1)/2　」
		 * 　交換回数は「　n-1　」
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort02_SelectionSort();
		/**
		 * 挿入ソート 3511mm
 		 * https://ja.wikipedia.org/wiki/挿入ソート
		 * https://www.youtube.com/watch?v=DFG-XuyPYUQ
		 * 　平均計算時間が O(ｎ^2)
		 * 　安定ソート
		 * 　比較回数は「　n(n-1)/2以下　」
		 * 　交換回数は「　約n^2/2以下　」
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort03_InsertionSort();
		/**
		 * マージソート 1085mm
 		 * https://ja.wikipedia.org/wiki/マージソート
		 * https://www.youtube.com/watch?v=EeQ8pwjQxTM
		 *   平均計算時間が O(n Log n)
		 *   安定ソート
		 *   コメント
		 *   50以下挿入ソート、5万以下マージソート、あとはクイックソート 
		 *   # バブルソート、挿入ソート、選択ソートがO(N^2)の時間を要するのに対し、
		 *  マージソートはO(N*logN)です。
		 *  例えば、N(ソートする項目の数）が10,000ですと、N^2は100,000,000ですが、
		 *  n*logNは40,000です。別の言い方をすると、マージソートで４０秒を要するソートは、
		 *  挿入ソートでは約２８時間かかります。
		 *  マージソートの欠点は、ソートする配列と同サイズの配列をもう一つ必要とする事です。
		 *  元の配列がかろうじてメモリに治まるという大きさだったら、マージソートは使えません。
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort04_MergeSort();
		/**
		 * シェルソート 1052mm
 		 * https://ja.wikipedia.org/wiki/シェルソート
		 * https://www.youtube.com/watch?v=M9YCh-ZeC7Y
		 * 　平均計算時間が O(ｎ^1.25)
		 * 　安定ソートではない
		 * 　挿入ソート改造版
		 *　 ３倍して１を足すという処理を要素を超えるまで行う
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort05_ShellSort();
		/**
		 * クイックソート 1131mm
 		 * https://ja.wikipedia.org/wiki/クイックソート
		 * https://www.youtube.com/watch?v=aQiWF4E8flQ
		 * 　平均計算時間が O(n Log n)
		 * 　安定ソートではない
		 * 　最大計算時間が O(n^2)
		 *
		 * データ数が 50 以下なら挿入ソート (Insertion Sort)
		 * データ数が 5 万以下ならマージソート (Merge Sort)
		 * データ数がそれより多いならクイックソート (Quick Sort)
		 */
		// ↓ のコメントアウトを外して実行
//		new Sort06_QuickSort();
		
		
		/**
		 * 再帰
		 * 
		 * 三角数
		 * https://ja.wikipedia.org/wiki/三角数
		 * 5 + 4 + 3 + 2 + 1 = 15
		 */
		// ↓ のコメントアウトを外して実行
//		System.out.println(new Recursive01_Triangle().triangle(5));
		/**
		 * https://ja.wikipedia.org/wiki/階乗
		 * 5 * 4 * 3 * 2 * 1 = 120 
		 */
		// ↓ のコメントアウトを外して実行
//		System.out.println(new Recursive02_Factorial().factorial(5));
		/**
		 * https://ja.wikipedia.org/wiki/ユークリッドの互除法
		 * (問題) 1071 と 1029 の最大公約数を求める。
		 * 1071 を 1029 で割った余りは 42
		 * 1029 を 42 で割った余りは 21
		 * 42 を 21 で割った余りは 0
		 * よって、最大公約数は21である。
		 */
		// ↓ のコメントアウトを外して実行
//		System.out.println(new Recursive03_Euclid().euclid(1071, 1029));
		/**
		 * https://ja.wikipedia.org/wiki/ハノイの塔
		 *  古代インドの神話では、遠い秘境のお寺で僧侶達が毎日毎晩、
		 *  ６４枚の黄金の円盤をダイヤモンドをちりばめた３つの塔の間で移し替え作業をしている。
		 *  その移し替え作業が完了したら世界の終末が訪れるのだそうだ。
		 * n3: 7
		 * n4: 15 
		 * n5: 31
		 * n6: 63
		 * n7: 127
		 * n8: 255
		 * n9: 511
		 * n10:1,023
		 * n20:1,048,575
		 * n30:1,073,741,823
		 * n40:1,099,511,627,775
		 * n50:1,125,899,906,842,620
		 * n60:1,152,921,504,606,850,000
		 * n70:1,180,591,620,717,410,000,000
		 * n80:1,208,925,819,614,630,000,000,000
		 * n90:1,237,940,039,285,380,000,000,000,000
		 * n100:1,267,650,600,228,230,000,000,000,000,000
		 */
		// ↓ のコメントアウトを外して実行
//		new Recursive04_Hanoi().hanoi(20, 'A', 'B', 'C');

		/**
		 * 再帰　Nクイーン問題
		 * https://ja.wikipedia.org/wiki/エイト・クイーン
		 * 
		 * 当方の開発実行環境
		 * MacOSX Macbook Pro
		 * プロセッサ：2.5Ghz intel core i7
		 * メモリ：16G 
		 * 
		 * コンパイルおよび実行方法
		 * $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
		 *
		 * 
		 * 　1. ブルートフォース	(NQueen1())
		 * 　2. バックトラック 	(NQueen2())
		 * 　3. 配置フラグ		(NQueen3())
		 * 　4. ビットマップ		(NQueen4())
		 * 　5. 回転と斜軸		(NQueen5())
     *   6. ユニーク解から全解への展開(NQueen6())
		 */
		/**
		 * ブルートフォース  
		 * 
		 * しらみつぶし探索：nx(n-1)x(n-2)x.....x2x1=n!
		 * 全てのパターンを生成 各行にひとつのクイーンを配置する
		 * 全ての可能性のある下位の候補を体系的に数え上げ、それぞれの解候補が問題の解と
		 * なるかをチェックする。
		 * パターンの生成後にチェックするので「生成検査法(generate & test)」とも言われる。
		 * 16,777,216通りある。
		 * 
		 * (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだけで
		 * あって8王妃問題を解いているわけではありません
		 */
		// ↓ のコメントアウトを外して実行
//		new NQueen1();

		/**
		 * バックトラック 
		 *  
		 * 途中で制約を満たさない事が明らかな場合は以降のパターン生成を行わない
		 * 各行に加え各列にひとつのクイーンを配置する
		 * パターンを生成し終わってからチェックを行うのではなく、途中で制約を満た
		 * さないことが明らかな場合はそれ以降のパターン生成を行わない。
		 * 「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
		 * 事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
		 * 40,320通りある。
		 * 
		 * (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組み
		 * 合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
		 */
		// ↓ のコメントアウトを外して実行
//		new NQueen2();

		/**
		 * 配置フラグによる制約テスト高速化
		 * 
		 * 各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
		 * これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、
		 * 王妃は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつ
		 * だけしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず
		 * 木を降りても解がないと判明した時点で木を引き返すということができます。
		 */
		// $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
		// ↓ のコメントアウトを外して実行
		// new NQueen3();

		/**
		 * 実行結果
		 N:        Total       Unique        hh:mm
		 0:                0            0     0:00:00
		 1:                1            0     0:00:00
		 2:                0            0     0:00:00
		 3:                0            0     0:00:00
		 4:                2            0     0:00:00
		 5:               10            0     0:00:00
		 6:                4            0     0:00:00
		 7:               40            0     0:00:00
		 8:               92            0     0:00:00
		 9:              352            0     0:00:00
		10:              724            0     0:00:00
		11:             2680            0     0:00:00
		12:            14200            0     0:00:00
		13:            73712            0     0:00:00
		14:           365596            0     0:00:02
		15:          2279184            0     0:00:13
		16:         14772512            0     0:01:34
		17:         95815104            0     0:11:18
		*/

		/**
		 * ビット演算を使って高速化 状態をビットマップにパックし、処理する
		 * 
		 * ビットマップであれば、シフトにより高速にデータを移動できる
		 * フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
		 * フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充分。
		 * 配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
		 * バックトラックよりも２０−３０倍高速。
		 * 
		 *考え方 1
		 * Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひとつのビッ
		 * トフィールドに対応させます。(クイーンが置いてある位置のビットをONにする)
		 * そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にいずれかの
		 * ビット位置をひとつだけONにして進めていきます。
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
		 * 考え方 2
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
		 * 考え方 3
		 * 　n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
		 * ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
		 * できない位置ということになります。次にその３つのビットフィールドをORしたビッ
		 * トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
		 * ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
		 * 次の演算を行なってみます。
		 * 
		 * bit = -bitmap & bitmap; //一番右のビットを取り出す
		 * 
		 * 　この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
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
		 * 　正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
		 * 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
		 * 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
		 * によって1ビット抽出を実現させていることが重要です。
		 * 
		 *      00010110   22
		 *  AND 11101010  -22
		 * ------------------
		 *      00000010
		 * 
		 * 　さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
		 * 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
		 * 生成されることになります。
		 * 
		 * while (bitmap) {
		 *     bit = -bitmap & bitmap;
		 *     bitmap ^= bit;
		 *     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
		 * }
		 */
		// $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
		// ↓ のコメントアウトを外して実行
		// new NQueen4();
	
		/**
		 * 実行結果 
		 N:        Total       Unique        hh:mm:ss
		 0:                1            0     0:00:00
		 1:                1            0     0:00:00
		 2:                0            0     0:00:00
		 3:                0            0     0:00:00
		 4:                2            0     0:00:00
		 5:               10            0     0:00:00
		 6:                4            0     0:00:00
		 7:               40            0     0:00:00
		 8:               92            0     0:00:00
		 9:              352            0     0:00:00
		10:              724            0     0:00:00
		11:             2680            0     0:00:00
		12:            14200            0     0:00:00
		13:            73712            0     0:00:00
		14:           365596            0     0:00:00
		15:          2279184            0     0:00:01
		16:         14772512            0     0:00:08
		17:         95815104            0     0:00:56
		18:        666090624            0     0:06:58
		*/

		/**
		 * ビット演算に加えてユニーク解(回転・反転）を使って高速化 
		 * 
		 * ■ユニーク解の判定方法
		 * 
		 * 　全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
		 * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
		 * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
		 * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
		 * ことを意味しています。
		 * 
		 * 　従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
		 * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
		 * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
		 * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
		 * わりません。
		 * 
		 * 　さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
		 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
		 * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
		 * 下の例では要素数が２個のものと８個のものがあります。
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
		 * 　それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
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
		 * 　探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
		 * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
		 * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
		 * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
		 * 変換を試みるだけでユニーク解の判定が可能になります。
		 * 
		 * 
		 * ■ユニーク解の個数を求める
		 * 　先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
		 * ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
		 * 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
		 * うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
		 * れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
		 * 央になければならず、それは互いの効き筋にあたるので有り得ません。
		 *
		 * ***********************************************************************
		 * 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
		 * ***********************************************************************
		 * 
		 * 　次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
		 * の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
		 * を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
		 * ます。突き詰めれば、
		 * 
		 * [上から２行目のクイーンの位置が右から何番目にあるか]
		 * [右から２列目のクイーンの位置が上から何番目にあるか]
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
		 * 　結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
		 * られる解は総てユニーク解であることが保証されます。
		 * 
		 * - - - - X Q
		 * - Q - - X -
		 * - - - - X -
		 * - - - - X -
		 * - - - - - -
		 * - - - - - -
		 * 
		 * 　次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
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
		 * 　次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
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
		 */
		// $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
		// ↓ のコメントアウトを外して実行
		// new NQueen5();
		/**
		 * ユニーク解だけが出力されます
		 * 
		 N:        Total       Unique        hh:mm:ss
		 2:                0            0     0:00:00
		 3:                0            0     0:00:00
		 4:                0            1     0:00:00
		 5:                0            2     0:00:00
		 6:                0            1     0:00:00
		 7:                0            6     0:00:00
		 8:                0           12     0:00:00
		 9:                0           46     0:00:00
		10:                0           92     0:00:00
		11:                0          341     0:00:00
		12:                0         1787     0:00:00
		13:                0         9233     0:00:00
		14:                0        45752     0:00:00
		15:                0       285053     0:00:00
		16:                0      1846955     0:00:02
		17:                0     11977939     0:00:14
		18:                0     83263591     0:01:48
		19:                0    621012754     0:17:05
		*/

		 /**
		 * ■ユニーク解から全解への展開
		 * 　これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
		 * ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
		 * 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
		 * 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
		 * を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
		 * れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
		 * いう考察が必要になってきます。
		 * 
		 * 　最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
		 * オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
		 * 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
		 * ず８個(＝２×４)になります。
		 * 
		 * 　次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
		 * ために次の事柄を各自で確認して下さい。
		 * TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
		 * 　(1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルから180度回転)
		 * 　　　させても、さらに90度回転(オリジナルから270度回転)させてもオリジナルと同型になる。  
		 * 		COUNT2 * 2
		 * 　(2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナルとは異なる。
		 * 　　　ただし、180度回転させた場合はオリジナルと同型になることも有り得る。 
		 * 		COUNT4 * 4
		 * 　(1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
		 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
		 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
		 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
		 *		COUNT8 * 8 
		 * 　以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
		 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
		 * りを外す必要がなくなったというわけです。 
		 * 
		 *   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
		 * 	 TOTAL	(COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
		 */
		// $ javac Algorithm.java && java -Xms4g -Xmx4g Algorithm
		// ↓ のコメントアウトを外して実行
		new NQueen6();
		
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
        *
		*/
	}
}
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
class Recursive01_Triangle{
	int triangle(int n){
		if(n==1)
			return 1;
		else
			return n+triangle(n-1);
	}
}
class Recursive02_Factorial{
	int factorial(int n){
		if(n==1)
			return 1;
		else 
			return n*factorial(n-1);
	}
}
class Recursive03_Euclid{
	int euclid(int x, int y){
		if(y==0)
			return x ;
		else
			return euclid(y, x%y);
	}
}
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
class NQueen1{
	int[] pos=new int[8];
	int count=1;
	public NQueen1(){
		set(0);
	}
	void print(){
		for(int i=0;i<8;i++)
			System.out.printf("%2d", pos[i]);
		System.out.println("  : " +count++);
	}
	void set(int n){
		for(int i=0;i<8;i++){ // 各列にひとつのクイーンを配置する
			pos[n]=i;
			if(n==7) //最後の列で出力
				print();
			else
				set(n+1); //次の行へ
		}
	}
}
class NQueen2{
	int[] pos = new int[8];
	boolean[] f = new boolean[8];
	int count=1;
	public NQueen2(){
		set(0);
	}
	void print(){
		for(int i=0; i<8; i++)
			System.out.printf("%2d",  pos[i]);
		System.out.println("  :  "+count++);
	}
	void set(int n){
		for(int i=0; i<8; i++){ // 各列にひとつのクイーンを配置する
			if(f[i]==false){ //バックトラック 各行に一つのクイーンを配置する
				pos[n]=i;
				if(n==7)
					print();
				else{
					f[i]=true ;
					set(n+1);
					f[i]=false;//リセット
				}
			}
		}
	}
}
class NQueen3{
	int[] pos=null ;
	boolean[] fa=null ;
	boolean[] fb=null ;
	boolean[] fc=null ;
	int N=0;
	int max=27 ;

	long TOTAL=0;

	void print(){
		TOTAL++;
	}
	public NQueen3(){
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(this.N=0; N<this.max; N++){
			pos=new int[N];
			fa=new boolean[N];
			fb=new boolean[N*2];
			fc=new boolean[N*2];
			TOTAL=0;
			long start = System.currentTimeMillis() ;
			set(0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",N,TOTAL,0,TIME); 
		}
	}
	void set(int n){
		for(int i=0; i<N; i++){
			// バックトラック+配置フラグ
			// fa 各行 fb 斜め fc 斜め
			if(fa[i]==false && fb[n+i]==false && fc[n-i+(N-1)]==false){
				pos[n]=i;
				if(n==(N-1))
					TOTAL++;
				else{
					fa[i]=fb[n+i]=fc[n-i+(N-1)]=true;
					set(n+1);
					fa[i]=fb[n+i]=fc[n-i+(N-1)]=false ;
				}	
			}
		}
	}
}

class NQueen4{
	int MAX=27 ;
	int SIZE=0;
	int MASK=0;

	long COUNT=0;

	public NQueen4(){
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(SIZE=0; SIZE<MAX; SIZE++){
			long start = System.currentTimeMillis() ;
			COUNT=0;
			MASK=(1<<SIZE)-1;
			backTrackBit(0,0,0,0);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",SIZE,COUNT,0,TIME); 
		}
	}
	void backTrackBit(int y, int left, int down, int right){
		int bitmap, bit ;
		if(y==SIZE)
			COUNT++;
		else{
			bitmap=MASK & ~(left|down|right);
			while(bitmap>0){
				bit=-bitmap&bitmap;
				bitmap^=bit;
				backTrackBit( y+1, (left|bit)<<1, (down|bit), (right|bit)>>1);
			}
		}	
	}
}
class NQueen5 {
	int bit;
	int MASK;
	int SIZEE;
	int[] BOARD;
	int TOPBIT;
	int SIDEMASK;
	int LASTMASK;
	int ENDBIT;
	int BOUND1;
	int BOUND2;
	int SIZE=0;
	int MAX=27;

	long TOTAL ;
	long UNIQUE;

	void Check(int bsize) {
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
				UNIQUE++;
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
				UNIQUE++;
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
		UNIQUE++;
	}
	/**
	 * 最上段のクイーンが角以外にある場合の探索
	 */
	void backTrack2(int y, int left, int down, int right){
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
	void backTrack1(int y, int left, int down, int right){
		int bitmap=( MASK & ~(left|down|right) );
		if(y==SIZEE){
			if(bitmap!=0){
				BOARD[y]=bitmap;
				UNIQUE++;
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
	public NQueen5(){
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(int SIZE=2; SIZE<MAX+1; SIZE++){
			long start = System.currentTimeMillis() ;
			bitmap_rotate(SIZE);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",SIZE,0,UNIQUE,TIME); 
		}
	}
	void bitmap_rotate(int SIZE) {
		this.SIZE=SIZE;
		SIZEE = SIZE-1;
		TOPBIT = 1<<SIZEE;
		MASK=(1<<SIZE)-1;
		UNIQUE=0;
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
	}
}
class NQueen6 {
	int bit;
	int MASK;
	int SIZEE;
	int[] BOARD;
	int TOPBIT;
	int SIDEMASK;
	int LASTMASK;
	int ENDBIT;
	int BOUND1;
	int BOUND2;

	long UNIQUE ;
	long TOTAL ;
	long COUNT8;
	long COUNT4;
	long COUNT2;

	void Check(int bsize) {
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
	void backTrack2(int y, int left, int down, int right){
		int bitmap= ( MASK & ~(left|down|right)) ;
		if(y==SIZEE){
			if(bitmap!=0){
				// ==  or !=
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
	void backTrack1(int y, int left, int down, int right){
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
	public NQueen6(){
	int MAX=27;
		System.out.println(" N:            Total       Unique    hh:mm:ss");
		for(int SIZE=2; SIZE<MAX+1; SIZE++){
			long start = System.currentTimeMillis() ;
			bitmap_rotate(SIZE);
			long end = System.currentTimeMillis();
			String TIME = DurationFormatUtils.formatPeriod(start, end, "HH:mm:ss");
			System.out.printf("%2d:%17d%13d%10s%n",SIZE,TOTAL,UNIQUE,TIME); 
		}
	}
	void bitmap_rotate(int SIZE) {
		//this.SIZE=SIZE;
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
}
