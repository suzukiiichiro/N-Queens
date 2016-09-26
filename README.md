# N-Queen

Javaエンジニアのための「アルゴリズムとデータ構造」
ソートや再帰といったアルゴリズムの基本を習熟し、
N-Queen問題をステップバイステップで学ぶ

開発環境：Java

Javaの基本知識があれば習熟可能です。

このアルゴリズムは他のプログラミング言語にも広く展開応用できます。
アルゴリズムの核心となるソートプログラムを通して基本を習熟します。
その後、中級プログラマの壁となる「再帰」を学び、応用学習として、
再帰を使った「Ｎ−Ｑｕｅｅｎ問題」を解決し、さらに高速化手法を追求
します。

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
		 2:                0            0    00:00:00
		 3:                0            0    00:00:00
		 4:                2            1    00:00:00
		 5:               10            2    00:00:00
		 6:                4            1    00:00:00
		 7:               40            6    00:00:00
		 8:               92           12    00:00:00
		 9:              352           46    00:00:00
		10:              724           92    00:00:00
		11:             2680          341    00:00:00
		12:            14200         1787    00:00:00
		13:            73712         9233    00:00:00
		14:           365596        45752    00:00:00
		15:          2279184       285053    00:00:00
		16:         14772512      1846955    00:00:02
		17:         95815104     11977939    00:00:15
		18:        666090624     83263591    00:01:49
		19:       4968057848    621012754    00:13:55
		20:      39029188884   4878666808    01:50:42
		21:     314666222712  39333324973    15:34:05
		*/

