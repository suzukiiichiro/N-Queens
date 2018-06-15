/**

 Javaで学ぶアルゴリズムとデータ構造  
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木 維一郎(suzuki.iichiro@kyodonews.jp)
 

 Java/C/Lua/Bash版　が用意してあります。お好みでどうぞ
 https://github.com/suzukiiichiro/N-Queen 
 			


*************************
はじめに
*************************

幸運にもこのページを参照することができたN-Queen(Nクイーン）エンジニアは少数だろ
う。Google検索またはGit検索でたどり着いたのだとは思うが、確率は奇跡に近い。エン
ジニアにしてこのページを参照できた奇跡ついでにもう少しだけ読み進めて欲しい。具体
的には以下のリンクにわかりやすく書いてある。

  エイト・クイーン問題
  https://ja.wikipedia.org/wiki/エイト・クイーン
 
エイト・クイーンは、1848年から存在し、ガウスなど著名な科学者が研究した工学研究の
頂点となる研究である。名前の通り８つのクイーンの解を求めるというパズルであり、N
クイーンは、エイトクイーンの拡張版で、Nの値は８、９、１０，１１，１２･･･と言った
風に増え続け、そのNの値であるボードの解を求めるものである。


*************************
歴史的未解決問題に懸賞金
*************************

歴史あるチェスのパズル問題が現代数学における未解決問題の解明につながる可能性
http://gigazine.net/news/20170905-million-dollar-chess-problem/

1000年を超える歴史を持つボードゲーム「チェス」には単なるゲームの勝敗ではなく、そ
のルールに即したさまざまなパズルの課題「チェス・プロブレム」が存在しています。エ
イト・クイーンはチェスの駒のうち、8個のクイーンだけを使うパズルなのですが、その
規模を大きく拡大して行くと、現代数学における未解決問題であり、1億円の賞金がかか
る「P対NP問題」の解明につながるものと考えられています。

2017 | “Simple” chess puzzle holds key to $1m prize | University of St Andrews
https://www.st-andrews.ac.uk/news/archive/2017/title,1539813,en.php

Can You Solve the Million-Dollar, Unsolvable Chess Problem? - Atlas Obscura
http://www.atlasobscura.com/articles/queens-puzzle-chess-problem-solution-software

「エイト・クイーン」は1848年にチェスプレイヤーのマックス・ベッツェルによって提案
されたパズル。8×8マスのチェス盤の上に、縦横と斜め方向にどこまででも進めるという
駒・クイーンを8個並べるというものなのですが、その際には「どの駒も他の駒に取られ
るような位置においてはいけない」というルールが設定されています。このルールに従っ
た場合にいくつの正解が存在するのか、長らくの間にわたって謎とされていたのですが、
考案から100年以上が経過した1874年にGuntherが行列式を用いて解く方法を提案し、イギ
リスのグレイシャー(Glaisher)によって全解(基本解)が12個であることを確認していま
す。

この問題は、チェス盤の一辺のマスの数とクイーンの数を同一にしたn-クイーン問題と
も呼ばれており、nの数が増えるに連れて飛躍的にその解数が増大することが知られてい
ます。記事作成時点で全ての解が判明しているのは、2009年にドレスデン工科大学で計
算された「26-クイーン」で、その基本解は2789兆7124億6651万289個、転回形などのバ
リエーション解を含めると、その数は2京2317兆6996億1636万4044個にもなることがわかっ
ています。

セント・アンドルーズ大学のコンピューターサイエンティストであるIan Gent博士らに
よる研究チームは、この「n-クイーン問題」から派生する「n-クイーン穴埋め問題」
(n-Queens Completion)パズルの複雑性に関する(PDF
http://jair.org/media/5512/live-5512-10126-jair.pdf)論文を作成しています。n-ク
イーン穴埋め問題は、チェス盤の上にあらかじめいくつかのクイーンの駒を並べておい
た状態で、残りのクイーンを全て埋めるというパズル問題です。

基本的にこの問題を解決するためにはバックトラック法と呼ばれる、いわば「総当たり
法」が用いられますが、全ての選択肢を試すためには膨大な時間が必要とされ、しかも
マスとクイーンの数が多くなるとその時間は指数関数的に一気に増加します。Gent氏に
よると、この「n-クイーン穴埋め問題」を素早く解決できるコンピューターやアルゴリ
ズムの開発が進むことで、我々が日々抱えている問題を解決する技術の進化が期待でき
るとのこと。先述のように、現代の科学でも解決できているn-クイーン問題は26×26マス
の「26-クイーン」にとどまっており、穴埋め問題であってもそこから先へと進むために
は、現在はまだ存在していない新しい技術を開発することが必須となってきます。

この問題は、2000年にアメリカのクレイ数学研究所が100万ドル(約1億1000万円)の賞金
とともに設定したミレニアム懸賞問題の一つに数えられる「P対NP問題」の証明につなが
るものとされています。これは、「答えを見つけるのは難しいかもしれないが、答えが
あっているかどうかは素早くチェックできる問題」のことをNP問題、「簡単に素早く解
ける問題」のことをP問題とした時に、「素早く解けるP問題はすべて答えを素早く確認
できるNP問題である」ことは証明されているが、その逆、つまり「答えを素早く確認で
きるNP問題はすべて、素早く解けるか？」という問題を証明するというもの。 これを解
くためには膨大な量の計算を素早く行うことが必要になり、現代のコンピューター技術
でも解決までには数万年の時間が必要になると考えられています。


*************************
参考リンクなど
*************************


GooleなどWebを探索すると無数のページがあることがわかる。その中でも充実したサイト
を紹介したい。おおよそ以下のサイトをかみしめて読み解けば情報は９０％網羅されてい
る。

N-Queens 問題(Nobuhide Tsudaさん)
*************************
  はじめに
  力まかせ探索（Brute-force search）
  バックトラッキング
  制約テスト高速化（配置フラグ）
  ビット演算（ビットマップ）による高速化
  対称解除去
  枝刈りによる高速化
  http://vivi.dyndns.org/tech/puzzle/NQueen.html

Puzzle DE Programming(M.Hiroiさん）
*************************
  バックトラックとビット演算による高速化
  http://www.geocities.jp/m_hiroi/puzzle/nqueens.html

takakenさん（高橋謙一郎さん）のページ
*************************
  Ｎクイーン問題（解の個数を求める）
    ビット処理を用いた基本形
    ビット処理を用いたプログラムの仕組み
    ユニーク解の判定方法
    ユニーク解の個数を求める
    ユニーク解から全解への展開
    ソースプログラムと実行結果
  http://www.ic-net.or.jp/home/takaken/nt/queen/index.html

の、みなさんが掲示板で議論している模様(貴重ですね）
http://www2.ic-net.or.jp/~takaken/auto/guest/bbs62.html

ptimal Queens
*************************
英語だが、上記の全てがJavaで書かれていて群を抜いている
http://penguin.ewu.edu/~trolfe/Queens/OptQueen.html

その他のリンク
https://rosettacode.org/wiki/N-queens_problem
http://www.cc.kyoto-su.ac.jp/~yamada/ap/backtrack.html
http://yucchi.jp/java/java_tip/n_queens_problem/n_queens_problem.html
http://www.shido.info/py/queen_py3.html
http://toraneko75.sakura.ne.jp/wp/?p=223
http://yoshiiz.blog129.fc2.com/blog-entry-380.html
http://nw.tsuda.ac.jp/class/algoB/c6.html
http://www.kawa.net/works/js/8queens/nqueens.html
http://www.yasugi.ai.kyutech.ac.jp/2012/4/nq.html
http://www.neuro.sfc.keio.ac.jp/~masato/jv/nqueen/MPneuron.java
http://fujimura2.fiw-web.net/java/lang/page-20-3.html
https://github.com/pankajmore/DPP/blob/master/EPI/src/puzzles/NQueens.java
http://www.kanadas.com/ccm/queens-sort/index-j.html
http://chiiji.s10.xrea.com/nn/nqueen/nqueenn.shtml
http://www.neuro.sfc.keio.ac.jp/~masato/jv/nqueen/nqueenDemo.htm


ここからは参考情報のメモとして

N=22発見 JeffSomers
  ビットマップを N-Queens に最初に応用したのは Jeff Somers 氏のようだ。 
  参照：The N Queens Problem
  http://www.jsomers.com/nqueen_demo/nqueens.html(リンク切れのようだ）

N=24発見 電気通信大学
  2004年、電気通信大学の研究グループが、処理を並列化し
  N=24 の解の個数を世界で初めて発見。 
  http://www.arch.cs.titech.ac.jp/~kise/nq/

  プレスリリース
  http://www.arch.cs.titech.ac.jp/~kise/nq/press-2004-10-05.txt

  電通大が「N-queens」問題の世界記録達成
  http://www.itmedia.co.jp/news/articles/0410/06/news079.html

  University of North Texas
  http://larc.unt.edu/ian/24queens/

  NQueens問題
  ＱＪＨの基本構想は、”部分解から全体解を構成するというアプローチ”（部分解合成
  法：Ｐａｒts Assembly Approach)です。
  http://deepgreen.game.coocan.jp/NQueens/nqueen_index.htm

  N Queens World records
  http://www.nqueens.de/sub/WorldRecord.en.html

  N=21-23 computed by Sylvain PION (Sylvain.Pion(AT)sophia.inria.fr) and Joel-Yann FOURRE (Joel-Yann.Fourre(AT)ens.fr).

  N=24 from Kenji KISE (kis(AT)is.uec.ac.jp), Sep 01 2004

  N=25 from Objectweb ProActive INRIA Team (proactive(AT)objectweb.org), Jun 11 2005 [Communicated by Alexandre Di Costanzo (Alexandre.Di_Costanzo(AT)sophia.inria.fr)]. This calculation took about 53 years of CPU time.N=25 has been confirmed by the NTU 25Queen Project at National Taiwan University and Ming Chuan University, led by Yuh-Pyng (Arping) Shieh, Jul 26 2005. This computation took 26613 days CPU time.

  N=26 as calculated by Queens(AT)TUD [http://queens.inf.tu-dresden.de/]. - Thomas B. Preußer, Jul 11 2009

  N=27 as calculated by the Q27 Project [https://github.com/preusser/q27]. - Thomas B. Preußer, Sep 23 2016


*****************************
このぺーじにはなにがあるのか
*****************************

具体的にこのページにはNクイーンのプログラムがある。

		  
		このテキストの使い方
		おおよそ上から読んでいきます。
		プログラムソースやアルゴリズムの説明はREADMEを参照して下さい。
		
		各章に new NQueen1() ;という行があります。
		  
		  (例）
		// new NQueen1(); //実行はコメントを外して実行
		 * 
		 上記のコメントを外してコンパイル＆実行して下さい。

		 ソースのコメントを外す
		// new NQueen1(); //実行はコメントを外して実行
		    ↓
		new NQueen1(); //実行はコメントを外して実行

		 
		 * コンパイルは以下の感じで行けます。 
		 javac -cp .:commons-lang3-3.4.jar: NQueen.java ;


		 　実行は以下の感じで行けます。(例 NQueen1 の場合）
		 java -cp .:commons-lang3-3.4.jar: NQueen1 ; 
		
		 実行にこだわりたい人は以下のパラメータで高速実行できます。
		 java -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen1 ; 
		 



Nクイーンの解決には処理を分解して一つ一つ丁寧に理解すべくステップが必要だ。
最初はステップ１のソースを何度も見て書いて理解するしかない。
もちろん、簡単なだけに解決時間も相当かかる。処理が終わるまでにコーヒーが飲み終
わってしまうかもしれない。ステップ15までくると、およそ１秒もかからずに処理が終了
する。１分かかっていたことが１秒で終わることに興味がわかないかもしれない。がしか
し、１００年かかることが１年かからないとしたらどうだろう。人工知能AI技術は、デバ
イスの進化、処理の高速化、解法の最適化（アルゴリズム）の三位一体だ。順番に、とば
すことなくじっくりと読み進めて欲しい。たぶん、日本中のNクイーンプログラムをここ
まで分解してステップにまとめているサイトはそう多くはないはずだ。

さらに、このサイトはNクイーンプログラムを複数のプログラム言語で習熟出来る準備がある。
例えば以下の通りだ。

  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen

  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash

  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua

  C版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_C
 
C版
　およそ全てのプログラム言語の中で最も高速に処理できると言われている。事実そう
だ。まだ何もわからない初学の人はC言語から始めるべきだ。
　マルチスレッドなど、Javaに比べて複雑に記述する必要がある分、プログラムの端々ま
での深い知識が必要だ。C言語マスターは間違いなく、Javaプログラマよりシステム技術
を網羅的に深く理解している。

Java版
　C言語があまりにも難解と言われ、取っつきやすい部分を残し、Cでできることを取りこ
ぼさずにできた言語がJavaだ。マルチスレッドも、C言語よりもわかりやすい。システム
技術の表層的な知識だけしかないのであればJavaがよい。システムがわかった気になる危
険な言語でもある。結論から言えばJavaができてもLinux コマンドやBash、カーネルの
理解は１つも進まない。

Bash版
　Linux/UNIXを学ぶのであればBash版をおすすめする。
　https://github.com/suzukiiichiro/AI_Algorithm_Bash

  なぜBashなのかは以下に書いておいた。
  https://github.com/suzukiiichiro/AI_Algorithm_Bash/blob/master/002UNIXBasic

  Bashは遅い。だが強力だ。Linuxの力を手に入れることができる。
  どの言語で学ぶのかを迷っているのであれば迷わず「Bash」を選んで欲しい。
  その次はLua->Java->Cだ。

Lua版
　スマートフォンアプリが世の中のテクノロジーを牽引しているのは間違いない。
　そのアプリ開発で幅を利かせているのがLua言語だ。コンパクトで高速、周りとの相性
も良いときている。


　上記、どの言語から始めても良いと思う。できる人はどの言語でもすらすら書ける。
　では、以下から本題に入る。


*****************************
  N-Queens問題とは
*****************************
 
　Nクイーン問題とは、「8列×8行のチェスボードに8個のクイーンを、互いに効きが当た
らないように並べよ」という８クイーン問題のクイーン(N)を、どこまで大きなNまで解を
求めることができるかという問題。
　クイーンとは、チェスで使われているクイーンを指し、チェス盤の中で、縦、横、斜め
にどこまでも進むことができる駒で、日本の将棋でいう「飛車と角」を合わせた動きとな
る。８列×８行で構成される一般的なチェスボードにおける8-Queens問題の解は、解の総
数は92個である。比較的単純な問題なので、学部レベルの演習問題として取り上げられる
ことが多い。
　8-Queens問題程度であれば、人力またはプログラムによる「力まかせ探索」でも解を求
めることができるが、Nが大きくなると解が一気に爆発し、実用的な時間では解けなくな
る。
　現在すべての解が判明しているものは、2004年に電気通信大学でIntel Pentium 4 Xeon
2.8GHzのプロセッサを68個搭載するPCクラスタ×20日をかけてn=24を解決し、世界一に、
その後2005 年にニッツァ大学でn=25、2009年にドレスデン工科大学で N-26、さらに2016
年に同工科大学でN=27の解を求めることに成功している。
　JeffSommers氏のビット演算を用いたエレガントなアルゴリズムに加え、対称解除法、
並列処理、部分解合成法、圧縮や枝刈りなど、先端技術でワールドレコードが次々と更新
されている。

 * https://ja.wikipedia.org/wiki/エイト・クイーン


はじめに
 　「エイト・クイーン問題」とは、チェスの盤とコマを使用したパズルの名称で、1848
 年、チェスプレイヤーのマックス・ベッツェルによって提案され、ガウスを含む多くの
 数学者がこの問題に挑戦した工学研究・離散数学、制約充足問題の頂点となる研究であ
 る。歴史　「エイト・クイーン問題」は、「縦8×横8のチェス盤上に、８つのクイーン
 を他のクイーンに緩衝することなくすべて配置せよ。というルールに従った場合、いく
 つの解が存在するのか？」という問題である。

１８７４年
 　考案から100年以上が経過した1874年、Guntherが行列式を用いて解く方法を提案し、
 イギリスのグレイシャー(Glaisher)によって全解(基本解)が12個であることが確認され
 た。

「Ｎ-クイーン問題」  
 　Ｎ-クイーン問題は、エイト・クイーン問題の拡張版で、チェス盤の一辺のマスの数Ｎ
 とクイーンの数Ｎとを同一にした制約充足問題で、Ｎの数が増えることにより、その解
 数と、処理時間が指数関数的に爆発的に増大することが知られている。


アルゴリズム
 　基本的にこの問題を解決するためには、バックトラックアルゴリズムと呼ばれる「総
 当たり法」が用いられる。このアルゴリズムは、「全ての選択肢を試し、途中で制約を
 満たさないことが明らかな場合は、以降のパターン生成を行わない」という単純なアル
 ゴリズムである。


日本記録
 　2004年「電気通信大学」が、Intel Pentium4Xeon 2.8GHzのプロセッサを68個搭載する
 PCクラスタ群で、６.6年分の計算を2２日かけて世界で初めてN24（２4×２4のチェス盤
 に２４個のクイーンを配置した場合の解を求める）を発見した。この記録は現在もなお
 日本記録となっている

世界記録
　2009年ドイツのドレスデン工科大学が、2.5GHz-QuadCoreCPUを704個搭載するサーバー
群で２４０日をかけてＮ２６を発見した。その基本解は、2789兆7124億6651万289個、転
回形などのバリエーション解を含めると、その数は2京2317兆6996億1636万4044個にもな
る。
　その後、同大学にてN２７が発見されたが、Ｎ２７の正解が現時点では未知のため、他
の研究者によって同じ解が発見されるまでＮ２７は認定されない。


将来技術開発室鈴木の試み
　現在、鈴木のＰＣでは、電気通信大学が保有するＮ２４の日本記録の一つ手前となるＮ
２３を解決し、その後も安定して処理が進捗している。実行環境は、通常業務を同ノート
パソコンで行いつつ、業務の合間を見て開発、挑戦している。
負荷が限りなく高い計算処理
　一般的に広く普及しているＮ−クイーンアルゴリズムは、Ｎ１７を解決するだけで処理
に数時間を要し、Ｎ２０あたりで計算量が物理メモリからあふれ、処理はアボート、また
はOSが危機を察知し自身を守るため、ＰＣの電源が切断または再起動を余儀なくされる。
運良くＮ２２に処理が移ったとしても、しばらくするとCPUが暴走、その後、ハングアッ
プ、または操作不能な状態に陥り、正常にＰＣの電源を切ることすらできない。

鈴木が用いた手法とアルãアルゴリズムを導入している。これにより同時に起動
する数千のスレッドそれぞれが受け持つ処理量の偏りが均一化され、処理が安定した。結
果、ノートＰＣといった非力なマシンでも処理がスタックし、オーバーフローまたはＰＣ
が落ちるなどといった事がなくなった。自動車のシフトギヤ−の導入である。

プログラミングソースとテキストの配布
　鈴木の実装は、一般的に知られている数十行からなるＮ−クイーンプログラムをベース
に、前述したいくつかのアルゴリズムを含む３０以上のアルゴリズムで構成されている。
プログラムは、シェルスクリプトのみで記述されたBash版、汎用的なJava版に加え、刀鍛
冶職人向けC言語版など、プログラム初学からひたむきに学習が可能な「ステップバイス
テップ」で記述されており、全てのプログラミングソースは以下に公開している。
https://github.com/suzukiiichiro/N-Queens

懸賞金
　Ｎ-クイーン問題は、2000年にアメリカのクレイ数学研究所、2017年にはスコットラン
ドアンドリュー大学などが、「P対NP問題」の証明につながるものとして懸賞金100万ドル
(約1億1000万円)をかけている。

おわりに　ＡＩ時代に乗り遅れるないために
　Google社は、人気立方体パズル「ルービックキューブ（Rubik's Cube）」の全パターン
を調べ上げ、どんな状態からでも20手以内で全面の色を揃えることを突き止めた。11億秒
(約35年)かかる演算処理をわずか２週間で（数百台のGPUサーバーを使って）解決したと
いう。（鈴木も挑戦している）
　資源に恵まれない企業にはもはや勝ち目はないのかとため息をつく中、革新的なアルゴ
リズムの発見によりＡＩブームの火付け役の一つとなった研究


 --------------------------------------- 共同通信社   電通大(N24） QJH(GPU)版  高橋謙一郎 Somers版(N22)
18:         666090624         83263591   00:00:00:30  00:00:00:12    00:00:25    00:03:48  00:19:26
19:        4968057848        621012754   00:00:05:08  00:00:00:42    00:03:17    00:29:22  02:31:24
20:       39029188884       4878666808   00:00:40:31  00:00:04:46    00:24:07    03:54:10  20:35:06
21:      314666222712      39333324973   00:05:38:49  00:00:41:37    03:05:28 01:09:17:19                      
22      2691008701644     336376244042   02:02:03:49  00:05:50:02 01:03:08:20                                       
23     24233937684440    3029242658210   22:12:20:11  02:08:52:30                         
24    227514171973736   28439272956934                21:18:10:31                         
25   2207893435808352  275986683743434
26  22317699616364044 2789712466510289
27 234907967154122528

24 ２００４年４月１１日 電気通信大学　 2009年4月 68CPU x 22日                  1,496CPU/日
25 ２００５年６月１１日 ProActive      単一CPU換算で５０年以上                18,250CPU/日 
26 ２００９年７月１１日 tu-dresden     FPGA ( *1 : 8*22 2.5 GHz-QuadCore systemsに相当
                                      （約176 * 4CPU = 704 CPU))  x ２４０日 168,960CPU/日
27 ２０１６年　月　　日 tu-dresden





 このテキストの ステップ
 *
 * NQueen1  しらみつぶし探索
 * NQueen2  縦横列に配置
 * NQueen3  縦横斜め列に配置                                  00:01:36
 * NQueen4  バックトラック                                    00:01:46
 * NQueen5  対称解除法(2004)                                  00:01:35
 * NQueen6  枝刈りと最適化 最上段クイーンの位置による条件分岐 00:00:23
 * NQueen7  枝刈りと最適化 symmetryOps()部分                  00:00:23
 * NQueen8  ビットマップ  nQueens()部分のbitmap対応           00:00:09
 * NQueen9  ビットマップ 最上段クイーンの位置による判定を導入 00:00:02
 * NQueen10 並列処理の下準備 シングルスレッド                 00:00:02
 * NQueen11 並列処理　シングルスレッド threadの実装           00:00:02
 * NQueen12 並列処理　マルチスレッド マルチスレッドの実装     00:00:00
 

 *
 */

  /**
   Java版 NQueen12 の実行結果

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

// ここからソースが始まります。
//
//
import org.apache.commons.lang3.time.DurationFormatUtils;
class NQueen{
	public static void main(String[] args){

		// new NQueen1(8);  //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen2(8);  //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen3();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen4();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen5();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen6();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen7();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen8();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen9();   //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen10();  //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen11();  //実行はコメントを外して $ ./MAIN.SH を実行
		// new NQueen12();  //実行はコメントを外して $ ./MAIN.SH を実行

	}
}


    /**
     * 
	   NQueen1 
		 Carl Gauss（1777-1855）のブルートフォース(しらみつぶし探索）
		 各縦列に１個の王妃を配置する組み合わせを再帰的に列挙

   　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
   なるかをチェックする方法(※)各行に１個の王妃を配置する組み合わせを再帰的に列
   挙組み合わせを生成するだけであって8王妃問題を解いているわけではない
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


/**
 * 各縦列に１個の王妃を配置する組み合わせを再帰的に列挙
 * 
 * @author suzukiiichiro
 *
 */
class NQueen1{
	private int		size;
	private int		count;
	private int[]	board;
	public NQueen1(int size){
		this.size=size;
		count=1;            // 解数は1からカウント
		board=new int[size];
		nQueens(0);         // ０列目に王妃を配置してスタート
	}
	private void nQueens(int row){
		if(row==size){      // 全列に配置完了 最後の列で出力
			print();
		}else{
			for(int col=0;col<size;col++){
				board[row]=col; // 各列にひとつのクイーンを配置する
				nQueens(row+1); // 次の列に王妃を配置
			}
		}
	}
	private void print(){
		for(int col=0;col<size;col++){
			System.out.printf("%2d",board[col]);
		}
		System.out.println(" : "+count++);
	}
}


    /**
	   NQueen2 Ellis HorowitzとSartaj Sahni(1978)の配置フラグ
		 各縦横列に１個の王妃を配置する組み合わせを再帰的に列挙
		 
		 実行結果 
		 6 1 5 2 0 3 7 4 : 83
		 6 2 0 5 7 4 1 3 : 84
		 6 2 7 1 4 0 5 3 : 85
		 6 3 1 4 7 0 2 5 : 86
		 6 3 1 7 5 0 2 4 : 87
		 6 4 2 0 5 7 1 3 : 88
		 7 1 3 0 6 4 2 5 : 89
		 7 1 4 2 0 6 3 5 : 90
		 7 2 0 5 1 4 6 3 : 91
		 7 3 0 2 5 1 6 4 : 92
		*/




/**
 * 各縦横列に１個の王妃を配置する組み合わせを再帰的に列挙 Ellis HorowitzとSartaj
 * Sahni(1978)の配置フラグ
 * 
 * @author suzukiiichiro
 *
 */
class NQueen2{
	private int[]	board;
	private int		count;
	private int		size;
	// コンストラクタ
	public NQueen2(int size){
		this.size=size;
		board=new int[size];
		count=1; // 解数は1からカウント
		nQueens(0); // ０列目に王妃を配置してスタート
	}
	// Horowitz and Sahni’s validity check
	private boolean Valid(int row){
		for(int Idx=0;Idx<row;Idx++){
			if(board[Idx]==board[row]||Math.abs(board[row]-board[Idx])==(row-Idx)){
				return false; // boolean false
			}
		}
		return true; // boolean true
	}
	// 再帰関数
	private void nQueens(int row){
		if(row==size){
			print(); // 全列に配置完了 最後の列で出力
		}else{
			for(int col=0;col<size;col++){
				board[row]=col; // 各列にひとつのクイーンを配置する
				if(Valid(row)){
					nQueens(row+1); // 次の列に王妃を配置
				}
			}
		}
	}
	// 出力
	private void print(){
		for(int col=0;col<size;col++){
			System.out.printf("%2d",board[col]);
		}
		System.out.println(" : "+count++);
	}
}


		/**
	   NQueen3  バックトラック
    各縦横列に加え斜め１個の王妃を配置する組み合わせの配置フラグ各列、対角線上に
    クイーンがあるかどうかのフラグを用意し、途中で制約を満たさない事が明らかな場
    合は、それ以降のパターン生成を行わない。各列、対角線上にクイーンがあるかどう
    かのフラグを用意することで高速化を図る。これまでは行方向と列方向に重複しない
    組み合わせを列挙するものですが、王妃は斜め方向のコマをとることができるので、
    どの斜めライン上にも王妃をひとつだけしか配置できない制限を加える事により、深
    さ優先探索で全ての葉を訪問せず木を降りても解がないと判明した時点で木を引き返
    すということができる。

		 実行結果 
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            0     00:00:00.000
			 5:               10            0     00:00:00.000
			 6:                4            0     00:00:00.000
			 7:               40            0     00:00:00.000
			 8:               92            0     00:00:00.000
			 9:              352            0     00:00:00.002
			10:              724            0     00:00:00.003
			11:             2680            0     00:00:00.014
			12:            14200            0     00:00:00.070
			13:            73712            0     00:00:00.374
			14:           365596            0     00:00:02.210
			15:          2279184            0     00:00:14.182
			16:         14772512            0     00:01:36.79
		*/





/**
 * Wirth's validity check(1986)の配置フラグ 各縦横列に１個の王妃を配置する組み合
 * わせを再帰的に列挙
 * 
 * @author suzukiiichiro
 *
 */
class NQueen3{
	private int[]			board;
	private int				size;
	private boolean[]	colChk,diagChk,antiChk;
	private long			TOTAL;
	// コンストラクタ
	public NQueen3(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			TOTAL=0;
			board=new int[size];
			colChk=new boolean[size];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,TOTAL,0,TIME);
		}
	}
	// 再帰関数
	private void nQueens(int row){
		if(row==size){
			TOTAL++;
    //  picture(board,size);
		}else{
			for(int col=0;col<size;col++){
				board[row]=col; // 各列にひとつのクイーンを配置する
				if(colChk[col]==false&&antiChk[row+col]==false&&diagChk[row-col+(size-1)]==false){
					colChk[col]=diagChk[row-board[row]+size-1]=antiChk[row+board[row]]=true;
					nQueens(row+1);
					colChk[col]=diagChk[row-board[row]+size-1]=antiChk[row+board[row]]=false;
				}
			}
		}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
}


		 /**
	   NQueen4  バックトラック
		   縦横斜めの配置フラグ アルゴリズムとデータ構造のNiklaus Wirth(1986)
		   実行結果
		   N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            0     00:00:00.000
			 5:               10            0     00:00:00.000
			 6:                4            0     00:00:00.000
			 7:               40            0     00:00:00.001
			 8:               92            0     00:00:00.002
			 9:              352            0     00:00:00.001
			10:              724            0     00:00:00.004
			11:             2680            0     00:00:00.014
			12:            14200            0     00:00:00.064
			13:            73712            0     00:00:00.376
			14:           365596            0     00:00:02.274
			15:          2279184            0     00:00:14.899
			16:         14772512            0     00:01:46.616 
		*/



/**
 * Wirth's validity check(1986)の配置フラグ 各縦横列に１個の王妃を配置する 組み
 * 合わせを再帰的に列挙
 * 
 * @author suzukiiichiro
 *
 */
class NQueen4{
	private int[]			board;
	private int				size;
	private boolean		opt1;
	private boolean[]	colChk,diagChk,antiChk;
	private long			TOTAL;
	// コンストラクタ
	public NQueen4(){
		opt1=true;
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			TOTAL=0;
			board=new int[size];
			colChk=new boolean[size];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,TOTAL,0,TIME);
		}
	}
	// Wirth's validity check
	void mark(int row,int col,boolean value){
		int idx;
		colChk[col]=value;
		idx=row-col+size-1;
		diagChk[idx]=value;
		idx=row+col;
		antiChk[idx]=value;
	}
	private boolean Valid(int row){
		int k;
		boolean chk;
		if(opt1){
			chk=colChk[board[row]];
			k=row-board[row]+size-1;
			chk=chk|diagChk[k];
			k=row+board[row];
			chk=chk|antiChk[k];
			return !chk; /* Valid if NOT any occupied */
		}else{
			for(int Idx=0;Idx<row;Idx++){
				if(board[Idx]==board[row]||Math.abs(board[row]-board[Idx])==(row-Idx)){
					return false; // boolean false
				}
			}
			return true; // boolean true
		}
	}
	// 再帰関数
	private void nQueens(int row){
		if(row==size){
			TOTAL++;
		}else{
			for(int col=0;col<size;col++){
				board[row]=col; // 各列にひとつのクイーンを配置する
				if(Valid(row)){// 効き筋チェック
					mark(row,board[row],true);
					nQueens(row+1); // 次の列に王妃を配置
					mark(row,board[row],false);
				}
			}
		}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
}


		 /**
		 NQueen5 対称解除法(2004)

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
 * 　さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純
 * にはいきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ
 * ＝５の下の例では要素数が２個のものと８個のものがあります。
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
 *   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３
     *   つの角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であ
     *   るかどうかを判定するには、右上角から左下角を通る斜軸で反転させたパター
     *   ンとの比較だけになります。突き詰めれば、
 * 
 * [上から２行目のクイーンの位置が右から何番目にあるか]
 * [右から２列目のクイーンの位置が上から何番目にあるか]
 * 
 *
 * を比較するだけで判定することができます。この２つの値が同じになることはないか
 * らです。
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
 *   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解であ
 *   るためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れ
 *   ておきます。
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
 *   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターン
 *   がオリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他
 *   の３つの角に写像させることができるので(×４)、このユニーク解が属するグルー
 *   プの要素数は必ず８個(＝２×４)になります。
 * 
 *   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にす
 *   るために次の事柄を確認します。
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
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパ
  *   ターンを加えて２個しかありません。(2)に該当するユニーク解が属するグループ
  *   の要素数は、180度回転させて同型になる場合は４個(左右反転×縦横回転)、そし
  *   て180度回転させてもオリジナルと異なる場合は８個になります。(左右反転×縦横
  *   回転×上下反転)
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
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。です
 * が、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛躍的に
 * 高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
 *
			 実行結果
			 
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.001
			 5:               10            2     00:00:00.000
			 6:                4            1     00:00:00.000
			 7:               40            6     00:00:00.000
			 8:               92           12     00:00:00.002
			 9:              352           46     00:00:00.002
			10:              724           92     00:00:00.006
			11:             2680          341     00:00:00.012
			12:            14200         1787     00:00:00.062
			13:            73712         9233     00:00:00.345
			14:           365596        45752     00:00:02.076
			15:          2279184       285053     00:00:13.782
			16:         14772512      1846955     00:01:35.638
		*/




/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen5{
	private int[]			board,trial,scratch;
	private int				size,nUnique,nTotal;
	private boolean[]	colChk,diagChk,antiChk;																				// Antidiagonal
	// コンストラクタ
	public NQueen5(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			nUnique=nTotal=0;
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
			colChk=new boolean[size];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,nTotal,nUnique,TIME);
		}
	}
	/* Check two vectors for equality; return first inequality (a la strncmp) */
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
	private void vMirror(int[] check,int n){
		int j;
		for(j=0;j<n;j++)
			check[j]=(n-1)-check[j];
		return;
	}
	int symmetryOps(){
		int k; /* String offset */
		int nEquiv; /* Number equivalent boards */
		/* Copy over; now trial will be subjected to the transformations */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		/* 90 degrees --- clockwise */
		rotate(trial,scratch,size,false);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k==0)
			nEquiv=1;
		else{
			/* 180 degrees */
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k==0)
				nEquiv=2;
			else{
				/* 270 degrees */
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				nEquiv=4;
			}
		}
		/* Reflect -- vertical mirror */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		/* -90 degrees --- equiv. to diagonal mirror */
		rotate(trial,scratch,size,true);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k<0){
			/* -180 degrees --- equiv. to horizontal mirror */
			rotate(trial,scratch,size,true);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k<0){
				/* -270 degrees --- equiv. to anti-diagonal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
			}
		}
		/* WE HAVE A GOOD ONE! */
		return nEquiv*2;
	}
	// Wirth's validity check
	private void mark(int row,int col,boolean value){
		int idx;
		colChk[col]=value;
		idx=row-col+size-1;
		diagChk[idx]=value;
		idx=row+col;
		antiChk[idx]=value;
	}
	private boolean Valid(int row){
		int k;
		boolean chk;
		chk=colChk[board[row]];
		k=row-board[row]+size-1;
		chk=chk|diagChk[k];
		k=row+board[row];
		chk=chk|antiChk[k];
		return !chk; /* Valid if NOT any occupied */
	}
	// 再帰関数
	private void nQueens(int row){
		if(row==size){
			int tst=symmetryOps();
			if(tst!=0){
				nUnique++;
				nTotal+=tst;
        // picture(board,size);
			}
		}else
			for(int col=0;col<size;col++){
				board[row]=col;
				if(Valid(row)){
					mark(row,board[row],true);
					nQueens(row+1);
					mark(row,board[row],false);
				}
			}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
}


		/**
		NQueen6 * 枝刈りと最適化 最上段クイーンの位置による条件分岐
  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省
  略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、右半
  分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左
  右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが
  出来ない


  1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
    １行目かどうかの条件判断はループ外に出してもよい
    処理時間的に有意な差はないので、分かりやすいコードを示した
  2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
    １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    鏡像についても、主対角線鏡像のみを判定すればよい
    ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい

  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置
  くことはできない
  置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくな
  る
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
			 6:                4            1     00:00:00.000
			 7:               40            6     00:00:00.001
			 8:               92           12     00:00:00.000
			 9:              352           46     00:00:00.002
			10:              724           92     00:00:00.002
			11:             2680          341     00:00:00.007
			12:            14200         1787     00:00:00.030
			13:            73712         9233     00:00:00.102
			14:           365596        45752     00:00:00.535
			15:          2279184       285053     00:00:03.673
			16:         14772512      1846955     00:00:23.243
		*/




/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen6{
	private int[]			board,trial,scratch;
	private int				size,nUnique,nTotal;
	private boolean[]	colChk,diagChk,antiChk;
	public NQueen6(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			nUnique=nTotal=0;
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
			colChk=new boolean[size];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,nTotal,nUnique,TIME);
		}
	}
	/* Check two vectors for equality; return first inequality (a la strncmp) */
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
	private void vMirror(int[] check,int n){
		int j;
		for(j=0;j<n;j++)
			check[j]=(n-1)-check[j];
		return;
	}
	private int symmetryOps(){
		int k; /* String offset */
		int nEquiv; /* Number equivalent boards */
		/* Copy over; now trial will be subjected to the transformations */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		/* 90 degrees --- clockwise */
		rotate(trial,scratch,size,false);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k==0)
			nEquiv=1;
		else{
			/* 180 degrees */
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k==0)
				nEquiv=2;
			else{
				/* 270 degrees */
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				nEquiv=4;
			}
		}
		/* Reflect -- vertical mirror */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		/* -90 degrees --- equiv. to diagonal mirror */
		rotate(trial,scratch,size,true);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k<0){
			/* -180 degrees --- equiv. to horizontal mirror */
			rotate(trial,scratch,size,true);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k<0){
				/* -270 degrees --- equiv. to anti-diagonal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
			}
		}
		/* WE HAVE A GOOD ONE! */
		return nEquiv*2;
	}
	// Wirth's validity check
	private void mark(int row,int col,boolean value){
		int idx;
		colChk[col]=value;
		idx=row-col+size-1;
		diagChk[idx]=value;
		idx=row+col;
		antiChk[idx]=value;
	}
	private boolean valid(int row){
		int k;
		boolean chk;
		chk=colChk[board[row]];
		k=row-board[row]+size-1;
		chk=chk|diagChk[k];
		k=row+board[row];
		chk=chk|antiChk[k];
		return !chk; /* Valid if NOT any occupied */
	}
	// 再帰関数
	private void nQueens(int row){
		int k,lim,vTemp;
		if(row<size-1){
			if(valid(row)){
				mark(row,board[row],true);
				nQueens(row+1);
				mark(row,board[row],false);
			}
			lim=(row!=0) ? size : (size+1)/2;
			for(k=row+1;k<lim;k++){
				vTemp=board[k];
				board[k]=board[row];
				board[row]=vTemp;
				if(valid(row)){
					mark(row,board[row],true);
					nQueens(row+1);
					mark(row,board[row],false);
				}
			}
			/* Regenerate original vector from row to size-1: */
			vTemp=board[row];
			for(k=row+1;k<size;k++)
				board[k-1]=board[k];
			board[k-1]=vTemp;
		}else // Complete permutation.
		{
			if(!valid(row))
				return;
			k=symmetryOps();
			if(k!=0){
				nUnique++;
				nTotal+=k;
			}
		}
		return;
	}
}


		/** 
		 NQueen7 枝刈りと最適化 symmetryOps()部分
     *
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
			
			実行結果
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.000
			 5:               10            2     00:00:00.000
			 6:                4            1     00:00:00.000
			 7:               40            6     00:00:00.000
			 8:               92           12     00:00:00.001
			 9:              352           46     00:00:00.002
			10:              724           92     00:00:00.001
			11:             2680          341     00:00:00.007
			12:            14200         1787     00:00:00.032
			13:            73712         9233     00:00:00.107
			14:           365596        45752     00:00:00.554
			15:          2279184       285053     00:00:03.688
			16:         14772512      1846955     00:00:23.284
		*/



/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen7{
	private int[]			board,trial,scratch;
	private int				size,nUnique,nTotal;
	private boolean[]	colChk,diagChk,antiChk;																							// Antidiagonal
	// コンストラクタ
	public NQueen7(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			nUnique=nTotal=0;
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
			colChk=new boolean[2*size-1];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,nTotal,nUnique,TIME);
		}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	/* Check two vectors for equality; return first inequality (a la strncmp) */
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
	private void vMirror(int[] check,int n){
		int j;
		for(j=0;j<n;j++)
			check[j]=(n-1)-check[j];
		return;
	}
	int unique(){
		return nUnique;
	}
	int total(){
		return nTotal;
	}
	private int symmetryOps(){
		int k; /* String offset                     */
		int nEquiv; /* Number equivalent boards          */
		/* Copy over; now trial will be subjected to the transformations    */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		/* 90 degrees --- clockwise */
		rotate(trial,scratch,size,false);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k==0)
			nEquiv=1;
		else{
			/* 180 degrees */
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k==0)
				nEquiv=2;
			else{
				/* 270 degrees */
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				nEquiv=4;
			}
		}
		/* Reflect -- vertical mirror */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(nEquiv>1)        // I.e., no four-fold rotational symmetry
		{
			/* -90 degrees --- equiv. to diagonal mirror */
			rotate(trial,scratch,size,true);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(nEquiv>2)     // I.e., no two-fold rotational symmetry
			{
				/* -180 degrees --- equiv. to horizontal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				/* -270 degrees --- equiv. to anti-diagonal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
			}
		}
		/* WE HAVE A GOOD ONE! */
		return nEquiv*2;
	}
	// Wirth's validity check
	private void mark(int row,int col,boolean value){
		int idx;
		colChk[col]=value;
		idx=row-col+size-1;
		diagChk[idx]=value;
		idx=row+col;
		antiChk[idx]=value;
	}
	private boolean valid(int row){
		int k;
		boolean chk;
		chk=colChk[board[row]];
		k=row-board[row]+size-1;
		chk=chk|diagChk[k];
		k=row+board[row];
		chk=chk|antiChk[k];
		return !chk; /* Valid if NOT any occupied */
	}
	// 再帰関数
	private void nQueens(int row){
		int k,lim,vTemp;
		if(row<size-1){
			if(valid(row)){
				mark(row,board[row],true);
				nQueens(row+1);
				mark(row,board[row],false);
			}
			lim=(row!=0) ? size : (size+1)/2;
			for(k=row+1;k<lim;k++){
				vTemp=board[k];
				board[k]=board[row];
				board[row]=vTemp;
				if(valid(row)){
					mark(row,board[row],true);
					nQueens(row+1);
					mark(row,board[row],false);
				}
			}
			/* Regenerate original vector from row to size-1: */
			vTemp=board[row];
			for(k=row+1;k<size;k++)
				board[k-1]=board[k];
			board[k-1]=vTemp;
		}else // Complete permutation.
		{
			if(!valid(row))
				return;
			k=symmetryOps();
			if(k!=0){
				nUnique++;
				nTotal+=k;
			}
		}
		return;
	}
}


		/**
		 NQueen8 ビットマップ  nQueens()部分のbitmap対応

 *   ビット演算を使って高速化 状態をビットマップにパックし、処理する
 *   単純なバックトラックよりも２０〜３０倍高速
 * 
 * 　ビットマップであれば、シフトにより高速にデータを移動できる。
 *  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
 *  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
 *  分。
 *
 * 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処
 * 理。
 * 　バックトラックよりも２０−３０倍高速。
 * 
 * ===================
 * 考え方 1
 * ===================
 *
 * 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひ
 * とつのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
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
 * また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目
 * (0x40)になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。
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
 * ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くこと
 * ができない位置ということになります。次にその３つのビットフィールドをORしたビッ
 * トフィールドをビット反転させます。つまり「配置可能なビットがONになったビット
 * フィールド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼
 * ぶとして、次の演算を行なってみます。
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
		 
		 実行結果
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            0     00:00:00.000
			 5:               10            0     00:00:00.000
			 6:                4            0     00:00:00.000
			 7:               40            0     00:00:00.000
			 8:               92            0     00:00:00.000
			 9:              352            0     00:00:00.001
			10:              724            0     00:00:00.001
			11:             2680            0     00:00:00.002
			12:            14200            0     00:00:00.008
			13:            73712            0     00:00:00.045
			14:           365596            0     00:00:00.248
			15:          2279184            0     00:00:01.439
			16:         14772512            0     00:00:09.711
		*/



/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen8{
	private int[]			board,trial,scratch;
	private int				size,nUnique,nTotal;
	private int				bit;
	private int				mask;
	private boolean[]	colChk,diagChk,antiChk;
	public NQueen8(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			nUnique=nTotal=0;
			mask=(1<<size)-1;
			board=new int[size];
			trial=new int[size];
			scratch=new int[size];
			colChk=new boolean[2*size-1];
			diagChk=new boolean[2*size-1];
			antiChk=new boolean[2*size-1];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(0,0,0,0); // ０列目に王妃を配置してスタート
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,nTotal,nUnique,TIME);
		}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	/* Check two vectors for equality; return first inequality (a la strncmp) */
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
	private void vMirror(int[] check,int n){
		int j;
		for(j=0;j<n;j++)
			check[j]=(n-1)-check[j];
		return;
	}
	int unique(){
		return nUnique;
	}
	int total(){
		return nTotal;
	}
	private int symmetryOps(){
		int k; /* String offset                     */
		int nEquiv; /* Number equivalent boards          */
		/* Copy over; now trial will be subjected to the transformations    */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		/* 90 degrees --- clockwise */
		rotate(trial,scratch,size,false);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(k==0)
			nEquiv=1;
		else{
			/* 180 degrees */
			rotate(trial,scratch,size,false);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(k==0)
				nEquiv=2;
			else{
				/* 270 degrees */
				rotate(trial,scratch,size,false);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				nEquiv=4;
			}
		}
		/* Reflect -- vertical mirror */
		for(k=0;k<size;k++)
			trial[k]=board[k];
		vMirror(trial,size);
		k=intncmp(board,trial,size);
		if(k>0)
			return 0;
		if(nEquiv>1)        // I.e., no four-fold rotational symmetry
		{
			/* -90 degrees --- equiv. to diagonal mirror */
			rotate(trial,scratch,size,true);
			k=intncmp(board,trial,size);
			if(k>0)
				return 0;
			if(nEquiv>2)     // I.e., no two-fold rotational symmetry
			{
				/* -180 degrees --- equiv. to horizontal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
				/* -270 degrees --- equiv. to anti-diagonal mirror */
				rotate(trial,scratch,size,true);
				k=intncmp(board,trial,size);
				if(k>0)
					return 0;
			}
		}
		/* WE HAVE A GOOD ONE! */
		return nEquiv*2;
	}
	// Wirth's validity check
	private void mark(int row,int col,boolean value){
		int idx;
		colChk[col]=value;
		idx=row-col+size-1;
		diagChk[idx]=value;
		idx=row+col;
		antiChk[idx]=value;
	}
	private boolean valid(int row){
		int k;
		boolean chk;
		chk=colChk[board[row]];
		k=row-board[row]+size-1;
		chk=chk|diagChk[k];
		k=row+board[row];
		chk=chk|antiChk[k];
		return !chk; /* Valid if NOT any occupied */
	}
	// 再帰関数
	private void nQueens(int row,int left,int down,int right){
		int bitmap=mask&~(left|down|right);
		int k=0;
		if(row==size){
			nTotal++;
			//			if(bitmap!=0){
			//				board[row]=bitmap;
			//				k=symmetryOps();
			//				if(k!=0){
			//					nUnique++;
			//					nTotal+=k;
			//				}
			//			}
		}else{
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				nQueens(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
		//		int k,lim,vTemp;
		//		if(row<size-1){
		//			if(valid(row)){
		////				mark(row,board[row],true);
		//				bitmap^=board[row]=bit=(-bitmap&bitmap);
		////					nQueens(row+1);
		//				nQueens(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		////				mark(row,board[row],false);
		//			}
		//			lim=(row!=0) ? size : (size+1)/2;
		//			for(k=row+1;k<lim;k++){
		////				vTemp=board[k];
		////				board[k]=board[row];
		////				board[row]=vTemp;
		//				if(valid(row)){
		//				bitmap^=board[row]=bit=(-bitmap&bitmap);
		////					mark(row,board[row],true);
		//					nQueens(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		////					nQueens(row+1);
		////					mark(row,board[row],false);
		//				}
		//			}
		//			/* Regenerate original vector from row to size-1: */
		////			vTemp=board[row];
		////			for(k=row+1;k<size;k++)
		////				board[k-1]=board[k];
		////			board[k-1]=vTemp;
		//		}else // Complete permutation.
		//		{
		////			if(!valid(row))
		////				return;
		//			k=symmetryOps();
		//			if(k!=0){
		//				nUnique++;
		//				nTotal+=k;
		//			}
		//		}
		//		return;
	}
}


		/** 
NQueen9  ビットマップ 最上段クイーンの位置による判定を導入
対称解除法部分のbitmap対応

■ユニーク解の個数を求める
先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユニー
ク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判定値の
方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はどうでしょ
う。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、それがユニー
ク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中央になければ
ならず、それは互いの効き筋にあたるので有り得ません。


***********************************************************************
最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
***********************************************************************
 
次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つの角
にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうかを判
定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになります。
突き詰めれば、

 [上から２行目のクイーンの位置が右から何番目にあるか]
 [右から２列目のクイーンの位置が上から何番目にあるか]
 

を比較するだけで判定することができます。この２つの値が同じになることはないからで
す。
 
       3 0
       ↓↓
 - - - - Q ←0
 - Q - - - ←3
 - - - - -         上から２行目のクイーンの位置が右から４番目にある。
 - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
 - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
 
   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 られる解は総てユニーク解であることが保証されます。
 
 - - - - X Q
 - Q - - X -
 - - - - X -
 - - - - X -
 - - - - - -
 - - - - - -
 
   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 ます。
 
 X X - - - Q X X
 X - - - - - - X
 - - - - - - - -
 - - - - - - - -
 - - - - - - - -
 - - - - - - - -
 X - - - - - - X
 X X - - - - X X
 
   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 になります。
 
 X X x x x Q X X
 X - - - x x x X
 C - - x - x - x
 - - x - - x - -
 - x - - - x - -
 x - - - - x - A
 X - - - - x - X
 X X B - - x X X


 ■ユニーク解から全解への展開
   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 いう考察が必要になってきます。
 
   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つ
 の角に写像させることができるので(×４)、このユニーク解が属するグループの要素数
 は必ず８個(＝２×４)になります。
 
   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 ために次の事柄を確認します。

 TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
    ジナルと同型になる。  

    COUNT2 * 2
 
   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
    り得る。 

    COUNT4 * 4
 
   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパター
   ンを加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数
  は、180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転さ
  せてもオリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 
    COUNT8 * 8 

   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 りを外す必要がなくなったというわけです。 
 
   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)

 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛躍的
 に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
     *
		 * 
		 実行結果
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.000
			 5:               18            3     00:00:00.000
			 6:                4            1     00:00:00.000
			 7:               48            7     00:00:00.000
			 8:              100           13     00:00:00.000
			 9:              408           53     00:00:00.000
			10:              756           96     00:00:00.000
			11:             2744          349     00:00:00.001
			12:            14472         1821     00:00:00.003
			13:            75752         9488     00:00:00.010
			14:           369516        46242     00:00:00.054
			15:          2306552       288474     00:00:00.323
			16:         14898664      1862724     00:00:02.087
		*/


/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen9{
	private int[]	board;
	private int		size;
	private int		bit;
	private int		mask;
	private int		sizeE;
	private int		topbit,endbit,sidemask,lastmask;
	private long	COUNT2,COUNT4,COUNT8,UNIQUE,TOTAL;
	private int		bound1,bound2;
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	private void symmetryOps(int bitmap){
		//90度回転
		if(board[bound2]==1){
			int own=1;
			for(int ptn=2;own<=sizeE;own++,ptn<<=1){
				int bit=1;
				int bown=board[own];
				for(int you=sizeE;(board[you]!=ptn)&&(bown>=bit);you--)
					bit<<=1;
				if(bown>bit){
					return;
				}
				if(bown<bit){
					break;
				}
			}
			//90度回転して同型なら180度/270度回転も同型である
			if(own>sizeE){
				COUNT2++;
				return;
			}
		}
		//180度回転
		if(bitmap==endbit){
			int own=1;
			for(int you=sizeE-1;own<=sizeE;own++,you--){
				int bit=1;
				for(int ptn=topbit;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)
					bit<<=1;
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if(own>sizeE){
				COUNT4++;
				return;
			}
		}
		//270度回転
		if(board[bound1]==topbit){
			int own=1;
			for(int ptn=topbit>>1;own<=sizeE;own++,ptn>>=1){
				int bit=1;
				for(int you=0;board[you]!=ptn&&board[own]>=bit;you++){
					bit<<=1;
				}
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
		}
		COUNT8++;
	}
	// 再帰関数
	private void backTrack2(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				if((bitmap&lastmask)==0){
					board[row]=bitmap;
					symmetryOps(bitmap);
				}
			}
		}else{
			if(row<bound1){
				bitmap&=~sidemask;
			}else if(row==bound2){
				if((down&sidemask)==0){
					return;
				}
				if((down&sidemask)!=sidemask){
					bitmap&=sidemask;
				}
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void backTrack1(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				board[row]=bitmap;
				COUNT8++;
			}
		}else{
			if(row<bound1){
				bitmap&=~sidemask;
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void nQueens(int size){
		board[0]=1;
		sizeE=size-1;
		mask=(1<<size)-1;
		topbit=1<<sizeE;
		/* 0行目:000000001(固定) */
		/* 1行目:011111100(選択) */
		for(bound1=2;bound1<sizeE;bound1++){
			board[1]=bit=(1<<bound1);
			backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
		}
		/* 0行目:000001110(選択) */
		sidemask=lastmask=(topbit|1);
		endbit=(topbit>>1);
		for(bound1=1,bound2=size-2;bound1<bound2;bound1++,bound2--){
			board[0]=bit=(1<<bound1);
			backTrack2(1,bit<<1,bit,bit>>1);
			lastmask|=lastmask>>1|lastmask<<1;
			endbit>>=1;
		}
		UNIQUE=COUNT8+COUNT4+COUNT2;
		TOTAL=(COUNT8*8)+(COUNT4*4)+(COUNT2*2);
	}
	public NQueen9(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			COUNT8=COUNT4=COUNT2=UNIQUE=TOTAL=0;
			board=new int[size];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(size);
			long end=System.currentTimeMillis();
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,TOTAL,UNIQUE,TIME);
		}
	}
}



		/**
		 NQueen10 * 並列処理の下準備 シングルスレッド
		 実行結果
		   N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.000
			 5:               10            2     00:00:00.000
			 6:                4            1     00:00:00.000
			 7:               40            6     00:00:00.000
			 8:               92           12     00:00:00.000
			 9:              352           46     00:00:00.001
			10:              724           92     00:00:00.001
			11:             2680          341     00:00:00.001
			12:            14200         1787     00:00:00.004
			13:            73712         9233     00:00:00.010
			14:           365596        45752     00:00:00.053
			15:          2279184       285053     00:00:00.317
			16:         14772512      1846955     00:00:02.089 
		*/


/**
 * 
 * @author suzukiiichiro
 *
 */
class NQueen10{
	private int[]	board;
	private int		mask;
	private int		size,sizeE;
	private int		topbit,endbit,sidemask,lastmask,bound1,bound2,B1,B2;
	private long	COUNT2,COUNT4,COUNT8,UNIQUE,TOTAL;
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	private void symmetryOps(int bitmap){
		int own,you,ptn,bit;
		//90度回転
		if(board[bound2]==1){
			own=1;
			for(ptn=2;own<=sizeE;own++,ptn<<=1){
				bit=1;
				int bown=board[own];
				for(you=sizeE;(board[you]!=ptn)&&(bown>=bit);you--)
					bit<<=1;
				if(bown>bit){
					return;
				}
				if(bown<bit){
					break;
				}
			}
			//90度回転して同型なら180度/270度回転も同型である
			if(own>sizeE){
				COUNT2++;
				return;
			}
		}
		//180度回転
		if(bitmap==endbit){
			own=1;
			for(you=sizeE-1;own<=sizeE;own++,you--){
				bit=1;
				for(ptn=topbit;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)
					bit<<=1;
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if(own>sizeE){
				COUNT4++;
				return;
			}
		}
		//270度回転
		if(board[bound1]==topbit){
			own=1;
			for(ptn=topbit>>1;own<=sizeE;own++,ptn>>=1){
				bit=1;
				for(you=0;board[you]!=ptn&&board[own]>=bit;you++){
					bit<<=1;
				}
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
		}
		COUNT8++;
	}
	private void backTrack2(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				if((bitmap&lastmask)==0){
					board[row]=bitmap;
					symmetryOps(bitmap);
				}
			}
		}else{
			if(row<bound1){
				bitmap&=~sidemask;
			}else if(row==bound2){
				if((down&sidemask)==0){
					return;
				}
				if((down&sidemask)!=sidemask){
					bitmap&=sidemask;
				}
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void backTrack1(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				board[row]=bitmap;
				COUNT8++;
			}
		}else{
			if(row<bound1){
				bitmap&=~2;
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void BOUND2(int B1,int B2){
		int bit;
		bound1=B1;
		bound2=B2;
		board[0]=bit=(1<<bound1);
		backTrack2(1,bit<<1,bit,bit>>1);
		lastmask|=lastmask>>1|lastmask<<1;
		endbit>>=1;
	}
	private void BOUND1(int B1){
		int bit;
		bound1=B1;
		board[1]=bit=(1<<bound1);
		backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
	}
	private void nQueens(int size){
		board[0]=1;
		sizeE=size-1;
		mask=(1<<size)-1;
		topbit=1<<sizeE;
		B1=2;
		while(B1>1&&B1<sizeE){
			BOUND1(B1);
			B1++;
		}
		sidemask=lastmask=(topbit|1);
		endbit=(topbit>>1);
		B1=1;
		B2=size-2;
		while(B1>0&&B2<size-1&&B1<B2){
			BOUND2(B1,B2);
			B1++;
			B2--;
		}
	}
	public NQueen10(){
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(size=4;size<max;size++){
			COUNT8=COUNT4=COUNT2=UNIQUE=TOTAL=0;
			board=new int[size];
			for(int k=0;k<size;k++){
				board[k]=k;
			}
			long start=System.currentTimeMillis();
			nQueens(size);
			long end=System.currentTimeMillis();
			UNIQUE=COUNT8+COUNT4+COUNT2;
			TOTAL=(COUNT8*8)+(COUNT4*4)+(COUNT2*2);
			String TIME=DurationFormatUtils.formatPeriod(start,end,"HH:mm:ss.SSS");
			System.out.printf("%2d:%17d%13d%17s%n",size,TOTAL,UNIQUE,TIME);
		}
	}
}


		/**
		 NQueen11  並列処理　シングルスレッド threadの実装
		 
		 実行結果
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.001
			 5:               10            2     00:00:00.000
			 6:                4            1     00:00:00.001
			 7:               40            6     00:00:00.000
			 8:               92           12     00:00:00.000
			 9:              352           46     00:00:00.001
			10:              724           92     00:00:00.001
			11:             2680          341     00:00:00.002
			12:            14200         1787     00:00:00.005
			13:            73712         9233     00:00:00.012
			14:           365596        45752     00:00:00.059
			15:          2279184       285053     00:00:00.319
			16:         14772512      1846955     00:00:02.075
		*/




/**
 * 
 * @author suzukiiichiro
 *
 */
class NQ11_Board{
	private long COUNT8,COUNT4,COUNT2;
	public NQ11_Board(){
		COUNT8=COUNT4=COUNT2=0;
	}
	public long getTotal(){
		return COUNT8*8+COUNT4*4+COUNT2*2;
	}
	public long getUnique(){
		return COUNT8+COUNT4+COUNT2;
	}
	public synchronized void setCount(long COUNT8,long COUNT4,long COUNT2){
		this.COUNT8+=COUNT8;
		this.COUNT4+=COUNT4;
		this.COUNT2+=COUNT2;
	}
}
class NQ11_WorkingEngine extends Thread{
	private int[]								board;
	private int									mask;
	private int									size,sizeE;
	private int									topbit,endbit,sidemask,lastmask,bound1,bound2,B1,B2;
	private NQ11_WorkingEngine	child;
	private NQ11_Board					info;
	private int									nMore;
	private boolean							bThread	=false;
	public NQ11_WorkingEngine(int size,int nMore,NQ11_Board info,int B1,int B2){
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
					child=new NQ11_WorkingEngine(size,nMore-1,info,B1-1,B2+1);
					child.start();
					//          child.join();
				}
			}catch(Exception e){
				System.out.println(e);
			}
		}else{
			child=null;
		}
	}
	private void BOUND2(int B1,int B2){
		int bit;
		bound1=B1;
		bound2=B2;
		board[0]=bit=(1<<bound1);
		backTrack2(1,bit<<1,bit,bit>>1);
		lastmask|=lastmask>>1|lastmask<<1;
		endbit>>=1;
	}
	private void BOUND1(int B1){
		int bit;
		bound1=B1;
		board[1]=bit=(1<<bound1);
		backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
	}
	public void run(){
		if(child==null){
			if(nMore>0){
				board[0]=1;
				sizeE=size-1;
				mask=(1<<size)-1;
				topbit=1<<sizeE;
				B1=2;
				while(B1>1&&B1<sizeE){
					BOUND1(B1);
					B1++;
				}
				sidemask=lastmask=(topbit|1);
				endbit=(topbit>>1);
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
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	private void symmetryOps(int bitmap){
		int own,you,ptn,bit;
		//90度回転
		if(board[bound2]==1){
			own=1;
			for(ptn=2;own<=sizeE;own++,ptn<<=1){
				bit=1;
				int bown=board[own];
				for(you=sizeE;(board[you]!=ptn)&&(bown>=bit);you--)
					bit<<=1;
				if(bown>bit){
					return;
				}
				if(bown<bit){
					break;
				}
			}
			//90度回転して同型なら180度/270度回転も同型である
			if(own>sizeE){
				//				COUNT2++;
				info.setCount(0,0,1);
				return;
			}
		}
		//180度回転
		if(bitmap==endbit){
			own=1;
			for(you=sizeE-1;own<=sizeE;own++,you--){
				bit=1;
				for(ptn=topbit;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)
					bit<<=1;
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if(own>sizeE){
				//				COUNT4++;
				info.setCount(0,1,0);
				return;
			}
		}
		//270度回転
		if(board[bound1]==topbit){
			own=1;
			for(ptn=topbit>>1;own<=sizeE;own++,ptn>>=1){
				bit=1;
				for(you=0;board[you]!=ptn&&board[own]>=bit;you++){
					bit<<=1;
				}
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
		}
		//		COUNT8++;
		info.setCount(1,0,0);
	}
	private void backTrack2(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				if((bitmap&lastmask)==0){
					board[row]=bitmap;
					symmetryOps(bitmap);
				}
			}
		}else{
			if(row<bound1){
				bitmap&=~sidemask;
			}else if(row==bound2){
				if((down&sidemask)==0){
					return;
				}
				if((down&sidemask)!=sidemask){
					bitmap&=sidemask;
				}
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void backTrack1(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				board[row]=bitmap;
				//				COUNT8++;
				info.setCount(1,0,0);
			}
		}else{
			if(row<bound1){
				bitmap&=~2;
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
}
class NQueen11{
	public NQueen11(){
		NQ11_Board info;
		NQ11_WorkingEngine child;
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(int size=4;size<max;size++){
			int nThreads=size;
			long start=System.currentTimeMillis();
			info=new NQ11_Board();
			try{
				child=new NQ11_WorkingEngine(size,nThreads,info,size-1,0);
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
}



		/**

NQueen12  並列処理　マルチスレッド マルチスレッドの実装

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
 *   
		実行結果
			 N:            Total       Unique     hh:mm:ss.SSS
			 4:                2            1     00:00:00.002
			 5:               10            2     00:00:00.001
			 6:                4            1     00:00:00.001
			 7:               40            6     00:00:00.001
			 8:               92           12     00:00:00.001
			 9:              352           46     00:00:00.001
			10:              724           92     00:00:00.001
			11:             2680          341     00:00:00.003
			12:            14200         1787     00:00:00.004
			13:            73712         9233     00:00:00.005
			14:           365596        45752     00:00:00.023
			15:          2279184       285053     00:00:00.112
			16:         14772512      1846955     00:00:00.637
		*/


/**
 * 
 * @author suzukiiichiro
 *
 */
class NQ12_Board{
	private long COUNT8,COUNT4,COUNT2;
	public NQ12_Board(){
		COUNT8=COUNT4=COUNT2=0;
	}
	public long getTotal(){
		return COUNT8*8+COUNT4*4+COUNT2*2;
	}
	public long getUnique(){
		return COUNT8+COUNT4+COUNT2;
	}
	public synchronized void setCount(long COUNT8,long COUNT4,long COUNT2){
		this.COUNT8+=COUNT8;
		this.COUNT4+=COUNT4;
		this.COUNT2+=COUNT2;
	}
}
class NQ12_WorkingEngine extends Thread{
	private int[]								board;
	private int									mask;
	private int									size,sizeE;
	private int									topbit,endbit,sidemask,lastmask,bound1,bound2;
	private int									B1,B2;
	private NQ12_WorkingEngine	child;
	private NQ12_Board					info;
	private int									nMore;
	private boolean							bThread	=true;
	public NQ12_WorkingEngine(int size,int nMore,NQ12_Board info,int B1,int B2){
		this.size=size;
		this.info=info;
		this.nMore=nMore;
		this.B1=B1;
		this.B2=B2;
		board=new int[size];
		for(int k=0;k<size;k++){
			board[k]=k;
		}
		if(nMore>0){
			try{
				if(bThread){
					child=new NQ12_WorkingEngine(size,nMore-1,info,B1-1,B2+1);
					child.start();
				}
			}catch(Exception e){
				System.out.println(e);
			}
		}else{
			child=null;
		}
	}
	private void BOUND2(int B1,int B2){
		int bit;
		bound1=B1;
		bound2=B2;
		board[0]=bit=(1<<bound1);
		backTrack2(1,bit<<1,bit,bit>>1);
		lastmask|=lastmask>>1|lastmask<<1;
		endbit>>=1;
	}
	private void BOUND1(int B1){
		int bit;
		bound1=B1;
		board[1]=bit=(1<<bound1);
		backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
	}
	public void run(){
		// シングルスレッド
		if(child==null){
			if(nMore>0){
				board[0]=1;
				sizeE=size-1;
				mask=(1<<size)-1;
				topbit=1<<sizeE;
				B1=2;
				while(B1>1&&B1<sizeE){
					BOUND1(B1);
					B1++;
				}
				sidemask=lastmask=(topbit|1);
				endbit=(topbit>>1);
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
			mask=(1<<size)-1;
			topbit=1<<sizeE;
			if(B1>1&&B1<sizeE){
				BOUND1(B1);
			}
			endbit=(topbit>>B1);
			sidemask=lastmask=(topbit|1);
			if(B1>0&&B2<size-1&&B1<B2){
				for(int i=1;i<B1;i++){
					lastmask=lastmask|lastmask>>1|lastmask<<1;
				}
				BOUND2(B1,B2);
				endbit>>=nMore;
			}
			try{
				child.join();
			}catch(Exception e){
				System.out.println(e);
			}
		}
	}
	static void picture(int[] board,int size){
		int row,col,tst;
		for(row=0;row<size;row++){
			System.out.println();
			tst=board[row];
			for(col=0;col<size;col++){
				System.out.print(" "+(col==tst ? "Q" : "."));
			}
		}
		System.out.println('\n');
	}
	private void symmetryOps(int bitmap){
		int own,you,ptn,bit;
		//90度回転
		if(board[bound2]==1){
			own=1;
			for(ptn=2;own<=sizeE;own++,ptn<<=1){
				bit=1;
				int bown=board[own];
				for(you=sizeE;(board[you]!=ptn)&&(bown>=bit);you--)
					bit<<=1;
				if(bown>bit){
					return;
				}
				if(bown<bit){
					break;
				}
			}
			//90度回転して同型なら180度/270度回転も同型である
			if(own>sizeE){
				//				COUNT2++;
				info.setCount(0,0,1);
				return;
			}
		}
		//180度回転
		if(bitmap==endbit){
			own=1;
			for(you=sizeE-1;own<=sizeE;own++,you--){
				bit=1;
				for(ptn=topbit;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)
					bit<<=1;
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
			//90度回転が同型でなくても180度回転が同型である事もある
			if(own>sizeE){
				//				COUNT4++;
				info.setCount(0,1,0);
				return;
			}
		}
		//270度回転
		if(board[bound1]==topbit){
			own=1;
			for(ptn=topbit>>1;own<=sizeE;own++,ptn>>=1){
				bit=1;
				for(you=0;board[you]!=ptn&&board[own]>=bit;you++){
					bit<<=1;
				}
				if(board[own]>bit){
					return;
				}
				if(board[own]<bit){
					break;
				}
			}
		}
		//		COUNT8++;
		info.setCount(1,0,0);
	}
	private void backTrack2(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				if((bitmap&lastmask)==0){
					board[row]=bitmap;
					symmetryOps(bitmap);
				}
			}
		}else{
			if(row<bound1){
				bitmap&=~sidemask;
			}else if(row==bound2){
				if((down&sidemask)==0){
					return;
				}
				if((down&sidemask)!=sidemask){
					bitmap&=sidemask;
				}
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
	private void backTrack1(int row,int left,int down,int right){
		int bit;
		int bitmap=mask&~(left|down|right);
		if(row==sizeE){
			if(bitmap!=0){
				board[row]=bitmap;
				//				COUNT8++;
				info.setCount(1,0,0);
			}
		}else{
			if(row<bound1){
				bitmap&=~2;
			}
			while(bitmap!=0){
				bitmap^=board[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
				backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
			}
		}
	}
}
class NQueen12{
	public NQueen12(){
		NQ12_Board info;
		NQ12_WorkingEngine child;
		int max=27;
		System.out.println(" N:            Total       Unique     hh:mm:ss.SSS");
		for(int size=4;size<max;size++){
			int nThreads=size;
			long start=System.currentTimeMillis();
			info=new NQ12_Board();
			try{
				child=new NQ12_WorkingEngine(size,nThreads,info,size-1,0);
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
}
