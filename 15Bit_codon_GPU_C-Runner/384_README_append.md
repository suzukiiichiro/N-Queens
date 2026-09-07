## 384: マルチファイルimport + os.system往復の実現性プローブ

**位置づけ**: 383本実装(maxdでCRunnerバイナリを振り分けるos.systemディスパッチ、
および検証用関数群の別ファイル分離)に着手する前の、実機確認専用revision。
カーネルコード・GPUコード・constellations生成は一切含まない。

**確認したい2点**:

1. `codon build -release` で、同一ディレクトリ内の別`.py`ファイルを
   `import`した状態でもビルドが通るか
2. `os.system()`で外部コマンドを実行し、その出力をログファイル経由で
   Codonのネイティブfile I/Oから読み戻せるか
   (383の本実装がCRunnerバイナリ呼び出しに使う予定の仕組みそのもの)

**構成**:

```
384Py_multifile_osexec_probe.py   # エントリポイント。ビルド対象。
rev384_helper_probe.py            # importされる側のヘルパー。
384_validate.sh                   # ビルド+実行+マーカー行チェック。
```

**命名の例外**: importされる側のファイルは `384Py_helper_probe.py` ではなく
`rev384_helper_probe.py` としている。Python/Codonの`import`文はモジュール名を
識別子として要求し、識別子は数字始まりにできない(`import 384Py_helper_probe`
は構文エラー)ため。import対象になるファイルに限った一回限りの例外で、
プロジェクト全体の命名規則を変えるものではない。

**事前予測(実行前に記録)**:

- **(1) import**: 成立する可能性が高いと見ている。Codon公式ドキュメント
  (Standard Library / Usage)を確認した限り、`-pyext`拡張ビルドモード固有の
  既知の制約(GitHub Issue #647など、複数ファイルのPython拡張ビルドで
  `__init__.py`以外がコンパイルされない問題)はあるが、これは`codon build
  -pyext`固有の話であり、今回使う通常の`codon build -release`単体実行ファイル
  ビルドでのローカルimportを妨げる記述は見当たらなかった。
- **(2) os.system往復**: ほぼ確実に成立すると見ている。`os.system(cmd:str)
  ->int`はCodon公式ドキュメントにネイティブ対応モジュールとして明記されて
  いる(`subprocess`はネイティブ非対応)。`382_validate.sh`など既存のbash
  ハーネスと同型のログファイル+マーカー行grepパターンを、Codon内の
  ネイティブfile I/Oで再現するだけであり、新規の依存は増えない。

**判定基準**: `384_validate.sh`が`384 PASSED`で終了すれば両方YES。
`codon_build_succeeded`でFAILすれば(1)がNO、`osexec_probe`でFAILすれば
(2)がNOと確定する。

**結果**: 2026-09-07、cudacodon実機(`NQ_CRunner$`)で`bash 384_validate.sh`実行、
`OK=7 FAIL=0`で`384 PASSED`。

- Q1(ローカルマルチファイルimport): **PASS**。`codon build -release`は
  `rev384_helper_probe.py`のimportを含んだまま単一バイナリとしてビルドに
  成功し、`helper_expected_value()`/`helper_marker_line()`とも実際に呼び出され
  実データ(`value=12345`)を返した。
- Q2(os.system+ログファイル往復): **PASS**。`os.system()`で書き出した
  ログファイルを、Codonネイティブのfile I/Oで読み戻し、3つのマーカー行
  (`start`/`value`/`done`)すべてを正しく検出した。

**この後の分岐(確定)**: 両方PASSしたため、383系の本実装(maxd→CRunnerバイナリ
対応表、os.systemディスパッチ、検証関数の別ファイル分離)は、単一ファイル構成
への後退や別方式への再検討なしに、このプローブで確認した仕組みのまま進める。
なお384を確認専用revisionとして消費したため、本実装側のrevision番号は385から
採番する。

**この後の分岐**:
- 両方PASS → 383の本実装(maxd→CRunnerバイナリの対応表、os.system
  ディスパッチ、検証関数の別ファイル分離)へ単一variableずつ進める。
- (1)がNO → 検証関数の別ファイル分離は見送り、単一ファイル構成のまま
  383を進める。
- (2)がNO → 383のCRunner連携は別の呼び出し方式(`from python import
  subprocess`経由など)を再検討する。
