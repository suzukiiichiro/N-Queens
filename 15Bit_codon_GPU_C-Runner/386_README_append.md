## 386: 検証専用ヘルパー関数の別ファイル分離

**位置づけ**: 384で実機確認した「ローカルマルチファイルimport」の仕組みを、
実際の本番ファイル(385Py→386Py)に対して初めて適用したrevision。関数の
**移動のみ**で、ロジック変更は一切なし。

**移動した12関数**(`385Py_kernel_maxd14_final.py` → `rev386_validation_
helpers.py`):

```
validate_chunk_range          validate_reordered_count
validate_reordered_indices    file_exists
validate_bin_file             count_constellations_bin_records
read_stream_done_count        write_stream_done_count
read_vmhwm_kb                 crunner_parse_result
crunner_input_fname           crunner_input_valid
```

**選定基準**: `TaskSoA` / `build_soa_for_range` / `meta_next`スケジュール
テーブルに一切依存しない、純粋なファイルI/O・整合性チェック・文字列
パース関数のみ。`check_required_maxd_for_N`や`probe_partial_load_memory`
は目的こそ診断的だが上記コア構造体に依存するため、循環importかコア構造体
ごと移動するかの選択を迫られる。今回はどちらも避け、依存ゼロの関数群に
絞った**保守的な一巡目**。

**移動元・移動先の完全一致を機械的に確認済み**: 12関数すべてについて、
385の元ソースから抽出したテキストと`rev386_validation_helpers.py`内の
該当関数が完全に一致することをスクリプトで検証した(diffではなく
文字列包含チェックだが、空白・改行含め完全一致でなければ検出される)。
呼び出し箇所は計57箇所、すべて`rev386_validation_helpers.`接頭辞を
機械的に付与(コメント行・docstring内の地の文への誤爆はスキップ)。

**結果**: (鈴木さんの実機実行後、ここに追記。`386_validate.sh`は静的な
関数所在チェックに加え、385と同じ`bench_mode=37` N=21実行が同じ結果を
再現するかを見る回帰チェック)

---

## 次のステップ: 不要パラメータ・分岐の棚卸し(387候補、未着手)

ご依頼のあった「不要となった過去のパラメータ処理や分岐」について、386では
**あえて何も削除していません**。385で「型は通ったが中身が違う」というミスが
実機40分の空振りを生んだばかりで、削除は移動よりリスクが高い(使われなく
なったと"思い込んで"消したものが実は誰かのワンライナーに残っていた、という
事故が最も避けたいパターン)ためです。

以下、コード上「今は使われていなさそうに見える」候補を洗い出しました。
**いずれも憶測であり、鈴木さんに実際の使用状況を確認していただくまでは
何も変更しません**:

| 候補 | 根拠(コード上の観察) | 確認したいこと |
|---|---|---|
| `bench_mode==28`(broadmarktail-reorder-sim-only)/`29`(同gpu) | 330で`BROADMARK_VARIANT`軸がクローズ済み、かつ`bench_mode==31`(split145フル)が内部で同じビルドを暗黙に行うため、28/29を単独実行する場面が今も残っているか不明 | 直近でこの2つを単独で叩いたことがあるか |
| `worker_id`/`worker_count`(`bench_mode==29,31`のマルチワーカー分割) | 単一マシン・単一GPUの運用が続いている前提だと、分割実行の出番が無さそうに見える | 複数ワーカーでの並列実行を今後使う予定があるか |
| `broadmark_tail_variant`のCLI経由指定 | `A10G_FINAL_DEFAULT_BROADMARK_VARIANT=2`で固定採用済み(軸クローズ)なので、CLIで上書きする経路自体が今は使われていなさそうに見える | variantを2以外で試す予定が残っているか |
| `bench_mode==33`(Codon単体single-shot) | 385で`bench_mode==37`(Cバイナリ経由)の方が速いことを確認したが、373の「Codon-C 13.6%差の原因究明」が未着手のまま残っているため、比較用として現役の可能性が高い | 13.6%差の調査に33を今後も使うか、それとも別の比較方法に切り替えるか |
| `bench_mode==34/35/36`(maxd-check/diag/mem-probe) | N=22時代の診断用だが、今週のN=23以降maxd拡張でも同じ形の診断が要りそうに見える | 386以降もこのまま使う想定か、拡張時に作り直すか |

このテーブルはあくまで**棚卸しの叩き台**です。ご確認いただいた上で、
「消してよい」「まだ要る」「作り直す」を項目ごとに教えていただければ、
387としてorderly進めます。
