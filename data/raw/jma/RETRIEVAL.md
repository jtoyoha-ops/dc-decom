# 気象庁の時別値の取得記録（DC-DECOM v2）

## 取得したもの
- サービス：気象庁「過去の気象データ・ダウンロード」 https://www.data.jma.go.jp/risk/obsdl/
  - 2026年11月時点のページは React で作り直されており、CSV は `POST https://www.data.jma.go.jp/risk/obsdl/show/table` で返る（ページの JavaScript `web/dist/top.bundle.js` が送る項目と同じ項目を送った。セッション ID は不要）。
- 取得日：2026-11-06（JST 18:58〜18:59）。地点ごとに1回、計5回の要求。間隔5秒。
- 地点（括弧は obsdl の地点番号 `s` + 国際地点番号）：

| 地域 | 地点 | 番号 | ファイル | バイト数 | sha256（先頭16桁） |
|---|---|---|---|---|---|
| Hokkaido | 札幌 | s47412 | jma_hourly_47412_Sapporo_2020-2023.csv | 1,265,042 | af0fd596ad2c3f8c |
| Tohoku | 仙台 | s47590 | jma_hourly_47590_Sendai_2020-2023.csv | 1,263,649 | f4ddd34d4d02a84d |
| Kanto | 東京 | s47662 | jma_hourly_47662_Tokyo_2020-2023.csv | 1,266,373 | 4277e8ea6c4bb943 |
| Kansai | 大阪 | s47772 | jma_hourly_47772_Osaka_2020-2023.csv | 1,266,731 | c3dcc37cf327a2b5 |
| Kyushu | 福岡 | s47807 | jma_hourly_47807_Fukuoka_2020-2023.csv | 1,268,183 | ae7f0d30e40f4730 |

- 要素：気温（要素番号 201、℃、毎正時の値）、全天日射量（要素番号 610、MJ/㎡、前1時間の積算）。品質情報・均質番号つき。
- 期間：2020-01-01〜2023-12-31（連続期間 interAnnualType=1）。時別値（aggrgPeriod=9）。Energies §3.11 の「JMA 2020–2023」に合わせた。
- 要求の項目（全文は fetch_log.json の `fields`）：
  `stationNumList=["s<番号>"]`, `aggrgPeriod=9`, `elementNumList=[["201",""],["610",""]]`, `interAnnualType=1`, `ymdList=["2020","2023","1","12","1","31"]`, `option={"list":["obs"],"nyear":0}`, `downloadFlag=true`, `rmkFlag=1`, `disconnectFlag=1`, `youbiFlag=0`, `fukenFlag=0`, `kijiFlag=0`, `csvFlag=1`, `jikantaiFlag=0`, `jikantaiList=[]`, `ymdLiteral=1`。
- 取得の手順：`.venv/bin/python data/raw/jma/fetch_jma.py`（既存のファイルは `--force` なしでは上書きしない）。要求・HTTP 状態・サイズ・sha256 は fetch_log.json。
- ファイルは返されたまま（Shift_JIS／cp932、CRLF）保存し、変更していない。1行目にダウンロード時刻が入るので、取り直すとファイルの中身（ハッシュ）は変わる。
- 原ファイル（5つの CSV）はこのリポジトリに含めていない。`fetch_jma.py` で同じ要求を送って取得し、`data/preprocess_jma.py` で加工する。加工した値（data/parameters_v2/temperature_96h_jma.csv、pv_cf_96h_jma.csv、jma_summary.csv）はこのリポジトリにある。

## ファイルの形式
- 1行目：ダウンロードした時刻。3行目：地点名。4行目：見出し（年月日時、気温、気温、気温、日射量、日射量、日射量）。5行目：品質情報・均質番号の見出し。6行目以降：データ。
- 時刻は `2020/1/1 1:00:00`〜`2024/1/1 00:00:00`（気象庁の24時は翌日の00:00:00 と表示）。各ファイル 35,064 行。
- 品質情報 8（正常）が気温で 35,060〜35,064 件、日射量で 35,060〜35,063 件。均質番号はすべて 1（期間内に観測環境の変更なし）。

## 加工（data/preprocess_jma.py）
- 代表時間の1時間 h は、その日の h:00〜h+1:00（JST）。気温は h:00 の値（00:00:00 の行はその日の0時）、日射量は h+1:00 の値（(h, h+1] の積算）を割り当てる。MJ/㎡ ÷ 3.6 = kWh/㎡。
- 日付が 2020-01-01〜2023-12-31 の値だけを使い、品質情報 8（正常）と 5（準正常）だけを使う（4 以下は除く）。
- 季節（DJF・MAM・JJA・SON）×24時の平均。重み＝季節の日数（90/92/92/91、合計 8,760 時間、v1 と同じ）。
- 出力：data/parameters_v2/temperature_96h_jma.csv、pv_cf_96h_jma.csv、jma_summary.csv。

## 利用条件と出典の記載
- 気象庁ホームページのコンテンツは「公共データ利用規約（第1.0版）」に準拠した条件で利用できる（https://www.jma.go.jp/jma/kishou/info/coment.html、2026-11-06 確認）。出典の記載が必要で、加工した場合はその旨を書く。
- 出典の書き方：「出典：気象庁ホームページ（https://www.data.jma.go.jp/risk/obsdl/）の時別値（2020〜2023年、札幌・仙台・東京・大阪・福岡の気温・全天日射量）を加工して作成」。
- サービスの注意事項（一回の要求の上限、自動化ツールによる過度のアクセスを控えること）に従い、地点ごとに1回だけ要求した。

## 代替手段
- 代替手段1（過去の気象データ検索の時別ページ）・代替手段2（平年値）は使っていない（ダウンロードの機能で取得できたため）。
