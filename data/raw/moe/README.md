# data/raw/moe — 電気事業者別排出係数（取得の記録）

- r08_denki_coefficient_rev4.xlsx：環境省・経済産業省「電気事業者別排出係数（特定排出者の温室効果ガス排出量算定用）－R6年度実績－」（令和8年提出用、R8.1.9 公表、R8.8.3 一部追加・更新）。
  - URL：https://policies.env.go.jp/earth/ghg-santeikohyo/files/calc/r08_denki_coefficient_rev4.xlsx（一覧のページ https://policies.env.go.jp/earth/ghg-santeikohyo/calc.html）
  - 取得日：2026-11-06。sha256 先頭16桁 a2d509a2b53eeb54。原ファイルはこのリポジトリに含めていない（上の URL から取得してこのフォルダに置くと data/build_parameters_v2.py が読む）。
- 使った行（data/build_parameters_v2.py が読む）：北海道電力・東北電力・東京電力エナジーパートナー・関西電力・九州電力の「メニュー…(残差)」と「(参考値)事業者全体」の基礎排出係数、「全国平均係数」0.000423 t-CO₂/kWh、「代替値」0.000416。
