# data/parameters_spec — 仕様版の入力

Energies 19(10) 2485（2026）の §3.2、表6・表8・表9 に合わせた入力。
仕様版のパッケージ `src/dc_decom_spec/` が `SpecFixes.table9_inputs=True`（修正 E6）のときに読む。
`False` のときは v1 のパッケージの入力 `data/parameters_csv/` を読む。v2.0.0 で `data/parameters_csv/regions.csv` を修正 E6 に合わせて直した（下の表の「v1.0.1」の列は直す前の値で、タグ v1.0.1 にある）。

この版は仕様版の計算より前に固定した。
値は論文の表と本文から写しただけで、Energies の報告値（6.2/8.5/6.8 兆円、18〜23% など）に合わせて選んだ値はない。

## v1.0.1 の data/parameters_csv/ との違い（すべて）

### regions.csv
| 地域 | 列 | v1.0.1（parameters_csv） | 仕様版 | 論文の根拠 |
|---|---|---|---|---|
| Hokkaido | re_max_GW | 337 | 337.471 | §3.2「337,471 MW in Hokkaido」（表9 は 337 GW） |
| Tohoku | re_max_GW | 180 | 142 | 表9 |
| Kanto | re_max_GW | 98.8 | 98.833 | §3.2「98,833 MW in Kanto」（表9 は 99 GW） |
| Kansai | re_max_GW | 75 | 73 | 表9 |
| Kyushu | re_max_GW | 140 | 115 | 表9 |
| Kyushu | ppa_yen_per_kWh | 14.0 | 13.0 | 表9 |
| 全地域 | grid_ef_base（新しい列） | なし | 0.49 / 0.52 / 0.44 / 0.36 / 0.37（北海道・東北・関東・関西・九州） | 表9（地域別の系統排出係数、MOE [59]） |
| 全地域 | paper_ref（新しい列） | なし | 各行の値の根拠 | — |

気温・基準年の比率・系統料金・PPA 価格（九州以外）は v1 と同じで、表9 と一致する。
東北の系統料金 18.5 円/kWh は表9 にあり、§3.6 の本文には出てこない。

### global_parameters.csv
- 値は v1 と同じ（42行）。列 paper_ref を加えた。
- 論文に値がなくコードだけが持つ値（paper_ref が `code-only`）：P_base_2024（2.85 GW）、EF_grid_2040（0.10）、P_carbon_2040（18,000 円/t）、cooling_air_capex（150 千円/kW）、opex_om_rate（2%）。
- 論文の幅の中からコードが1点を選んだ値：P_carbon_2026（1,700〜4,300 → 3,000）、P_carbon_2035（7,000〜15,000 → 11,000）、cooling_dlc_capex（+20〜30% → +30%）、cooling_immersion_capex（+40〜60% → +50%）、pv_cf_avg（13〜17% → 15%）。
- LP が使わない値：g_AI、eta_HW_rate、eta_arch_rate、rep_hours_per_year、cgs_eff_total、pue_air_2024_avg、pue_dlc_avg、pue_immersion_avg（需要は model.py の Wood Mackenzie の折れ線 19→70 TWh を使い、PUE は式8・9 を年平均気温で評価する）。

### temperature_96h.csv
- 値は v1 と同じ（480行）。列 paper_ref を加えた。
- この表は年平均気温に季節（−12/0/+12/0℃）と日変化（±5℃）を足した合成の値で、§3.11 の「気象庁 2020〜2023 年の時別値」ではない。v1 と仕様版の LP はどちらも PUE を regions.csv の年平均気温で評価するので、この表は結果に影響しない。

## CSV にない、コードの中だけの値（仕様版では SpecReadings で読み方を切り替える）
| 値 | v1.0.1 の場所 | 論文 | 仕様版の主な読み方（primary） | 対案（paper_only など） |
|---|---|---|---|---|
| PPA の比率の上限（2024年5%→2040年に北海道50%、東北・九州45%、関東・関西20%） | model.py l.291-305 | なし | コードどおり | 上限なし（式15 は残る） |
| 自家 PV ≤ 施設電力の10% | model.py l.273-280 | なし | コードどおり | なし |
| CGS の電力量 ≤ 施設電力の5% | model.py l.308-310 | なし | コードどおり | なし |
| CGS の設備利用率 0.5 | model.py l.264-266 | なし | コードどおり | 物理上限（1.0） |
| 蓄電池 250 サイクル/年 | model.py l.255-260 | なし | コードどおり | 上限なし |
| 稼働率 0.80 を平均 IT 電力に掛ける | model.py l.230 | 表8 の P_AI,0 の注記「19 TWh と稼働率80%から推定」だけ | コードどおり | 1.00 |
| 首都圏の立地割増 1.10 | model.py l.363 | 表10 の「立地関連の費用」（式の係数としては書かれていない） | コードどおり | 1.00 |
| C-CN の立地の規則 | scenarios.py | §3.11「REF と C-CN は同じ比率」、表7「新しい大型 DC は関東・関西のみ」 | 同じ比率 | 関東・関西のみ（非首都圏は2025年の水準で固定） |
| 炭素価格の点 | global_parameters.csv | 表8 の幅 | 中央値 | 下限（1,700/7,000）・上限（4,300/15,000） |
| CGS の燃料費 8 円/kWh（熱量） | model.py l.394-397 | なし | コードどおり | （変えない） |
| 需要の折れ線（19→70 TWh） | model.py l.82-110 | §4.1 の Wood Mackenzie の幅 | コードどおり | （感度で AI 分に係数を掛ける） |
