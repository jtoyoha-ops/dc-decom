# 固定（data/parameters_v2）

## 規則
- この版の data/parameters_v2/ は、v2 の計算（outputs/stage3/、outputs/abc/）より前に固定した。
- 固定まで、候補の v2 の値で LP を解いていない（シナリオの総額、C-CN と R-CN の差は計算していない）。実行したのは、試験用の仮の値（tests/fixtures/params_dev、status 'dev'）での試験と、調査値の幅を目標にする PUE の較正（data/calibrate_pue_v2.py、LP なし）だけ。
- どの値も、Energies の報告値の範囲（18〜23% など）や、モデルの結果への影響を見て選んでいない。
- 固定の後に値を変えていない。感度・不確実性の分析の値は、設定ファイル（config/）の上書きとしてメモリの中で変える。

## 固定に含むもの
- data/raw/jma/（気象庁の時別値5ファイル、fetch_jma.py、fetch_log.json、RETRIEVAL.md）、data/raw/egc/（電力取引報 FY2025 の12ファイル）、data/raw/moe/（電気事業者別排出係数 R6年度実績）。生データのファイルはこのリポジトリに含めていない（取得の方法と sha256 は各フォルダの README.md・RETRIEVAL.md）。
- data/preprocess_jma.py、data/calibrate_pue_v2.py、data/build_parameters_v2.py
- data/parameters_v2/（parameters.csv、regions.csv、paths.csv、sources.csv、SOURCES.md、temperature_96h_jma.csv、pv_cf_96h_jma.csv、jma_summary.csv、pue_calibration.json、tariff_fy2025.csv、ef_fy2024.csv、FREEZE.md）
- tests/v2/test_parameters_v2.py

## 数（status 別）
| ファイル | sourced | derived | paper | assumption |
|---|---|---|---|---|
| parameters.csv（50行） | 7 | 12 | 14 | 17 |
| regions.csv（38行） | 10 | 5 | 15 | 8 |
| paths.csv（153行） | 4 | 95 | 18 | 36 |

## 仮定の一覧（出典のない値。幅は感度に使う）
| id | 地域・年 | 値 | 低 | 高 | 単位 |
|---|---|---|---|---|---|
| pue_immersion_offset | — | 0.025 | 0 | 0.04 | — |
| utilization | — | 0.8 | 0.6 | 0.9 | 比率 |
| s_inf_tol | — | 0.3 | 0.1 | 0.5 | 推論のうちの比率 |
| cooling_air_capex | — | 150 | 100 | 200 | 千円/kW |
| bess_duration_h | — | 4 | 2 | 6 | h |
| pv_performance_ratio | — | 0.8 | 0.75 | 0.85 | 比率 |
| cgs_fuel_yen_per_kWh_th | — | 8 | 5 | 11 | 円/kWh（熱量） |
| gen_capex | — | 80 | 40 | 150 | 千円/kW |
| opex_om_rate | — | 0.02 | 0.01 | 0.04 | CAPEX に対する年の比率 |
| ppa_tier1_share | — | 0.05 | 0.02 | 0.1 | ポテンシャルの電力量に対する比率 |
| ppa_tier2_share | — | 0.1 | 0.05 | 0.3 | 同上 |
| ppa_tier2_premium_yen_per_kWh | — | 3 | 1 | 6 | 円/kWh |
| operator_burden_share（θ） | — | 0.5 | 0 | 1 | 比率 |
| reinforcement_year（T_g） | — | 2030 | 2028 | 2035 | 年 |
| ppa_floor_subsidized（ψ） | — | 0.3 | 0.1 | 0.5 | 比率 |
| ppa_tier_multiplier_induced | — | 1.5 | 1.2 | 2 | 倍率 |
| carbon_pass_through | — | 1 | 0 | 1 | 比率 |
| conn_capacity_GW_per_yr（c0） | Hokkaido・Tohoku・Kyushu | 0.2 | 0.05 | 0.6 | GW/年 |
| conn_capacity_reinforced_GW_per_yr（c1） | Hokkaido・Tohoku・Kyushu | 0.6 | 0.3 | 1.2 | GW/年 |
| shell_premium_yen_per_kW | Kanto・Kansai | 0 | 0 | 100,000 | 円/kW |
| ai_share | 2031–2040 | 0.385→0.665 | 0.231→0.399 | 0.577→0.9 | DC の電力に対する比率 |
| carbon_price | 2036–2040 | 12,525→18,623 | 8,017→12,087 | 17,032→25,160 | 円/t-CO₂ |
| demand_facility_TWh | 2035–2040 | 65.75→87.0 | 28.17→32.33 | 70.7→94.2 | TWh（施設） |
| ef_national | 2025–2039（直線の補間） | 0.399→0.059 | 0.397→0.026 | 0.405→0.148 | kg-CO₂/kWh |

このほか、出典のある値に付けた幅のうち次は仮定：ppa_yen_per_kWh の ±2 円/kWh、ai_share_2024 の 0.10〜0.25、s_train の 0.2〜0.6、discount_rate の 2〜4%、siting_subsidy_yen_per_kW の低い側 100,000、reinforcement_cost_yen_per_kW の低い側 50,000、liquid_deploy_cap の遅い経路 0.25、需要の低い経路の 2031年以降。

## 固定前の判断（理由は SOURCES.md）
- PUE：時間別評価の再較正。α・β ≥ 0 だけでは冬の代表時間で PUE < 1 になるので、PUE ≥ 1 を全代表時間に課した当てはめを基準にした（指定どおりの当てはめも pue_calibration.json の bounds_only に記録）。
- 需要：Wood Mackenzie の 19 TWh は施設の電力量。IT 電力量は 2024年の全空冷の PUE（1.462794）で割って作る。
- 排出係数：温対法の規則に従い地域別（旧一般電気事業者の残差の基礎排出係数、R6年度実績）。全国は感度。2040年の全国値 0.035 は第7次エネルギー基本計画の関連資料のシナリオの中央値。
- PPA：年単位の一致（温対法のマーケット基準）。段の量と上乗せは仮定。
- N＝1（式どおり）。感度 N＝2、4、v1 の 5%。
- λ_B(t)＝0.82×[nonAI＋AI(1−s_train)(1−s_inf_tol)]、κ＝λ×60/82、λ_A＝0.82。
