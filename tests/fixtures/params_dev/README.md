# tests/fixtures/params_dev — 試験用の仮の構造パラメータ（DEV）

- DC-DECOM v2（`src/dc_decom_v2/`）の新しい構造パラメータに、試験と煙試験のためだけに置いた仮の値。すべて status `dev`。**結果には使わない。** 出典はなく、v2 の値の候補でもない（v2 の値は出典つきで `data/parameters_v2/` に置き、固定した）。
- 既存のパラメータ（PUE 係数、料金、PPA 価格、需要の折れ線など）は `data/parameters_spec/`（論文の値）から取り、`ParamSet.compose(existing_from=parameters_spec, structural_from=params_dev)` で組み合わせる。結果の `meta["uses_dev_params"]` が True になる。
- ファイルの形式は `src/dc_decom_v2/params.py` の冒頭の説明（v2 layout）と同じ。
  - parameters.csv：蓄電池の時間（4 h）、非常用発電機の CAPEX、PPA の段（第1段・第2段の量の比率と第2段の上乗せ）、非化石証書の価格、増強費、事業者負担率 θ、立地補助（円/kW）。
  - regions.csv：非首都圏の接続容量（GW/年）、地域の建設の割増（円/kW）。
  - paths.csv、pv_cf_96h.csv：`make_dev_profiles.py` が作る（首都圏比率の下限 λ、関東の下限 κ、PPA の量の年の係数、季節ごとの半正弦の PV 設備利用率。年平均は 13〜16%）。
