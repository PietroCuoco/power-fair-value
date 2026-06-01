# Data QA Report

## Index integrity
- Rows: 29,857  |  Monotonic: True  |  Duplicates: 0
- Expected hours: 29,857  |  Missing: 0

## DST transitions (expected, not errors)
- 23-hour days: 4 (e.g. ['2023-03-26', '2024-03-31', '2025-03-30', '2026-03-29'])
- 25-hour days: 3 (e.g. ['2023-10-29', '2024-10-27', '2025-10-26'])

## Price diagnostics
- Negative-price hours: 1562 (5.232%), min -500.0 EUR/MWh (preserved)
- Spikes (|robust z| > 6.0): 109 flagged, top |values| [936.28, 820.11, 818.98, 805.08, 674.18] (preserved)

## Cross-series consistency
- |residual_load - (load - wind - solar)| median 0.0 MW, p95 0.0 MW

## Gap handling
- Short gaps (<= 3h) time-interpolated.
- Values filled: {'price_da': 0, 'load_actual': 0, 'residual_load_actual': 0, 'gen_wind_onshore_actual': 0, 'gen_wind_offshore_actual': 0, 'gen_pv_actual': 0, 'fc_gen_total': 0, 'fc_gen_wind_pv': 0, 'fc_gen_wind_onshore': 0, 'fc_gen_wind_offshore': 0, 'fc_gen_pv': 0, 'fc_load_total': 0}
- Remaining NaNs: {'price_da': 0, 'load_actual': 0, 'residual_load_actual': 0, 'gen_wind_onshore_actual': 0, 'gen_wind_offshore_actual': 0, 'gen_pv_actual': 0, 'fc_gen_total': 0, 'fc_gen_wind_pv': 0, 'fc_gen_wind_onshore': 0, 'fc_gen_wind_offshore': 0, 'fc_gen_pv': 0, 'fc_load_total': 0}
