import pandas as pd

date_str = '2026-01-08'
target_ticker = 'AAPL'
target_id = 15
target_time = '2026-01-08 14:31:00+00:00'

print(f"=== Alignment Verification for {target_ticker} (symbol_id={target_id}) at {target_time} ===\n")

# 1. P0 Label
p0_path = rf'D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical\date={date_str}\labels.parquet'
df_p0 = pd.read_parquet(p0_path)
df_p0['decision_time'] = df_p0['decision_time'].astype(str)
row_p0 = df_p0[(df_p0['symbol_id'] == target_id) & (df_p0['decision_time'] == target_time)]
print("[1. P0 DATA] labels.parquet")
print(row_p0[['decision_time', 'symbol_id', 'label_entry_price', 'label_30m']].to_string(index=False))

# 2. P2 Theme Returns
p2_returns_path = rf'C:\GFF_Cache\p2_alpha_lab\theme_returns\date={date_str}\layer_id=9\scale=30m\theme_returns.parquet'
try:
    df_p2r = pd.read_parquet(p2_returns_path)
    df_p2r['decision_time'] = df_p2r['decision_time'].astype(str)
    row_p2r = df_p2r[(df_p2r['member_id'] == target_id) & (df_p2r['decision_time'] == target_time)]
    print("\n[2. P2 DATA] theme_returns.parquet (member_id perfectly maps to symbol_id)")
    print(row_p2r[['decision_time', 'member_id', 'theme_id', 'theme_return', 'member_return', 'theme_excess_return']].head(1).to_string(index=False))
except Exception as e:
    print(f"\n[2. P2 DATA] theme_returns.parquet error: {e}")

# 3. P2 Daily Features
p2_feat_path = rf'C:\GFF_Cache\p2_alpha_lab\daily_relation_features\date={date_str}\layer_id=9\scale=30m\daily_relation_features.parquet'
try:
    df_p2f = pd.read_parquet(p2_feat_path)
    df_p2f['decision_time'] = df_p2f['decision_time'].astype(str)
    row_p2f = df_p2f[(df_p2f['symbol_id'] == target_id) & (df_p2f['decision_time'] == target_time)]
    print("\n[3. P2 DATA] daily_relation_features.parquet (Final output aligned by symbol_id)")
    print(row_p2f[['decision_time', 'symbol_id', 'layer_id', 'past_theme_return_15m', 'past_neighbor_spillover_15m']].head(1).to_string(index=False))
except Exception as e:
    print(f"\n[3. P2 DATA] daily_relation_features.parquet error: {e}")

