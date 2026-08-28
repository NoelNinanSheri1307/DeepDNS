import os
import sys
import json
import gzip
import math
import hashlib
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

os.makedirs('reports/dataset_audit', exist_ok=True)
os.makedirs('scripts', exist_ok=True)

print("Starting DeepDNS Full Forensic Audit...")

# ==========================================
# 1. INVENTORY & DISCOVERY
# ==========================================
raw_dir = os.path.join('data', 'raw')
files_info = []

for root, dirs, files in os.walk(raw_dir):
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, '.').replace('\\', '/')
        ext = os.path.splitext(f)[1].lower()
        is_gz = f.endswith('.gz')
        file_size = os.path.getsize(full_path)
        
        parts = rel_path.split('/')
        source = 'cic_bell_dns_exf_2021' if 'cic_bell_dns_exf_2021' in parts else ('dns_threats' if 'dns_threats' in parts else 'other')
        
        is_attack = 'attack' in rel_path.lower() and 'benign' not in f.lower()
        if 'benign' in f.lower():
            is_attack = False
        
        is_light = 'light' in rel_path.lower() or 'light' in f.lower()
        is_heavy = 'heavy' in rel_path.lower() or 'heavy' in f.lower()
        if is_light:
            intensity = 'light'
        elif is_heavy:
            intensity = 'heavy'
        else:
            intensity = 'standard_benign'

        if 'stateful' in f.lower():
            state_type = 'stateful'
        elif 'stateless' in f.lower():
            state_type = 'stateless'
        else:
            state_type = 'n/a'

        modality = 'benign'
        if is_attack:
            for m in ['audio', 'compressed', 'exe', 'image', 'text', 'video']:
                if m in f.lower():
                    modality = m
                    break

        row_count = 0
        if is_gz:
            with gzip.open(full_path, 'rt', encoding='utf-8', errors='ignore') as gf:
                for _ in gf:
                    row_count += 1
            row_count = max(0, row_count - 1)
        elif ext == '.csv':
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as cf:
                for _ in cf:
                    row_count += 1
            row_count = max(0, row_count - 1)

        files_info.append({
            'filename': f,
            'rel_path': rel_path,
            'full_path': os.path.abspath(full_path),
            'extension': ext,
            'compressed': is_gz,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 4),
            'row_count': row_count,
            'dataset_source': source,
            'is_attack': is_attack,
            'modality': modality,
            'intensity': intensity,
            'state_type': state_type
        })

inv_df = pd.DataFrame(files_info)
inv_df.to_csv('reports/dataset_audit/file_inventory.csv', index=False)
with open('reports/dataset_audit/file_inventory.json', 'w', encoding='utf-8') as f:
    json.dump(files_info, f, indent=2)

print(f"[Phase 1] Inventoried {len(files_info)} files.")

# ==========================================
# 2. CIC-BELL DEEP SCHEMA & PAIR ANALYSIS
# ==========================================
cic_files = [f for f in files_info if f['dataset_source'] == 'cic_bell_dns_exf_2021']

# Group by pairs
pairs = defaultdict(dict)
for finfo in cic_files:
    fpath = finfo['rel_path']
    fname = finfo['filename']
    base_key = fname.replace('stateful_features-', '').replace('stateless_features-', '')
    dir_key = os.path.dirname(fpath)
    pair_id = f"{dir_key}/{base_key}"
    pairs[pair_id][finfo['state_type']] = finfo

pair_records = []
stateless_master_list = []
stateful_master_list = []

# Collect all column stats
stateless_cols_set = set()
stateful_cols_set = set()

for pair_id, pfiles in pairs.items():
    if 'stateless' in pfiles and 'stateful' in pfiles:
        sl_info = pfiles['stateless']
        sf_info = pfiles['stateful']
        
        sl_df = pd.read_csv(sl_info['rel_path'], low_memory=False)
        sf_df = pd.read_csv(sf_info['rel_path'], low_memory=False)
        
        stateless_cols_set.update(sl_df.columns)
        stateful_cols_set.update(sf_df.columns)
        
        # Add labels and metadata
        sl_df['src_file'] = sl_info['filename']
        sl_df['pair_id'] = pair_id
        sl_df['label_binary'] = 1 if sl_info['is_attack'] else 0
        sl_df['label_modality'] = sl_info['modality']
        sl_df['intensity'] = sl_info['intensity']
        
        sf_df['src_file'] = sf_info['filename']
        sf_df['pair_id'] = pair_id
        sf_df['label_binary'] = 1 if sf_info['is_attack'] else 0
        sf_df['label_modality'] = sf_info['modality']
        sf_df['intensity'] = sf_info['intensity']
        
        # Check timestamp
        ts_monotonic = False
        duration_sec = 0.0
        min_ts = None
        max_ts = None
        if 'timestamp' in sl_df.columns:
            ts_dt = pd.to_datetime(sl_df['timestamp'], errors='coerce')
            ts_monotonic = bool(ts_dt.is_monotonic_increasing)
            min_ts = str(ts_dt.min())
            max_ts = str(ts_dt.max())
            if ts_dt.notnull().sum() > 1:
                duration_sec = float((ts_dt.max() - ts_dt.min()).total_seconds())
                # Add dt
                sl_df['timestamp_dt'] = ts_dt
                sl_df['inter_arrival_sec'] = ts_dt.diff().dt.total_seconds().fillna(0)
            else:
                sl_df['inter_arrival_sec'] = 0.0

        pair_records.append({
            'pair_id': pair_id,
            'category': 'Attack' if sl_info['is_attack'] else 'Benign',
            'intensity': sl_info['intensity'],
            'modality': sl_info['modality'],
            'stateless_rows': len(sl_df),
            'stateful_rows': len(sf_df),
            'row_match': (len(sl_df) == len(sf_df)),
            'row_diff': len(sl_df) - len(sf_df),
            'stateless_file': sl_info['filename'],
            'stateful_file': sf_info['filename'],
            'ts_monotonic': ts_monotonic,
            'start_time': min_ts,
            'end_time': max_ts,
            'duration_sec': duration_sec,
            'queries_per_sec': round(len(sl_df) / duration_sec, 2) if duration_sec > 0 else 0
        })
        
        stateless_master_list.append(sl_df)
        stateful_master_list.append(sf_df)

pair_df = pd.DataFrame(pair_records)
all_sl_df = pd.concat(stateless_master_list, ignore_index=True)
all_sf_df = pd.concat(stateful_master_list, ignore_index=True)

print(f"[Phase 2/3] Master Stateless rows: {len(all_sl_df)}, Master Stateful rows: {len(all_sf_df)}")

# Stateful columns evaluation: check set formatting and data types
stateful_type_anomalies = {}
for c in all_sf_df.columns:
    if c in ['src_file', 'pair_id', 'label_binary', 'label_modality', 'intensity']:
        continue
    sample_val = str(all_sf_df[c].iloc[0])
    is_set_or_list = any(s in sample_val for s in ['{', '}', '[', ']', 'set()'])
    stateful_type_anomalies[c] = {
        'dtype': str(all_sf_df[c].dtype),
        'sample': sample_val,
        'contains_python_structures': is_set_or_list,
        'unique_values_count': int(all_sf_df[c].nunique(dropna=False))
    }

# ==========================================
# 3. LABEL ANALYSIS
# ==========================================
print("\n--- PHASE 4: Label Analysis ---")
label_summary = {
    'total_records': len(all_sl_df),
    'binary_counts': {str(k): int(v) for k, v in all_sl_df['label_binary'].value_counts().items()},
    'binary_proportions': {str(k): round(float(v / len(all_sl_df) * 100), 2) for k, v in all_sl_df['label_binary'].value_counts().items()},
    'intensity_counts': {str(k): int(v) for k, v in all_sl_df['intensity'].value_counts().items()},
    'modality_counts': {str(k): int(v) for k, v in all_sl_df['label_modality'].value_counts().items()},
    'attack_vs_benign_by_intensity': all_sl_df.groupby(['intensity', 'label_binary']).size().unstack(fill_value=0).to_dict(),
    'modality_by_intensity': all_sl_df.groupby(['intensity', 'label_modality']).size().unstack(fill_value=0).to_dict(),
}

# ==========================================
# 4. DUPLICATE ANALYSIS
# ==========================================
print("\n--- PHASE 5: Duplicate Analysis ---")
# Exact duplicates in stateless
stateless_feature_cols = [c for c in all_sl_df.columns if c not in ['timestamp', 'timestamp_dt', 'inter_arrival_sec', 'src_file', 'pair_id', 'label_binary', 'label_modality', 'intensity']]
exact_sl_dups = int(all_sl_df.duplicated(subset=stateless_feature_cols).sum())
exact_sl_row_dups = int(all_sl_df.duplicated(subset=['timestamp'] + stateless_feature_cols).sum())

# Duplicates within attack vs benign
attack_sl_df = all_sl_df[all_sl_df['label_binary'] == 1]
benign_sl_df = all_sl_df[all_sl_df['label_binary'] == 0]

attack_hashes = set(attack_sl_df[stateless_feature_cols].apply(lambda row: tuple(row), axis=1))
benign_hashes = set(benign_sl_df[stateless_feature_cols].apply(lambda row: tuple(row), axis=1))
cross_class_dups = len(attack_hashes.intersection(benign_hashes))

# ==========================================
# 5. DATA LEAKAGE & FEATURE RISK
# ==========================================
print("\n--- PHASE 6: Data Leakage Analysis ---")
feature_risks = []

# Evaluate Stateless features
for c in stateless_feature_cols:
    ser = all_sl_df[c]
    # Clean numeric if needed
    ser_num = pd.to_numeric(ser, errors='coerce')
    non_null_pct = ser_num.notnull().mean()
    
    risk_cat = "SAFE"
    reason = "Standard statistical/lexical property."
    evidence = ""
    corr = None
    
    if non_null_pct > 0.8:
        corr = float(ser_num.corr(all_sl_df['label_binary']))
        if abs(corr) > 0.85:
            risk_cat = "SUSPICIOUS"
            reason = f"High linear correlation with label (r={corr:.3f})."
        elif abs(corr) > 0.95:
            risk_cat = "LEAKAGE-PRONE"
            reason = f"Extreme correlation with label (r={corr:.3f}), potential target leakage."
    else:
        # Categorical check
        u_att = set(attack_sl_df[c].astype(str).unique())
        u_ben = set(benign_sl_df[c].astype(str).unique())
        overlap = len(u_att.intersection(u_ben))
        if overlap == 0 and len(u_att) > 0 and len(u_ben) > 0:
            risk_cat = "LEAKAGE-PRONE"
            reason = "Values are completely disjoint between attack and benign."
            evidence = f"Attack unique: {len(u_att)}, Benign unique: {len(u_ben)}, Overlap: 0"

    # Check sld and longest_word
    if c == 'sld':
        # Inspect what sld is
        sld_sample = list(all_sl_df['sld'].dropna().unique()[:5])
        evidence = f"Sample sld values: {sld_sample}"
        if all_sl_df['sld'].nunique() < 50:
            risk_cat = "SUSPICIOUS"
            reason = "sld has very low cardinality and may encode specific lab domains."
    
    feature_risks.append({
        'feature': c,
        'feature_type': 'stateless',
        'dtype': str(ser.dtype),
        'correlation_with_label': round(corr, 4) if corr is not None and not math.isnan(corr) else 'N/A',
        'unique_values': int(ser.nunique()),
        'risk_category': risk_cat,
        'reason': reason,
        'evidence': evidence,
        'recommended_action': "Normalize and include in behavioral/lexical feature vector" if risk_cat == "SAFE" else "Inspect domain distribution; consider anonymizing or excluding if synthetic artifact"
    })

# Evaluate Stateful features
stateful_raw_cols = [c for c in all_sf_df.columns if c not in ['src_file', 'pair_id', 'label_binary', 'label_modality', 'intensity']]
for c in stateful_raw_cols:
    anom = stateful_type_anomalies[c]
    risk_cat = "LEAKAGE-PRONE" if anom['contains_python_structures'] else "SUSPICIOUS"
    reason = "Stateful feature aggregated across session/window. Contains raw Python set/list string dumps and global session aggregates."
    action = "EXCLUDE from causal online models unless causally recomputed in dynamic window"
    if not anom['contains_python_structures']:
        ser_num = pd.to_numeric(all_sf_df[c], errors='coerce')
        if ser_num.notnull().mean() > 0.8:
            corr = float(ser_num.corr(all_sf_df['label_binary']))
            reason = f"Windowed/stateful feature with correlation r={corr:.3f}. Risk of non-causal aggregation."
    
    feature_risks.append({
        'feature': c,
        'feature_type': 'stateful',
        'dtype': anom['dtype'],
        'correlation_with_label': 'N/A' if anom['contains_python_structures'] else (round(corr, 4) if 'corr' in locals() and corr is not None and not math.isnan(corr) else 'N/A'),
        'unique_values': anom['unique_values_count'],
        'risk_category': risk_cat,
        'reason': reason,
        'evidence': f"Python data structures: {anom['contains_python_structures']}, sample: {anom['sample'][:50]}",
        'recommended_action': action
    })

risk_df = pd.DataFrame(feature_risks)
risk_df.to_csv('reports/dataset_audit/feature_risk_matrix.csv', index=False)

# ==========================================
# 6. LOW-AND-SLOW COMPARISON
# ==========================================
print("\n--- PHASE 8: Low-and-Slow Analysis ---")
low_slow_stats = []
for intensity_val in ['light', 'heavy', 'standard_benign']:
    sub_df = all_sl_df[all_sl_df['intensity'] == intensity_val]
    if len(sub_df) == 0:
        continue
    
    low_slow_stats.append({
        'intensity': intensity_val,
        'row_count': len(sub_df),
        'mean_inter_arrival_sec': float(sub_df['inter_arrival_sec'].mean()),
        'median_inter_arrival_sec': float(sub_df['inter_arrival_sec'].median()),
        'std_inter_arrival_sec': float(sub_df['inter_arrival_sec'].std()),
        'p95_inter_arrival_sec': float(sub_df['inter_arrival_sec'].quantile(0.95)),
        'mean_fqdn_length': float(pd.to_numeric(sub_df['FQDN_count'], errors='coerce').mean()),
        'mean_entropy': float(pd.to_numeric(sub_df['entropy'], errors='coerce').mean()),
        'mean_subdomain_length': float(pd.to_numeric(sub_df['subdomain_length'], errors='coerce').mean()),
        'mean_numeric': float(pd.to_numeric(sub_df['numeric'], errors='coerce').mean()),
    })

# ==========================================
# 7. DNS THREATS DATASET ANALYSIS
# ==========================================
print("\n--- PHASE 9: DNS Threats Dataset Analysis ---")
dns_threats_train_path = 'data/raw/dns_threats/original/train_combined_multiclass.csv.gz'
dns_threats_test_path = 'data/raw/dns_threats/original/test_combined_multiclass.csv.gz'

dt_train = pd.read_csv(dns_threats_train_path, compression='gzip')
dt_test = pd.read_csv(dns_threats_test_path, compression='gzip')

dt_train_classes = {str(k): int(v) for k, v in dt_train['class'].value_counts().items()}
dt_test_classes = {str(k): int(v) for k, v in dt_test['class'].value_counts().items()}

dt_train_domains = set(dt_train['domain'].dropna().astype(str))
dt_test_domains = set(dt_test['domain'].dropna().astype(str))
domain_overlap = len(dt_train_domains.intersection(dt_test_domains))

dt_train_lens = dt_train['domain'].dropna().astype(str).str.len()
dt_test_lens = dt_test['domain'].dropna().astype(str).str.len()

dns_threats_summary = {
    'train_rows': len(dt_train),
    'test_rows': len(dt_test),
    'train_columns': list(dt_train.columns),
    'test_columns': list(dt_test.columns),
    'train_classes': dt_train_classes,
    'test_classes': dt_test_classes,
    'unique_train_domains': len(dt_train_domains),
    'unique_test_domains': len(dt_test_domains),
    'train_test_domain_overlap': domain_overlap,
    'train_test_leakage_pct': round(domain_overlap / len(dt_test_domains) * 100, 2) if len(dt_test_domains) > 0 else 0,
    'domain_length_stats_train': {
        'min': int(dt_train_lens.min()),
        'mean': round(float(dt_train_lens.mean()), 2),
        'median': float(dt_train_lens.median()),
        'max': int(dt_train_lens.max())
    },
    'domain_length_stats_test': {
        'min': int(dt_test_lens.min()),
        'mean': round(float(dt_test_lens.mean()), 2),
        'median': float(dt_test_lens.median()),
        'max': int(dt_test_lens.max())
    }
}

# Save computed numerical summaries for markdown generation
audit_results = {
    'pair_records': pair_records,
    'label_summary': label_summary,
    'duplicate_analysis': {
        'exact_stateless_duplicates': exact_sl_dups,
        'exact_stateless_row_duplicates_with_ts': exact_sl_row_dups,
        'cross_class_duplicates': cross_class_dups,
        'total_stateless_rows': len(all_sl_df),
        'duplicate_percentage': round(exact_sl_dups / len(all_sl_df) * 100, 2)
    },
    'low_slow_stats': low_slow_stats,
    'dns_threats_summary': dns_threats_summary,
    'stateless_type_anomalies': {c: str(all_sl_df[c].dtype) for c in stateless_feature_cols},
    'stateful_type_anomalies': stateful_type_anomalies
}

with open('reports/dataset_audit/audit_numerical_results.json', 'w', encoding='utf-8') as f:
    json.dump(audit_results, f, indent=2)

print("\nAudit Calculations Complete! Saved audit_numerical_results.json.")
