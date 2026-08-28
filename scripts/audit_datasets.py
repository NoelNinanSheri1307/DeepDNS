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

print("Starting deep dataset forensic inspection...")

def audit_filesystem():
    print("\n--- PHASE 1: Filesystem Inventory ---")
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

            # Extract modality if attack
            modality = 'benign'
            if is_attack:
                for m in ['audio', 'compressed', 'exe', 'image', 'text', 'video']:
                    if m in f.lower():
                        modality = m
                        break

            # Fast row count
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

    print(f"Filesystem Inventory Complete: {len(files_info)} files found.")
    return files_info

def audit_cic_bell(files_info):
    print("\n--- PHASE 2, 3, 4, 5, 6, 7, 8: In-Depth CIC-Bell Analysis ---")
    cic_files = [f for f in files_info if f['dataset_source'] == 'cic_bell_dns_exf_2021']
    
    file_schemas = {}
    stateless_dfs = []
    stateful_dfs = []
    
    # Store pair comparisons
    pair_alignments = []
    
    # Organize by pair (stateful vs stateless)
    pairs = defaultdict(dict)
    for finfo in cic_files:
        fpath = finfo['rel_path']
        fname = finfo['filename']
        base_key = fname.replace('stateful_features-', '').replace('stateless_features-', '')
        dir_key = os.path.dirname(fpath)
        pair_id = f"{dir_key}/{base_key}"
        pairs[pair_id][finfo['state_type']] = finfo

    # Detailed column inspect per file
    detailed_file_stats = []
    
    for finfo in cic_files:
        p = finfo['rel_path']
        df = pd.read_csv(p, low_memory=False)
        cols = list(df.columns)
        
        col_stats = {}
        for c in cols:
            ser = df[c]
            null_count = int(ser.isnull().sum())
            null_pct = round(float(null_count / len(df) * 100), 2) if len(df) > 0 else 0
            n_unique = int(ser.nunique(dropna=False))
            is_const = (n_unique <= 1)
            is_near_const = (n_unique <= 3)
            
            # check inf
            inf_count = 0
            if pd.api.types.is_numeric_dtype(ser):
                inf_count = int(np.isinf(ser).sum())
            
            sample_vals = [str(x) for x in ser.dropna().unique()[:3]]
            
            col_stats[c] = {
                'dtype': str(ser.dtype),
                'null_count': null_count,
                'null_pct': null_pct,
                'inf_count': inf_count,
                'n_unique': n_unique,
                'is_constant': is_const,
                'is_near_constant': is_near_const,
                'samples': sample_vals
            }
            
        file_schemas[finfo['rel_path']] = {
            'filename': finfo['filename'],
            'row_count': len(df),
            'col_count': len(cols),
            'columns': cols,
            'col_stats': col_stats,
            'is_attack': finfo['is_attack'],
            'intensity': finfo['intensity'],
            'modality': finfo['modality'],
            'state_type': finfo['state_type']
        }
        
    with open('reports/dataset_audit/cic_bell_schema.json', 'w', encoding='utf-8') as f:
        json.dump(file_schemas, f, indent=2)
    print("Saved cic_bell_schema.json")
    
    # Statefulness alignment check
    print("Checking Stateful vs Stateless alignment across pairs...")
    pair_report = []
    for pair_id, pfiles in pairs.items():
        if 'stateless' in pfiles and 'stateful' in pfiles:
            sl_info = pfiles['stateless']
            sf_info = pfiles['stateful']
            
            sl_df = pd.read_csv(sl_info['rel_path'], low_memory=False)
            sf_df = pd.read_csv(sf_info['rel_path'], low_memory=False)
            
            exact_match_len = (len(sl_df) == len(sf_df))
            
            # Check timestamps in stateless
            has_timestamp = 'timestamp' in sl_df.columns
            sf_has_timestamp = 'timestamp' in sf_df.columns
            
            ts_monotonic = False
            ts_min = None
            ts_max = None
            duration_sec = 0.0
            if has_timestamp:
                try:
                    ts_dt = pd.to_datetime(sl_df['timestamp'], errors='coerce')
                    ts_monotonic = ts_dt.is_monotonic_increasing
                    ts_min = str(ts_dt.min())
                    ts_max = str(ts_dt.max())
                    duration_sec = (ts_dt.max() - ts_dt.min()).total_seconds()
                except Exception as e:
                    pass

            pair_report.append({
                'pair_id': pair_id,
                'stateless_file': sl_info['filename'],
                'stateful_file': sf_info['filename'],
                'stateless_rows': len(sl_df),
                'stateful_rows': len(sf_df),
                'row_count_delta': len(sl_df) - len(sf_df),
                'stateless_has_timestamp': has_timestamp,
                'stateful_has_timestamp': sf_has_timestamp,
                'timestamp_monotonic': ts_monotonic,
                'start_time': ts_min,
                'end_time': ts_max,
                'duration_seconds': duration_sec,
                'modality': sl_info['modality'],
                'intensity': sl_info['intensity'],
                'is_attack': sl_info['is_attack']
            })
            
    return file_schemas, pair_report

if __name__ == '__main__':
    files_info = audit_filesystem()
    schemas, pairs = audit_cic_bell(files_info)
    print("Audit step 1-2 done.")
