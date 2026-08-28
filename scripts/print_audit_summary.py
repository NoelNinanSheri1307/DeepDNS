import json

with open('reports/dataset_audit/audit_numerical_results.json') as f:
    data = json.load(f)

print("=== PAIR COMPARISONS ===")
for r in data['pair_records']:
    print(f"[{r['category']} - {r['intensity']} - {r['modality']}] Stateless: {r['stateless_rows']}, Stateful: {r['stateful_rows']}, Diff: {r['row_diff']}, Match: {r['row_match']}, QPS: {r['queries_per_sec']}, DurationSec: {r['duration_sec']}")

print("\n=== LABEL SUMMARY ===")
print("Binary Counts:", data['label_summary']['binary_counts'])
print("Binary %:", data['label_summary']['binary_proportions'])
print("Modality Counts:", data['label_summary']['modality_counts'])
print("Intensity Counts:", data['label_summary']['intensity_counts'])

print("\n=== DUPLICATES ===")
print(data['duplicate_analysis'])

print("\n=== DNS THREATS ===")
print("Train rows:", data['dns_threats_summary']['train_rows'])
print("Test rows:", data['dns_threats_summary']['test_rows'])
print("Train classes:", data['dns_threats_summary']['train_classes'])
print("Test classes:", data['dns_threats_summary']['test_classes'])
print("Train/Test Domain Overlap:", data['dns_threats_summary']['train_test_domain_overlap'])

print("\n=== LOW SLOW STATS ===")
for s in data['low_slow_stats']:
    print(s)
