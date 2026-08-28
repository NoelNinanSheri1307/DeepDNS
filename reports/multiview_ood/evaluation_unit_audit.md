# Phase 2: Evaluation-Unit & Sample Independence Audit Report

- **Raw Test Queries**: 205,538
- **Total Sequence Windows**: 20,683
- **Captures Evaluated**: 5
- **Effective Disjoint Sequences at K=30**: $\approx 6,894$

### Window Overlap Dynamics
| Horizon ($K$) | Stride | Shared Queries | Overlap % | Statistical Independence |
| :--- | :--- | :--- | :--- | :--- |
| $K=5$ | 10 | 0 | 0.0% | Strictly Disjoint (100% Independent) |
| $K=10$ | 10 | 0 | 0.0% | Strictly Disjoint (100% Independent) |
| $K=15$ | 10 | 5 | 33.33% | 66.7% Novel Content |
| $K=20$ | 10 | 10 | 50.0% | 50.0% Novel Content |
| $K=25$ | 10 | 15 | 60.0% | 40.0% Novel Content |
| $K=30$ | 10 | 20 | 66.67% | 33.3% Novel Content |
