import os
import sys
import json
import gzip
import pandas as pd
import numpy as np

os.makedirs('reports/dataset_audit', exist_ok=True)
os.makedirs('scripts', exist_ok=True)

print("Starting deep dataset forensic inspection...")
