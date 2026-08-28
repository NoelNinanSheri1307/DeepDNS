import os
import sys

# Ensure repository root is on sys.path for test discovery
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
