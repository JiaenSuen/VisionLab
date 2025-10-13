import sys
import os

def add_root_to_syspath(levels_up=1):
    path = os.path.abspath(__file__)
    for _ in range(levels_up):
        path = os.path.dirname(path)
    if path not in sys.path:
        sys.path.insert(0, path)
