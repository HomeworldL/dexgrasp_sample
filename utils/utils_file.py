import os

def ensure_dir_for_file(filepath: str):
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

