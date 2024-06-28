import os
import sys
import random
import numpy as np

sys.path.append(os.path.abspath('.'))



def check_cwd():
    
    print(os.getcwd())
    assert os.path.basename(os.path.normpath(os.getcwd())) == "MMSoc", "Must run this file from the repository root (MMSoc/)"


def project_setup():
    check_cwd()
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 60)
    set_seed(42)



def print_colored(text, color='red'):
    foreground_colors = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
    }
    print(f"\033[{foreground_colors[color]}m{text}\033[0m")


def set_seed(seed, use_torch=True):
    print(f"Setting seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

