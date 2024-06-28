import json
import os
import platform
import random
import re
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook

def truncate_input(text: str, max_num_words: int = 256) -> str:
    """Truncate a string to a maximum number of words, ensuring proper punctuation.

    Args:
        text (str): Text to truncate.
        max_num_words (int): Maximum number of words allowed.

    Returns:
        str: Truncated text.
    """
    try:
        text = re.sub(r'\s+', ' ', re.sub(r"\n+", " ", text))
        words = text.split(" ")
        words = words[:max_num_words]
        text = " ".join(words)
        if not any(text.endswith(punctuation) for punctuation in ".,:;?!|"):
            text += '.'
    except:
        traceback.print_exc()
        text = ""
    return text
