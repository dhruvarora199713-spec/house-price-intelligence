# ============================================
# Load Dataset Utility
# ============================================

# Purpose:

# Centralized dataset loading function.

import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    return df