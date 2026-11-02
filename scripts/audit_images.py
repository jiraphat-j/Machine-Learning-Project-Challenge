from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.eda_utils import image_inventory

base = ROOT / '04_image_classification/task4_image_classification'
output = ROOT / 'artifacts'
output.mkdir(exist_ok=True)
for split in ['train', 'test']:
    frame = pd.read_csv(base / f'{split}.csv')
    inventory = image_inventory(base, frame, split)
    inventory.to_csv(output / f'image_{split}_inventory.csv', index=False)
    print(split, inventory[['exists', 'readable']].sum().to_dict(), flush=True)
