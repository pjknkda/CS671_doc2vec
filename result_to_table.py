import json
import re
import sys

import pandas as pd

filepath = sys.argv[1]

print('Load', filepath)

rows = []
with open(filepath, 'r') as f:
    results = json.load(f)
    for k, v in results.items():
        m = re.match('(.*)_bat_(\d+)_maxlen_(\d+)_unit_(\d+)_layer_(\d+)_maxepoch_(\d+)_(\d+)', k)

        inf_mode = m.group(1)
        batch_size = m.group(2)
        max_length = m.group(3)
        num_unit = m.group(4)
        num_layer = m.group(5)
        epoch = m.group(6)

        precision = v[-2][0]
        recall = v[-2][1]
        f1 = v[-2][2]

        rows.append({
            'inf_mode': inf_mode,
            'batch_size': batch_size,
            'sentence_length': max_length,
            'num_unit': num_unit,
            'num_layer': num_layer,
            'epoch': epoch,
            'precision': precision,
            'recall': recall,
            'f1': f1})

pd.DataFrame(rows,
             columns=['inf_mode', 'batch_size', 'sentence_length',
                      'num_unit', 'num_layer', 'epoch',
                      'precision', 'recall', 'f1']).to_csv(filepath + '.csv')
