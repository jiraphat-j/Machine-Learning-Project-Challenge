from pathlib import Path
import ast
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.baseline_utils import validate_submission

notebooks = [ROOT / '00_data_audit.ipynb', ROOT / '02_baseline_submission.ipynb',
             *sorted(ROOT.glob('task*/01_eda.ipynb')),
             *sorted(ROOT.glob('task*/02_baseline.ipynb')), ROOT / '03_make_submission.ipynb']
assert len(notebooks) == 13
rows = []
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    codes = [cell for cell in notebook.cells if cell.cell_type == 'code']
    executed = sum(cell.execution_count is not None for cell in codes)
    if path.name in {'02_baseline.ipynb', '03_make_submission.ipynb'}:
        assert executed in {0, len(codes)}, path
        if executed == 0:
            assert notebook.metadata.get('execution_status') == 'not_run', path
        source = path.with_suffix('.py').read_text(encoding='utf-8')
        compile(source, str(path), 'exec')
    else:
        assert executed == len(codes), path
    for cell in codes:
        ast.parse(cell.source)
    assert not any(output.output_type == 'error' for cell in codes for output in cell.outputs), path
    plots = sum('image/png' in output.get('data', {}) for cell in codes for output in cell.outputs)
    if path.name == '01_eda.ipynb':
        headings = [cell.source for cell in notebook.cells if cell.cell_type == 'markdown'
                    and cell.source.startswith('## ')]
        assert len(headings) == 10, (path, len(headings))
        assert plots > 0, path
        for cell in codes:
            tree = ast.parse(cell.source)
            assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr in {'fit', 'fit_transform', 'partial_fit'}
                           for node in ast.walk(tree)), path
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and ('display(' in cell.source or 'plt.show()' in cell.source):
                next_cell = notebook.cells[index + 1]
                assert next_cell.cell_type == 'markdown', path
                for label in ['Findings & Interpretation', '**Findings:**', '**Interpretation:**',
                              '**ML Implications:**', '**Recommended Action:**']:
                    assert label in next_cell.source, (path, index, label)
    rows.append({'notebook': str(path.relative_to(ROOT)), 'executed_cells': executed,
                 'code_cells': len(codes), 'plots': plots})

sample = pd.read_csv(ROOT / 'sample_submission.csv')
submission = pd.read_csv(ROOT / 'submissions/baseline_submission.csv')
test_paths = ['01_binary_classification/test.csv', '02_multiclass_classification/test.csv',
              '03_regression/test.csv', '04_image_classification/task4_image_classification/test.csv',
              '05_anomaly_detection/test.csv']
lengths = [len(pd.read_csv(ROOT / path)) for path in test_paths]
validate_submission(submission, sample, lengths)
for task in range(1, 6):
    saved = pd.read_csv(ROOT / f'artifacts/task{task}_predictions.csv')
    assert np.array_equal(saved.row.to_numpy(), np.arange(len(sample)))
    assert np.allclose(saved.prediction, submission[f'task{task}'], rtol=0, atol=1e-8), task
scores = pd.read_csv(ROOT / 'artifacts/validation_scores.csv')
assert set(scores.task) == {1, 2, 3, 4}
assert np.isfinite(scores.value).all()
print(pd.DataFrame(rows).to_string(index=False))
print(scores.to_string(index=False))
print('Submission verified:', submission.shape, 'missing:', submission.isna().sum().sum())
