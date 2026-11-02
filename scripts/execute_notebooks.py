"""Run synchronous notebook cells in IPython and save their real rich outputs.

This avoids starting local Jupyter sockets in restricted environments.
Each notebook uses a clean namespace, and every code cell is executed in order.
"""
import os
from pathlib import Path
import sys

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

import matplotlib
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]
config = Config()
config.HistoryManager.enabled = False
config.HistoryManager.hist_file = ':memory:'
shell = InteractiveShell.instance(config=config)
configure_inline_support(shell, 'inline')

for argument in sys.argv[1:]:
    path = ROOT / argument
    notebook = nbformat.read(path, as_version=4)
    shell.reset(new_session=False)
    os.chdir(path.parent)
    count = 0
    print('Executing', path, flush=True)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'  cell {index + 1}: {cell.source.splitlines()[0][:90]}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data,
                                                       metadata=output.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__,
                                                       evalue=str(error), traceback=[captured.stdout]))
        nbformat.write(notebook, path)
        if error:
            print(captured.stdout, flush=True)
            raise RuntimeError(f'{path.name}, cell {index + 1}: {error}') from error
    notebook.metadata['execution_method'] = 'IPython, sequential cells, clean namespace, captured rich outputs'
    nbformat.write(notebook, path)
    print('Saved', path, flush=True)
