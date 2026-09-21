"""Execute and retain outputs, including diagnostics if a cell fails."""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / 'Clustering_Deep_Dive.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(root)}})
client.on_cell_start = lambda cell, cell_index: print(f'CELL {cell_index}: {cell.source[:65]}', flush=True)
start = time.time()
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print(f'Completed in {time.time()-start:.1f}s', flush=True)
