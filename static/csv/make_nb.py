"""Create a notebook containing code from a script.
Run as:  python make_nb.py my_script.py
"""
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

nb = new_notebook()
with open(sys.argv[1]) as f:
    reader = f.readlines()
    code = []
    block = False
    for row in reader:
        if row.startswith("#",0):
            if len(code)>0:
                print(code)
                nb.cells.append(new_code_cell(code))
                nb.cells.append(new_code_cell(row))
                code = []
            else:
                nb.cells.append(new_code_cell(row))
        elif row.startswith("#",0) == False:
            code.append(row)
    nb.cells.append(new_code_cell(code))
nbformat.write(nb, sys.argv[1]+'.ipynb')