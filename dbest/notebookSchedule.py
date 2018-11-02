import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


# notebooks = ["e_generate_queries_0.1.ipynb","e_generate_queries_10.ipynb"]
notebooks = ["etrade_1m_0.1.ipynb","etrade_1m_10.ipynb"]
notebooks = ["etrade_hive_0.1.ipynb","etrade_hive_10.ipynb","etrade_hive_percentile"]
for notebook in notebooks:
    print("Starting running "+notebook)
    with open(notebook) as f:
        nb = nbformat.read(f,as_version=4)
    ep_python = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        # path specifies which folder to execute the notebooks in, so set it to the one that you need so your file path references are correct
        out = ep_python.preprocess(nb, {'metadata': {'path': '.'}})
    except CellExecutionError:
        msg = 'Error executing the notebook "%s".\n\n' % notebook
        msg += 'See notebook "%s" for the traceback.' % notebook
        print(msg)
        raise
    finally:
        nbformat.write(nb, open(notebook, mode='wt'))


