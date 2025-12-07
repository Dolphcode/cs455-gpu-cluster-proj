# v0: Python Notebook

To run the code in this folder ideally we should be using a virtual environment. The following lines will setup and install all the requirements for the virtual environment.
```
mkdir .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
In general the virtual environment is activated using
```
source .venv/bin/activate
```
From here the ipynb file can be run using either jupyter notebook or some other IDE that can render/handle ipynb kernels. We've opted for using jupyter notebook. To set up the appropriate kernel use:
```
python -m ipykernel install --user --name .venv --display-name "455 Env"
```
