import torch.nn as nn

def printHello():
    print('hi')
    return -1

# -> Change to the correct Python installation (3.8) 
#
# pyenv("Version", "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3")


# -> Import module and then call function
#
# py.importlib.import_module('print')
# py.print.printHello()


# -> If you modified the Python file, here is how to reload it in Matlab.
#
# clear classes
# mod = py.importlib.import_module('hello');
# py.importlib.reload(mod);
# py.print.printHello()


# -> To check version of python:
#
# pe = pyenv;
# pe.Version
# (should return 3.8)
