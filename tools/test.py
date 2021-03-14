import sys
import os
file_path = os.path.dirname(__file__)
parent_path = os.path.dirname(file_path)
sys.path.append(parent_path)
import numpy as np
from lib.activate_function import ActivateFun
fun = ActivateFun()