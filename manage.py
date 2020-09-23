#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import sklearn.ensemble
import sklearn.tree
import pickle
import pandas as pd
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
import cython
import sklearn
import sklearn.utils._cython_blas
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class hash_vector:
    def __init__(self, vec):
        self.vec = vec


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'people_counter.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
