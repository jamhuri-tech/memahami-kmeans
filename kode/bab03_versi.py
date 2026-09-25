"""Bab 3: versi Python dan pustaka yang menghasilkan angka di buku."""
import platform

import matplotlib
import numpy
import openpyxl
import pandas
import PIL
import scipy
import sklearn

if __name__ == "__main__":
    print(f"{'Python':13s} {platform.python_version()}")
    for nama, modul in (("numpy", numpy), ("scipy", scipy),
                        ("scikit-learn", sklearn),
                        ("matplotlib", matplotlib), ("pandas", pandas),
                        ("openpyxl", openpyxl), ("pillow", PIL)):
        print(f"{nama:13s} {modul.__version__}")
