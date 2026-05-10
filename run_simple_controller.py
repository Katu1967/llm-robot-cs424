#!/usr/bin/env python3
import os
import sys
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
os.chdir(ROOT)

runpy.run_path(os.path.join(SRC, "simple_controller.py"), run_name="__main__")