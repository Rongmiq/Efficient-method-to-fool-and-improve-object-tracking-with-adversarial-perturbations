import os
import sys
import argparse
import importlib
import pdb

parser = argparse.ArgumentParser(description='adv type.')
parser.add_argument('--adv_type', type=str, default='adv_s')  # trsiam_vot / trdimp_vot
parser.add_argument('--run_id', type=int, default=None)
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
args = parser.parse_args()