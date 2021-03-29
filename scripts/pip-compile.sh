#!/bin/sh
lock_file="requirements-lock.txt"
pip-compile setup.py --find-links=https://download.pytorch.org/whl/torch_stable.html --generate-hashes --upgrade --output-file=$lock_file