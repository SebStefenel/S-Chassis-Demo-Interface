#!/bin/bash
# Launch the SDV Monitor using the project venv
cd "$(dirname "$0")"
source venv/bin/activate
python sdv_monitor.py "$@"
