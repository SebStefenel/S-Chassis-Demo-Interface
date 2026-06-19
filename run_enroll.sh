#!/bin/bash
# Launch the SDV Enrollment Tool using the project venv
cd "$(dirname "$0")"
source venv/bin/activate
python enroll_driver.py "$@"
