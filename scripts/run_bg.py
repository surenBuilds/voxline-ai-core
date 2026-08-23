#!/usr/bin/env python3
"""Background training launcher - writes output to log file."""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = open('training_log.txt', 'w', encoding='utf-8')
p = subprocess.Popen(
    [sys.executable, 'scripts/train_small.py'],
    stdout=log, stderr=subprocess.STDOUT,
    cwd=os.getcwd()
)
print(f"PID: {p.pid}")
print(f"Log: training_log.txt")
