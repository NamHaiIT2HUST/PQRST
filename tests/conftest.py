"""Pytest configuration and environment fixtures for PQRST."""

import os
from pathlib import Path

# Automatically configure JAVA_HOME for JPype / IDTxl / JIDT
if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
    candidates = [
        r"C:\Users\Nguyen Dao Nam Hai\.jdks\ms-17.0.17",
        r"C:\Program Files\Java\jdk-22",
        r"C:\Program Files\Java\jdk-17",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            os.environ["JAVA_HOME"] = cand
            break
