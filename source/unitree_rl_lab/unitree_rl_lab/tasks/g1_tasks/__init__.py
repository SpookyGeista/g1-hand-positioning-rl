
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module.

Aktuell ist in diesem Paket nur der einfache Wurf-Test (`g1_throw_test`)
aktiviert. Weitere Tasks (Pick&Place, Stack, etc.) können später wieder
hinzugefügt werden, sobald die entsprechenden Module im Repository vorhanden
oder portiert sind.
"""

# Nur Tasks importieren, die in diesem Repository tatsächlich existieren.
from . import g1_throw_test

__all__ = [
    "g1_throw_test",
]