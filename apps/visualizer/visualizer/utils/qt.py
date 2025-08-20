"""
Qt compatibility helper

Prefer PySide6; fall back to PyQt5 if needed.
"""

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    API = "PySide6"
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    API = "PyQt5"
