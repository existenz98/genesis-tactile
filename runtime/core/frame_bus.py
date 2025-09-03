# -----------------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later WITH LicenseRef-YF-Device-Interface-Exception
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# This file is part of the Runtime of the tactile vision platform.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE-RUNTIME-AGPL for details.
#
# Special Exception (Device Interface Exception):
#   Proprietary or separately-licensed device drivers or hardware interface
#   modules that communicate with the Runtime solely through the documented
#   TSI/plugin/IPC interfaces are not considered derivative works of the
#   Runtime by this project, and thus are not subject to the copyleft
#   obligations of the AGPL, provided they do not include or modify Runtime code.
#   See LICENSE-EXCEPTIONS for the full text.
#
# Patent Notice:
#   Except for any rights granted under the applicable open-source license,
#   no patent license is granted or implied. Users are responsible for ensuring
#   their use does not infringe third-party patents (e.g., tactile sensor
#   hardware or methods).
#
# Citation:
#   If you use this software in academic work, please cite the associated
#   publications when available.
# -----------------------------------------------------------------------------


"""
Thread-safe latest-value pub/sub bus.

- publish(topic, data): stores the newest payload; older payload is dropped.
- pull_latest(topic): returns (version:int, data:any) or (None, None) if nothing yet.  good for polling from a UI timer.
- subscribe(): callback mechanism  (not implemented yet)
"""

from __future__ import annotations
import threading
from typing import Any, Dict, Optional, Tuple

class _Topic:
    """
    a latest-value queu
    """
    __slots__ = ("_lock", "_data", "_ver")
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Any = None
        self._ver: int = 0

    def publish(self, data: Any) -> int:
        with self._lock:
            self._data = data
            self._ver += 1
            return self._ver

    def pull_latest(self) -> Tuple[Optional[int], Any]:
        with self._lock:
            if self._ver == 0:
                return None, None
            return self._ver, self._data

class FrameBus:
    """
    list of topics
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._topics: Dict[str, _Topic] = {}

    def _get_topic(self, name: str) -> _Topic:
        with self._lock:
            t = self._topics.get(name)
            if t is None:
                t = _Topic()
                self._topics[name] = t
            return t

    def publish(self, topic: str, data: Any) -> int:
        return self._get_topic(topic).publish(data)

    def pull_latest(self, topic: str) -> Tuple[Optional[int], Any]:
        return self._get_topic(topic).pull_latest()


