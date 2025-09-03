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
Profiler Utility

With context manager + aggregator, and sprinkle with prof("phase").
Gives per-phase averages/percentiles every N frames.

Usage example:

prof = Prof(enable=True, report_every=20)

for bgr in src.frames():
    with prof("resize"):
        bgr = _resize_bgr(bgr, cfg.downscale)

prof.tick()  # prints every N frames

"""


from time import perf_counter
from collections import defaultdict, deque
import threading, os

class Prof:
    def __init__(self, enable: bool = False, report_every: int = 20, keep_last: int = 100):
        self.enable = enable or bool(int(os.getenv("RUNTIME_PROFILE", "0")))
        self.report_every = report_every
        self.lock = threading.Lock()
        self.totals = defaultdict(float)    # sum per phase
        self.counts = defaultdict(int)
        self.lastN  = defaultdict(lambda: deque(maxlen=keep_last))  # last-N window
        self._frame = 0
        self._stack = []   # nested timing support

    def __call__(self, name: str):
        return _ProfScope(self, name)

    def _enter(self, name):
        if not self.enable: return None
        t = perf_counter()
        self._stack.append((name, t))
        return t

    def _exit(self, token):
        if not self.enable or not token: return
        name, t0 = self._stack.pop()
        dt = perf_counter() - t0
        with self.lock:
            self.totals[name] += dt
            self.counts[name] += 1
            self.lastN[name].append(dt)

    def tick(self):
        if not self.enable: return
        self._frame += 1
        if self._frame % self.report_every == 0:
            self.report()

    def report(self):
        print("\n[PROFILE] last={} frames".format(self.report_every))
        with self.lock:
            for name in sorted(self.totals.keys()):
                c = self.counts[name]
                if c == 0: continue
                avg = self.totals[name] / c
                win = list(self.lastN[name])
                if win:
                    win_sorted = sorted(win)
                    p50 = win_sorted[len(win)//2]
                    p95 = win_sorted[int(len(win)*0.95)-1] if len(win) > 1 else win_sorted[-1]
                else:
                    p50 = p95 = avg
                print(f"  {name:<24s} avg={avg*1e3:7.2f} ms  p50={p50*1e3:7.2f}  p95={p95*1e3:7.2f}  n={c}")


# ---- context manager shim -----------------------------------------------
class _ProfScope:
    """with prof('phase'):  ...  """
    def __init__(self, prof: "Prof", name: str):
        self._prof = prof
        self._name = name
        self._token = None

    def __enter__(self):
        # returns a token (or None when disabled)
        self._token = self._prof._enter(self._name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._prof._exit(self._token)
        # do not suppress exceptions
        return False