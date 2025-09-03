# -----------------------------------------------------------------------------
# 3D Visualizer — Application Source
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Yue Fei <feiyuefy@gmail.com>
#
# License (Code):
#   This source file is MIT-licensed. See LICENSE-APPS-MIT.
#
# Brand & UI Assets (Not MIT-licensed):
#   Project name, logo(s), icons, UI themes/skins, and other brand assets are
#   NOT licensed under MIT. Use requires a separate license. See:
#     - LICENSE-APPS-ASSETS
#     - TRADEMARKS.md
#
# Patents:
#   Except as may be granted under the applicable open-source license, no patent
#   rights are granted or implied. Users are responsible for third-party patent
#   clearances.
#
# Notes:
#   - Redistribution must retain this header and the referenced license files.
#   - Forks may not imply endorsement or use the original branding without
#     permission (see TRADEMARKS.md).
# -----------------------------------------------------------------------------


from __future__ import annotations
from .utils.qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .config import AppConfig
from .widgets.brand_bar import BrandBar
from .widgets.stats_bar import StatsBar
from .controller.app_controller import AppController
from .panes.camera_pane import CameraPane
from .panes.flow_pane import FlowPane
from .panes.force_pane import ForcePane
from .panes.view3d_pane import View3DPane

def launch_app():
    import sys
    app = QtWidgets.QApplication(sys.argv)    

    # ---- set up dark theme (paste before MainWindow creation) ----
    dark_ss = """
    /* general background */
    QMainWindow, QWidget, QFrame, QDockWidget, QGroupBox {
        background-color: #151515;
        color: #E6E6E6;
    }

    /* Brand bar override (keeps gradient text visible) */
    BrandBar, QLabel#brandLabel {
        background-color: #151515;
    }

    /* dock widget title & frame */
    QDockWidget::title { background: #1b1b1b; color: #dcdcdc; padding: 6px; }

    /* group boxes (panes) */
    QGroupBox {
        border: 1px solid #2a2a2a;
        margin-top: 6px;
        padding-top: 4px;
        background: #171717;
    }
    QGroupBox:title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 2px 6px;
        color: #dcdcdc;
    }

    /* status bar */
    QStatusBar { background-color: #101010; color: #cfcfcf; }

    /* buttons and combobox / spinboxes */
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #212121;
        color: #eaeaea;
        border: 1px solid #333333;
        padding: 4px;
    }
    QPushButton:hover { background-color: #2b2b2b; }

    /* line edits */
    QLineEdit, QTextEdit {
        background-color: #1a1a1a;
        color: #eaeaea;
        border: 1px solid #2c2c2c;
    }

    /* Scrollbar styling */
    QScrollBar:vertical { background: #151515; width: 12px; }
    QScrollBar::handle:vertical { background: #2a2a2a; min-height: 20px; border-radius: 6px; }
    QScrollBar::add-line, QScrollBar::sub-line { height: 0px; }

    /* tooltips */
    QToolTip {
        background-color: #2a2a2a;
        color: #f0f0f0;
        border: 1px solid #3a3a3a;
    }
    """
    app.setStyleSheet(dark_ss)

    # pyqtgraph global theme
    pg.setConfigOption('background', (21,21,21))
    pg.setConfigOption('foreground', 'w')

    cfg = AppConfig()

    w = MainWindow(cfg)
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(f"{cfg.info.name}  {cfg.info.version}")

        # ---- Top branding bar
        self.brand = BrandBar("Genesis", ".touch", version=cfg.info.version, build=cfg.info.build)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top); top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(self.brand)

        # ---- Central panes (2x2 grid) + inspector dock placeholder
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central); grid.setContentsMargins(8,8,8,8); grid.setSpacing(8)

        self.p_cam = CameraPane("Camera")
        self.p_flow = FlowPane("Flow")
        self.p_force = ForcePane("Force")
        self.p_3d = View3DPane()

        if False:
            # 2x2 layout
            grid.addWidget(wrap_group("Camera", self.p_cam), 0, 0)
            grid.addWidget(wrap_group("Flow", self.p_flow), 0, 1)
            grid.addWidget(wrap_group("Force", self.p_force), 1, 0)
            grid.addWidget(wrap_group("3D Force", self.p_3d), 1, 1)
        else:
            # (row1: Camera | Flow | Force2D ; row2: 3D Force)
            grid.addWidget(wrap_group("Camera",   self.p_cam),  0, 0)
            grid.addWidget(wrap_group("Flow",     self.p_flow), 0, 1)
            grid.addWidget(wrap_group("Force",    self.p_force),0, 2)
            grid.addWidget(wrap_group("3D Force", self.p_3d),   1, 0, 1, 3)  # row=1, col=0, rowspan=1, colspan=3
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 1)
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 2)



        # ---- Right inspector (algorithm switch, view params)
        self.dock = QtWidgets.QDockWidget("Inspector", self)
        self.dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.dock.setWidget(self._build_inspector())
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)

        # ---- Status bar
        self.status = StatsBar(self)
        self.setStatusBar(self.status)

        # ---- Layout stack
        main_central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(main_central); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(top)
        lay.addWidget(central, 1)
        self.setCentralWidget(main_central)

        # ---- Controller (SDK + timers)
        self.ctrl = AppController(cfg.sdk.notify_ep, cfg.sdk.ctrl_ep, update_hz=cfg.view.update_hz)
        self.ctrl.frame_arrived.connect(self._on_frame)
        self.ctrl.start()
        self.brand.set_status("Connected", QtGui.QColor(60,200,100))


        # ----- Inspector UI -----
    def _build_inspector(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # Algorithm
        gb_algo = QtWidgets.QGroupBox("Algorithm")
        f = QtWidgets.QFormLayout(gb_algo)
        self.cmb_algo = QtWidgets.QComboBox()
        self.cmb_algo.addItems(["Physics", "CNN", "iFEM"])
        self.cmb_algo.currentIndexChanged.connect(self._on_algo_changed)
        f.addRow("Choose:", self.cmb_algo)

        # Flow 2D plot params
        gb_flow = QtWidgets.QGroupBox("Flow")
        lf = QtWidgets.QFormLayout(gb_flow)
        self.sp_flow_stride = QtWidgets.QSpinBox(); self.sp_flow_stride.setRange(4, 64); self.sp_flow_stride.setValue(self.cfg.view.flow_stride)
        self.sp_flow_scale  = QtWidgets.QDoubleSpinBox(); self.sp_flow_scale.setRange(0.1, 40); self.sp_flow_scale.setValue(self.cfg.view.flow_scale)
        self.sp_flow_minlen = QtWidgets.QDoubleSpinBox(); self.sp_flow_minlen.setRange(0.0, 10); self.sp_flow_minlen.setValue(self.cfg.view.flow_minlen)
        lf.addRow("Stride(px)", self.sp_flow_stride)
        lf.addRow("Arrow scale", self.sp_flow_scale)
        lf.addRow("Min length", self.sp_flow_minlen)
        self.sp_flow_stride.valueChanged.connect(self._apply_flow_params)
        self.sp_flow_scale.valueChanged.connect(self._apply_flow_params)
        self.sp_flow_minlen.valueChanged.connect(self._apply_flow_params)

        # Force 2D plot params
        gb_force = QtWidgets.QGroupBox("Force")
        ff = QtWidgets.QFormLayout(gb_force)
        self.sp_force_scale = QtWidgets.QDoubleSpinBox(); self.sp_force_scale.setRange(0.1, 40); self.sp_force_scale.setValue(self.cfg.view.force_quiver_scale)
        ff.addRow("Quiver scale", self.sp_force_scale)
        self.sp_force_scale.valueChanged.connect(self._apply_force_params)

        # 3D Force view params
        gb_3d = QtWidgets.QGroupBox("3D")
        f3 = QtWidgets.QFormLayout(gb_3d)
        self.sp3_glyph = QtWidgets.QDoubleSpinBox()
        self.sp3_glyph.setRange(0.05, 5.0); self.sp3_glyph.setSingleStep(0.05); self.sp3_glyph.setValue(0.6)
        self.sp3_z = QtWidgets.QDoubleSpinBox()
        self.sp3_z.setRange(0.05, 5.0); self.sp3_z.setSingleStep(0.05); self.sp3_z.setValue(0.2)
        f3.addRow("Arrow size", self.sp3_glyph)
        f3.addRow("Arrow Z weight", self.sp3_z)
        self.sp3_glyph.valueChanged.connect(self._apply_3d_params)
        self.sp3_z.valueChanged.connect(self._apply_3d_params)

        v.addWidget(gb_algo)
        v.addWidget(gb_flow)
        v.addWidget(gb_force)
        v.addWidget(gb_3d)   # don't forget to add to the inspector layout
        v.addStretch(1)

        return w

    def _apply_3d_params(self):
        self.p_3d.set_params(glyph_factor=self.sp3_glyph.value(),
                         z_weight=self.sp3_z.value())
    
    def _on_algo_changed(self, idx: int):
        ok = self.ctrl.set_algo(idx+1)  # 1..3
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Algorithm", "Failed to set algorithm")

    def _apply_flow_params(self):
        self.p_flow.set_params(self.sp_flow_stride.value(),
                               self.sp_flow_scale.value(),
                               self.sp_flow_minlen.value())

    def _apply_force_params(self):
        self.p_force.set_params(self.sp_force_scale.value())

    
    # ----- Frame update -----

    @QtCore.Slot(dict)
    def _on_frame(self, fr: dict):
        # Camera
        cam = fr.get("camera", None)
        if cam is not None:
            # SDK provides RGB; pane expects RGB
            self.p_cam.set_image_rgb(cam)

        # Flow
        vy = fr.get("vy", None); vx = fr.get("vx", None)
        if vy is not None and vx is not None:
            self.p_flow.update_flow(vy, vx)

        # Force
        p = fr.get("p", None); tx = fr.get("tx", None); ty = fr.get("ty", None)
        if p is not None and tx is not None and ty is not None:
            self.p_force.update_force(p, tx, ty)
            self.p_3d.update_scene(p, tx, ty, cell_mm=float(fr.get("cell_mm", 1.0)))

        # Stats
        self.status.update_stats(
            Seq=fr.get("seq", "-"),
            CamFPS=f"{fr.get('_ui_fps', 0):.1f}",
            Algo=self.cmb_algo.currentText(),
            mm_per_px=f"{fr.get('mm_per_px', 0):.3f}",
            cell_mm=f"{fr.get('cell_mm', 0):.2f}"
        )


def wrap_group(title: str, w: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
    gb = QtWidgets.QGroupBox(title)
    gb.setLayout(QtWidgets.QVBoxLayout())
    gb.layout().setContentsMargins(4,4,4,4)
    gb.layout().addWidget(w)
    return gb
