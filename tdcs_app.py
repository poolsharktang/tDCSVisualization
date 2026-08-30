"""PySide6 GUI for 8-channel EEG + tES montage setup and SimNIBS simulation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mne
import numpy as np
from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QUrl
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "simnibs4_examples"
RUNNER_SCRIPT = ROOT / "simnibs_run_tdcs.py"

EEG_CHANNELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

# Hardware Ch1–Ch8 default 10-10 locations. EEG and tES share these sites.
DEFAULT_EEG = ["POz", "P1", "P2", "O2", "O1", "Fp1", "Fp2", "Cz"]
STIM_CHANNELS = (1, 8)
RETURN_CHANNELS = (2, 3, 4, 5, 6, 7)
DEFAULT_STIM = {1}
DEFAULT_RETURN = {2, 3, 4, 5}

SOURCE_COLOR = QColor("#c0392b")
SINK_COLOR = QColor("#1d4ed8")
ASSIGNED_COLOR = QColor("#0f766e")
IDLE_COLOR = QColor("#d7dee8")
HOVER_COLOR = QColor("#f4d03f")
HEAD_FILL = QColor("#f7f1ea")
HEAD_LINE = QColor("#2c3e50")

STYLESHEET = """
QMainWindow, QWidget {
    background: #e8edf3;
    color: #1b2430;
    font-family: "Segoe UI";
    font-size: 13px;
}
QLabel#AppTitle {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
}
QLabel#AppSubtitle {
    color: #c5d0de;
    font-size: 12px;
}
QFrame#HeaderBar {
    background: #152238;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d5dde6;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #334155;
}
QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #1d4ed8;
}
QPlainTextEdit#LogView {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 12px;
}
QPushButton {
    background: #eef2f7;
    border: 1px solid #c9d3de;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #e2eaf3;
}
QPushButton:disabled {
    color: #94a3b8;
}
QPushButton#PrimaryButton {
    background: #0f766e;
    color: #ffffff;
    border: none;
    padding: 10px 16px;
    font-size: 14px;
}
QPushButton#PrimaryButton:hover {
    background: #0d9488;
}
QPushButton#DangerButton {
    background: #b91c1c;
    color: #ffffff;
    border: none;
}
QPushButton#DangerButton:hover {
    background: #dc2626;
}
QCheckBox#StimCheck {
    color: #c0392b;
    font-weight: 600;
}
QCheckBox#ReturnCheck {
    color: #1d4ed8;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #e2e8f0;
}
QHeaderView::section {
    background: #f1f5f9;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QScrollArea {
    border: none;
    background: transparent;
}
QSplitter::handle {
    background: #cbd5e1;
}
QLabel#HintLabel {
    color: #64748b;
    font-weight: 400;
}
QLabel#WarnLabel {
    color: #b45309;
    font-weight: 600;
}
"""


def load_montage_xy() -> dict[str, np.ndarray]:
    montage = mne.channels.make_standard_montage("standard_1020")
    positions_3d = montage.get_positions()["ch_pos"]
    raw_xy = {}
    for name in EEG_CHANNELS:
        if name in positions_3d:
            xyz = positions_3d[name]
            raw_xy[name] = np.array([xyz[0], xyz[1]], dtype=float)
    names = list(raw_xy)
    coords = np.array([raw_xy[name] for name in names])
    coords = coords - np.mean(coords, axis=0)
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    coords = coords / np.max(radius)
    return {name: coords[i] for i, name in enumerate(names)}


def find_simnibs_python() -> str:
    home = Path.home()
    candidates = [
        home / "SimNIBS-4.6" / "simnibs_env" / "python.exe",
        home / "AppData" / "Local" / "SimNIBS" / "simnibs_env" / "python.exe",
        Path(r"C:\Users\Te Tang\SimNIBS-4.6\simnibs_env\python.exe"),
        Path(r"C:\Users\Te Tang\AppData\Local\SimNIBS\simnibs_env\python.exe"),
    ]
    candidates.extend(sorted(home.glob("SimNIBS-*/simnibs_env/python.exe")))
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


def current_table(
    eeg_names: list[str],
    stim_chs: list[int],
    return_chs: list[int],
    total_ma: float,
) -> list[tuple]:
    rows = []
    if not stim_chs or not return_chs or total_ma <= 0:
        return rows
    source_ma = total_ma / len(stim_chs)
    sink_ma = -total_ma / len(return_chs)
    for ch in stim_chs:
        rows.append((ch, eeg_names[ch - 1], "Stim", source_ma))
    for ch in return_chs:
        rows.append((ch, eeg_names[ch - 1], "Return", sink_ma))
    return rows


class MontageWidget(QWidget):
    def __init__(self, xy: dict[str, np.ndarray], parent=None):
        super().__init__(parent)
        self.xy = xy
        self.channel_of: dict[str, int] = {}
        self.sources: set[str] = set()
        self.sinks: set[str] = set()
        self._hover: str | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(560, 560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_state(self, channel_of: dict[str, int], sources: list[str], sinks: list[str]) -> None:
        self.channel_of = dict(channel_of)
        self.sources = set(sources)
        self.sinks = set(sinks)
        self.update()

    def _layout(self) -> tuple[float, float, float]:
        margin = 28
        size = min(self.width(), self.height()) - 2 * margin
        scale = size / (2.56)
        cx = self.width() / 2
        cy = self.height() / 2 + 8
        return cx, cy, scale

    def _to_pixel(self, x: float, y: float) -> tuple[float, float]:
        cx, cy, scale = self._layout()
        return cx + x * scale, cy - y * scale

    def _nearest(self, px: float, py: float) -> str | None:
        nearest = None
        nearest_d = 1e9
        for name, pos in self.xy.items():
            x, y = self._to_pixel(float(pos[0]), float(pos[1]))
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dist < nearest_d:
                nearest = name
                nearest_d = dist
        _, _, scale = self._layout()
        hit_radius = max(14.0, 0.078 * scale)
        if nearest is None or nearest_d > hit_radius:
            return None
        return nearest

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor("#f8fafc"))
        cx, cy, scale = self._layout()
        radius = 1.08 * scale

        painter.setBrush(QBrush(HEAD_FILL))
        painter.setPen(QPen(HEAD_LINE, 2.4))
        painter.drawEllipse(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))

        painter.drawPolyline(QPolygonF([
            self._qpoint(-0.11, 1.07),
            self._qpoint(0.0, 1.20),
            self._qpoint(0.11, 1.07),
        ]))

        left_ear = QPainterPath()
        left_ear.moveTo(*self._to_pixel(-1.08, 0.16))
        left_ear.cubicTo(*self._to_pixel(-1.20, 0.10), *self._to_pixel(-1.20, -0.10), *self._to_pixel(-1.08, -0.16))
        right_ear = QPainterPath()
        right_ear.moveTo(*self._to_pixel(1.08, 0.16))
        right_ear.cubicTo(*self._to_pixel(1.20, 0.10), *self._to_pixel(1.20, -0.10), *self._to_pixel(1.08, -0.16))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(left_ear)
        painter.drawPath(right_ear)

        electrode_r = max(11.0, 0.072 * scale)
        name_font = QFont("Segoe UI", max(7, min(9, int(scale * 0.048))))
        name_font.setBold(True)
        badge_font = QFont("Segoe UI", max(7, min(9, int(scale * 0.042))))
        badge_font.setBold(True)

        for name, pos in self.xy.items():
            x, y = self._to_pixel(float(pos[0]), float(pos[1]))
            ch = self.channel_of.get(name)
            if name in self.sources:
                fill, text, border_w = SOURCE_COLOR, QColor("white"), 2.2
            elif name in self.sinks:
                fill, text, border_w = SINK_COLOR, QColor("white"), 2.2
            elif ch is not None:
                fill, text, border_w = ASSIGNED_COLOR, QColor("white"), 2.0
            else:
                fill, text, border_w = IDLE_COLOR, QColor("#1e293b"), 1.0
            if name == self._hover:
                painter.setPen(QPen(HOVER_COLOR, 3))
            else:
                painter.setPen(QPen(QColor("#0f172a"), border_w))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QRectF(x - electrode_r, y - electrode_r, 2 * electrode_r, 2 * electrode_r))
            painter.setPen(QPen(text))
            painter.setFont(name_font)
            painter.drawText(
                QRectF(x - electrode_r, y - electrode_r, 2 * electrode_r, 2 * electrode_r),
                Qt.AlignCenter,
                name,
            )
            if ch is not None:
                badge_r = max(7.0, electrode_r * 0.42)
                bx = x + electrode_r * 0.72
                by = y - electrode_r * 0.72
                painter.setBrush(QBrush(QColor("#152238")))
                painter.setPen(QPen(QColor("white"), 1))
                painter.drawEllipse(QRectF(bx - badge_r, by - badge_r, 2 * badge_r, 2 * badge_r))
                painter.setFont(badge_font)
                painter.setPen(QPen(QColor("white")))
                painter.drawText(
                    QRectF(bx - badge_r, by - badge_r, 2 * badge_r, 2 * badge_r),
                    Qt.AlignCenter,
                    str(ch),
                )

        painter.setPen(QPen(QColor("#64748b")))
        painter.setFont(QFont("Segoe UI", 10))
        hint = "Red = stim  ·  Blue = return  ·  Teal = EEG only  ·  Badge = hardware channel"
        if self._hover:
            ch = self.channel_of.get(self._hover)
            extra = f"Ch{ch}  " if ch is not None else ""
            hint = f"{extra}{self._hover}  ·  {hint}"
        painter.drawText(self.rect().adjusted(12, 8, -12, -8), Qt.AlignTop | Qt.AlignHCenter, hint)

    def _qpoint(self, x: float, y: float) -> QPointF:
        px, py = self._to_pixel(x, y)
        return QPointF(px, py)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        name = self._nearest(event.position().x(), event.position().y())
        if name != self._hover:
            self._hover = name
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("8-ch EEG + tES Simulation")
        self.resize(1420, 920)
        self._output_edited = False
        self._updating_combos = False
        self.eeg_names = list(DEFAULT_EEG)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_process_output)
        self.process.finished.connect(self._on_process_finished)

        self.xy = load_montage_xy()
        self._build_ui()
        self._refresh_head_models()
        self._sync_view()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(72)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 10, 22, 10)
        title = QLabel("8-channel EEG + tES")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Assign 8 shared EEG/tES electrodes, then check stim (Ch1/Ch8) and return (Ch2-Ch7)")
        subtitle.setObjectName("AppSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)
        self.montage = MontageWidget(self.xy)
        left_layout.addWidget(self.montage, 1)
        splitter.addWidget(left)

        right_panel = QWidget()
        right_panel.setMinimumWidth(430)
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(8, 16, 16, 12)
        right_panel_layout.setSpacing(8)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 4, 0)
        right_layout.setSpacing(8)

        eeg_box = QGroupBox("EEG / tES Electrodes")
        eeg_layout = QVBoxLayout(eeg_box)
        hint = QLabel(
            "EEG and tES share these 8 sites. Electrode names cannot be duplicated.\n"
            "Ch1 and Ch8: check 1 or both as stimulation.\n"
            "Ch2-Ch7: check at least one as return."
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        eeg_layout.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("Channel"), 0, 0)
        grid.addWidget(QLabel("10-10 site"), 0, 1)
        grid.addWidget(QLabel("tES role"), 0, 2)

        self.electrode_combos: list[QComboBox] = []
        self.role_checks: dict[int, QCheckBox] = {}
        for i in range(8):
            ch = i + 1
            ch_label = QLabel(f"Ch{ch}")
            ch_label.setMinimumWidth(36)
            combo = QComboBox()
            combo.addItems(EEG_CHANNELS)
            combo.setCurrentText(DEFAULT_EEG[i])
            combo.currentTextChanged.connect(lambda _name, channel=ch: self._on_electrode_chosen(channel))
            self.electrode_combos.append(combo)

            if ch in STIM_CHANNELS:
                check = QCheckBox("Stimulation")
                check.setObjectName("StimCheck")
                check.setChecked(ch in DEFAULT_STIM)
            else:
                check = QCheckBox("Return")
                check.setObjectName("ReturnCheck")
                check.setChecked(ch in DEFAULT_RETURN)
            check.toggled.connect(self._sync_view)
            self.role_checks[ch] = check

            row = i + 1
            grid.addWidget(ch_label, row, 0)
            grid.addWidget(combo, row, 1)
            grid.addWidget(check, row, 2)

        eeg_layout.addLayout(grid)
        reset_btn = QPushButton("Reset default montage")
        reset_btn.clicked.connect(self._reset_montage)
        eeg_layout.addWidget(reset_btn, 0, Qt.AlignLeft)
        self.warn_label = QLabel()
        self.warn_label.setObjectName("WarnLabel")
        self.warn_label.setWordWrap(True)
        eeg_layout.addWidget(self.warn_label)
        right_layout.addWidget(eeg_box)

        stim_box = QGroupBox("Stimulation")
        stim_form = QFormLayout(stim_box)
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setRange(0.05, 4.0)
        self.current_spin.setDecimals(3)
        self.current_spin.setSingleStep(0.1)
        self.current_spin.setValue(2.0)
        self.current_spin.setSuffix(" mA")
        self.current_spin.valueChanged.connect(self._sync_view)
        stim_form.addRow("Total current", self.current_spin)
        current_hint = QLabel("Stim current is split equally across checked Ch1/Ch8.\nReturn current is split equally across checked Ch2-Ch7.")
        current_hint.setObjectName("HintLabel")
        stim_form.addRow(current_hint)
        right_layout.addWidget(stim_box)

        elec_box = QGroupBox("Electrode")
        elec_form = QFormLayout(elec_box)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Circular", "Rectangular"])
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(5.0, 80.0)
        self.diameter_spin.setDecimals(1)
        self.diameter_spin.setValue(10.0)
        self.diameter_spin.setSuffix(" mm")
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(5.0, 80.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setValue(50.0)
        self.width_spin.setSuffix(" mm")
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(5.0, 80.0)
        self.height_spin.setDecimals(1)
        self.height_spin.setValue(50.0)
        self.height_spin.setSuffix(" mm")
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(1.0, 15.0)
        self.thickness_spin.setDecimals(1)
        self.thickness_spin.setValue(4.0)
        self.thickness_spin.setSuffix(" mm")
        elec_form.addRow("Shape", self.shape_combo)
        self.diameter_label = QLabel("Diameter")
        elec_form.addRow(self.diameter_label, self.diameter_spin)
        self.width_label = QLabel("Width")
        self.height_label = QLabel("Height")
        elec_form.addRow(self.width_label, self.width_spin)
        elec_form.addRow(self.height_label, self.height_spin)
        elec_form.addRow("Thickness", self.thickness_spin)
        right_layout.addWidget(elec_box)
        self._on_shape_changed(self.shape_combo.currentText())

        sim_box = QGroupBox("Simulation")
        sim_form = QFormLayout(sim_box)
        self.work_edit = QLineEdit(str(EXAMPLES_DIR))
        self.work_edit.editingFinished.connect(self._refresh_head_models)
        work_row = self._browse_row(self.work_edit, self._browse_workdir)
        self.model_combo = QComboBox()
        self.output_edit = QLineEdit("tdcs_POz_4return")
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.fields_edit = QLineEdit("veEjJ")
        self.gmsh_check = QCheckBox("Open results in Gmsh")
        self.gmsh_check.setChecked(True)
        self.vol_check = QCheckBox("Map fields to MRI volume")
        self.python_edit = QLineEdit(find_simnibs_python())
        python_row = self._browse_row(self.python_edit, self._browse_python)
        sim_form.addRow("Working directory", work_row)
        sim_form.addRow("Head model", self.model_combo)
        sim_form.addRow("Output folder", self.output_edit)
        sim_form.addRow("Fields", self.fields_edit)
        sim_form.addRow(self.gmsh_check)
        sim_form.addRow(self.vol_check)
        sim_form.addRow("SimNIBS Python", python_row)
        right_layout.addWidget(sim_box)

        table_box = QGroupBox("Channel Currents")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Ch", "Electrode", "Role", "Current"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(150)
        self.sum_label = QLabel("Sum: 0.000 mA")
        table_layout.addWidget(self.table)
        table_layout.addWidget(self.sum_label)
        right_layout.addWidget(table_box)
        right_layout.addStretch(1)
        right_scroll.setWidget(right)
        right_panel_layout.addWidget(right_scroll, 1)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run_simulation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_simulation)
        self.open_btn = QPushButton("Open Output")
        self.open_btn.clicked.connect(self._open_output)
        btn_row.addWidget(self.run_btn, 2)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.open_btn)
        right_panel_layout.addLayout(btn_row)
        splitter.addWidget(right_panel)
        splitter.setSizes([820, 520])

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(splitter, 3)
        self.log = QPlainTextEdit()
        self.log.setObjectName("LogView")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        self.log.setPlaceholderText("Simulation log will appear here.")
        body_layout.addWidget(self.log)
        root_layout.addWidget(body, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    def _browse_row(self, edit: QLineEdit, handler) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(handler)
        layout.addWidget(btn)
        return row

    def _on_shape_changed(self, shape: str) -> None:
        circular = shape == "Circular"
        self.diameter_spin.setVisible(circular)
        self.diameter_label.setVisible(circular)
        self.width_spin.setVisible(not circular)
        self.width_label.setVisible(not circular)
        self.height_spin.setVisible(not circular)
        self.height_label.setVisible(not circular)

    def _on_output_edited(self, _text: str) -> None:
        self._output_edited = True

    def _reset_montage(self) -> None:
        self._updating_combos = True
        self.eeg_names = list(DEFAULT_EEG)
        for i, name in enumerate(self.eeg_names):
            self.electrode_combos[i].setCurrentText(name)
        for ch, check in self.role_checks.items():
            if ch in STIM_CHANNELS:
                check.setChecked(ch in DEFAULT_STIM)
            else:
                check.setChecked(ch in DEFAULT_RETURN)
        self._updating_combos = False
        self._output_edited = False
        self._sync_view()

    def _on_electrode_chosen(self, channel: int) -> None:
        if self._updating_combos:
            return
        index = channel - 1
        new_name = self.electrode_combos[index].currentText()
        old_name = self.eeg_names[index]
        if new_name == old_name:
            return
        if new_name in self.eeg_names:
            other = self.eeg_names.index(new_name)
            self.eeg_names[other] = old_name
            self._updating_combos = True
            self.electrode_combos[other].setCurrentText(old_name)
            self._updating_combos = False
        self.eeg_names[index] = new_name
        self._sync_view()

    def _stim_channels(self) -> list[int]:
        return [ch for ch in STIM_CHANNELS if self.role_checks[ch].isChecked()]

    def _return_channels(self) -> list[int]:
        return [ch for ch in RETURN_CHANNELS if self.role_checks[ch].isChecked()]

    def _sources(self) -> list[str]:
        return [self.eeg_names[ch - 1] for ch in self._stim_channels()]

    def _sinks(self) -> list[str]:
        return [self.eeg_names[ch - 1] for ch in self._return_channels()]

    def _montage_error(self) -> str | None:
        if len(set(self.eeg_names)) != 8:
            return "Each of the 8 channels must use a different electrode."
        if not self._stim_channels():
            return "Select Ch1, Ch8, or both as stimulation electrodes."
        if not self._return_channels():
            return "Select at least one return electrode from Ch2-Ch7."
        return None

    def _auto_output_name(self) -> str:
        stim = self._sources()
        n_return = len(self._return_channels())
        if not stim:
            return "tdcs_gui_results"
        return f"tdcs_{'_'.join(stim)}_{n_return}return"

    def _sync_view(self) -> None:
        channel_of = {name: i + 1 for i, name in enumerate(self.eeg_names)}
        self.montage.set_state(channel_of, self._sources(), self._sinks())
        if not self._output_edited:
            self.output_edit.setText(self._auto_output_name())

        error = self._montage_error()
        self.warn_label.setText(error or "")
        self.run_btn.setEnabled(error is None and self.process.state() == QProcess.NotRunning)

        rows = current_table(
            self.eeg_names,
            self._stim_channels(),
            self._return_channels(),
            self.current_spin.value(),
        )
        self.table.setRowCount(len(rows))
        total = 0.0
        for i, (channel, name, role, amp) in enumerate(rows):
            total += amp
            values = [f"Ch{channel}", name, role, f"{amp:+.4f} mA"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if role == "Stim":
                    item.setForeground(SOURCE_COLOR)
                elif role == "Return":
                    item.setForeground(SINK_COLOR)
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()
        self.sum_label.setText(f"Sum: {total:+.6f} mA")

        stim_n = len(self._stim_channels())
        ret_n = len(self._return_channels())
        self.statusBar().showMessage(f"{stim_n} stim, {ret_n} return  ·  8 shared EEG/tES electrodes")

    def _refresh_head_models(self) -> None:
        work = Path(self.work_edit.text().strip() or EXAMPLES_DIR)
        current = self.model_combo.currentText()
        self.model_combo.clear()
        if work.is_dir():
            models = sorted(p.name for p in work.iterdir() if p.is_dir() and p.name.startswith("m2m_"))
            if models:
                self.model_combo.addItems(models)
                if current in models:
                    self.model_combo.setCurrentText(current)
                elif "m2m_ernie" in models:
                    self.model_combo.setCurrentText("m2m_ernie")
                return
        self.model_combo.addItem("m2m_ernie")

    def _browse_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select SimNIBS working directory", self.work_edit.text())
        if path:
            self.work_edit.setText(path)
            self._refresh_head_models()

    def _browse_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SimNIBS Python",
            self.python_edit.text() or str(Path.home()),
            "Python (python.exe);;All files (*.*)",
        )
        if path:
            self.python_edit.setText(path)

    def _config(self) -> dict:
        circular = self.shape_combo.currentText() == "Circular"
        stim_chs = self._stim_channels()
        return_chs = self._return_channels()
        return {
            "subpath": self.model_combo.currentText() or "m2m_ernie",
            "pathfem": self.output_edit.text().strip() or "tdcs_gui_results",
            "fields": self.fields_edit.text().strip() or "veEjJ",
            "open_in_gmsh": self.gmsh_check.isChecked(),
            "map_to_vol": self.vol_check.isChecked(),
            "electrode_shape": "ellipse" if circular else "rect",
            "electrode_diameter_mm": self.diameter_spin.value(),
            "electrode_width_mm": self.width_spin.value(),
            "electrode_height_mm": self.height_spin.value(),
            "electrode_thickness_mm": self.thickness_spin.value(),
            "eeg_channels": {str(i + 1): name for i, name in enumerate(self.eeg_names)},
            "stim_hardware_channels": stim_chs,
            "return_hardware_channels": return_chs,
            "sources": self._sources(),
            "sinks": self._sinks(),
            "total_current_ma": self.current_spin.value(),
        }

    def _validate(self) -> str | None:
        error = self._montage_error()
        if error:
            return error
        python_path = Path(self.python_edit.text().strip())
        if not python_path.is_file():
            return "SimNIBS Python was not found. Browse to simnibs_env/python.exe."
        if not RUNNER_SCRIPT.is_file():
            return f"Runner script missing: {RUNNER_SCRIPT}"
        work = Path(self.work_edit.text().strip())
        if not work.is_dir():
            return "Working directory does not exist."
        model = work / (self.model_combo.currentText() or "m2m_ernie")
        if not model.is_dir():
            return f"Head model not found: {model}"
        return None

    def _run_simulation(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "Cannot run simulation", error)
            return
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A simulation is already running.")
            return

        work = Path(self.work_edit.text().strip())
        config = self._config()
        config_path = work / "_tdcs_gui_last_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        self.log.clear()
        self._append_log(f"Working directory: {work}")
        self._append_log(f"SimNIBS Python: {self.python_edit.text()}")
        self._append_log(f"Config: {config_path}")
        self._append_log("-" * 48)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(work))
        self.process.setProgram(self.python_edit.text().strip())
        self.process.setArguments([str(RUNNER_SCRIPT), str(config_path)])
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Simulation running...")
        self.process.start()
        if not self.process.waitForStarted(5000):
            self.stop_btn.setEnabled(False)
            self._sync_view()
            QMessageBox.critical(self, "Failed to start", self.process.errorString())

    def _stop_simulation(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self._append_log("\nSimulation stopped by user.")

    def _on_process_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if text:
            self._append_log(text.rstrip("\n"))

    def _on_process_finished(self, exit_code: int, _status) -> None:
        self.stop_btn.setEnabled(False)
        self._sync_view()
        if exit_code == 0:
            self.statusBar().showMessage("Simulation finished")
            self._append_log("\nDone.")
        else:
            self.statusBar().showMessage(f"Simulation failed (exit {exit_code})")
            self._append_log(f"\nProcess exited with code {exit_code}.")

    def _open_output(self) -> None:
        work = Path(self.work_edit.text().strip())
        output = work / (self.output_edit.text().strip() or "tdcs_gui_results")
        if not output.exists():
            QMessageBox.information(self, "Not found", f"Output folder does not exist yet:\n{output}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
