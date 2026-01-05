"""参数设置面板"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QGroupBox, QPushButton, QTabWidget, QSlider,
    QProgressBar, QFileDialog, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from dataclasses import dataclass


@dataclass
class GrooveParams:
    """凹槽参数"""
    width: float = 1.0
    depth: float = 0.5
    auto_width: bool = True
    min_width: float = 0.5
    max_width: float = 5.0


@dataclass
class ExportParams:
    """导出参数"""
    scale: float = 1.0
    output_dir: str = ""


class ParamPanel(QWidget):
    """参数设置面板"""

    # 信号
    groove_params_changed = pyqtSignal(object)  # GrooveParams
    generate_grooves_clicked = pyqtSignal()
    flatten_clicked = pyqtSignal()
    export_all_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.groove_params = GrooveParams()
        self.export_params = ExportParams()

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 选项卡
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # 凹槽设置
        groove_tab = self._create_groove_tab()
        tabs.addTab(groove_tab, "凹槽")

        # 展开设置
        flatten_tab = self._create_flatten_tab()
        tabs.addTab(flatten_tab, "展开")

        # 导出设置
        export_tab = self._create_export_tab()
        tabs.addTab(export_tab, "导出")

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # 状态标签
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def _create_groove_tab(self) -> QWidget:
        """创建凹槽设置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 宽度
        width_group = QGroupBox("宽度设置")
        width_layout = QVBoxLayout(width_group)

        # 自动宽度
        self.auto_width_check = QCheckBox("自动调整宽度（根据汇聚线路数）")
        self.auto_width_check.setChecked(True)
        self.auto_width_check.stateChanged.connect(self._on_auto_width_changed)
        width_layout.addWidget(self.auto_width_check)

        # 基础宽度
        base_width_layout = QHBoxLayout()
        base_width_layout.addWidget(QLabel("基础宽度 (mm):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 10.0)
        self.width_spin.setValue(1.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.valueChanged.connect(self._on_params_changed)
        base_width_layout.addWidget(self.width_spin)
        width_layout.addLayout(base_width_layout)

        # 最小/最大宽度
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("范围:"))
        self.min_width_spin = QDoubleSpinBox()
        self.min_width_spin.setRange(0.1, 5.0)
        self.min_width_spin.setValue(0.5)
        self.min_width_spin.valueChanged.connect(self._on_params_changed)
        range_layout.addWidget(self.min_width_spin)
        range_layout.addWidget(QLabel("-"))
        self.max_width_spin = QDoubleSpinBox()
        self.max_width_spin.setRange(0.5, 20.0)
        self.max_width_spin.setValue(5.0)
        self.max_width_spin.valueChanged.connect(self._on_params_changed)
        range_layout.addWidget(self.max_width_spin)
        range_layout.addWidget(QLabel("mm"))
        width_layout.addLayout(range_layout)

        layout.addWidget(width_group)

        # 深度
        depth_group = QGroupBox("深度设置")
        depth_layout = QVBoxLayout(depth_group)

        depth_spin_layout = QHBoxLayout()
        depth_spin_layout.addWidget(QLabel("凹槽深度 (mm):"))
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 5.0)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setSingleStep(0.1)
        self.depth_spin.valueChanged.connect(self._on_params_changed)
        depth_spin_layout.addWidget(self.depth_spin)
        depth_layout.addLayout(depth_spin_layout)

        # 深度滑块
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 50)
        self.depth_slider.setValue(5)
        self.depth_slider.valueChanged.connect(
            lambda v: self.depth_spin.setValue(v / 10)
        )
        depth_layout.addWidget(self.depth_slider)

        layout.addWidget(depth_group)

        # 生成按钮
        self.btn_generate_grooves = QPushButton("生成凹槽预览")
        self.btn_generate_grooves.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
        """)
        self.btn_generate_grooves.clicked.connect(
            lambda: self.generate_grooves_clicked.emit()
        )
        layout.addWidget(self.btn_generate_grooves)

        layout.addStretch()
        return widget

    def _create_flatten_tab(self) -> QWidget:
        """创建展开设置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 展开方法
        method_group = QGroupBox("展开算法")
        method_layout = QVBoxLayout(method_group)

        self.lscm_radio = QCheckBox("LSCM (最小二乘共形映射)")
        self.lscm_radio.setChecked(True)
        method_layout.addWidget(self.lscm_radio)

        method_layout.addWidget(QLabel("保持角度关系，适合FPC制造"))

        layout.addWidget(method_group)

        # 缩放
        scale_group = QGroupBox("缩放设置")
        scale_layout = QVBoxLayout(scale_group)

        scale_spin_layout = QHBoxLayout()
        scale_spin_layout.addWidget(QLabel("输出比例:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSuffix(" : 1")
        scale_spin_layout.addWidget(self.scale_spin)
        scale_layout.addLayout(scale_spin_layout)

        layout.addWidget(scale_group)

        # 展开按钮
        self.btn_flatten = QPushButton("展开曲面")
        self.btn_flatten.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        self.btn_flatten.clicked.connect(lambda: self.flatten_clicked.emit())
        layout.addWidget(self.btn_flatten)

        # 变形信息
        self.distortion_label = QLabel("变形量: -")
        layout.addWidget(self.distortion_label)

        layout.addStretch()
        return widget

    def _create_export_tab(self) -> QWidget:
        """创建导出设置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 输出目录
        dir_group = QGroupBox("输出目录")
        dir_layout = QHBoxLayout(dir_group)

        self.dir_edit = QLabel("未选择")
        self.dir_edit.setWordWrap(True)
        dir_layout.addWidget(self.dir_edit)

        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        dir_layout.addWidget(self.btn_browse)

        layout.addWidget(dir_group)

        # 输出文件
        files_group = QGroupBox("输出文件")
        files_layout = QVBoxLayout(files_group)

        self.export_stl = QCheckBox("3D模型 (STL)")
        self.export_stl.setChecked(True)
        files_layout.addWidget(self.export_stl)

        self.export_groove_stl = QCheckBox("带凹槽的3D模型 (STL)")
        self.export_groove_stl.setChecked(True)
        files_layout.addWidget(self.export_groove_stl)

        self.export_dxf = QCheckBox("2D路径 (DXF)")
        self.export_dxf.setChecked(True)
        files_layout.addWidget(self.export_dxf)

        self.export_svg = QCheckBox("2D路径 (SVG)")
        self.export_svg.setChecked(False)
        files_layout.addWidget(self.export_svg)

        self.export_project = QCheckBox("项目配置 (JSON)")
        self.export_project.setChecked(True)
        files_layout.addWidget(self.export_project)

        layout.addWidget(files_group)

        # 一键导出按钮
        self.btn_export_all = QPushButton("一键保存所有文件")
        self.btn_export_all.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.btn_export_all.clicked.connect(self._on_export_all_clicked)
        layout.addWidget(self.btn_export_all)

        layout.addStretch()
        return widget

    def _on_auto_width_changed(self, state: int):
        """自动宽度复选框变更"""
        enabled = state != Qt.Checked
        self.min_width_spin.setEnabled(enabled)
        self.max_width_spin.setEnabled(enabled)
        self._on_params_changed()

    def _on_params_changed(self):
        """参数变更"""
        self.groove_params.width = self.width_spin.value()
        self.groove_params.depth = self.depth_spin.value()
        self.groove_params.auto_width = self.auto_width_check.isChecked()
        self.groove_params.min_width = self.min_width_spin.value()
        self.groove_params.max_width = self.max_width_spin.value()

        self.groove_params_changed.emit(self.groove_params)

    def _on_browse_clicked(self):
        """浏览按钮点击"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录",
            self.export_params.output_dir
        )
        if dir_path:
            self.export_params.output_dir = dir_path
            self.dir_edit.setText(dir_path)

    def _on_export_all_clicked(self):
        """一键导出按钮点击"""
        if not self.export_params.output_dir:
            QMessageBox.warning(self, "警告", "请先选择输出目录")
            self._on_browse_clicked()
            if not self.export_params.output_dir:
                return

        self.export_all_clicked.emit()

    def get_groove_params(self) -> GrooveParams:
        """获取凹槽参数"""
        return self.groove_params

    def get_export_params(self) -> ExportParams:
        """获取导出参数"""
        self.export_params.scale = self.scale_spin.value()
        return self.export_params

    def set_progress(self, value: int, text: str = ""):
        """设置进度"""
        self.progress.setVisible(value > 0 and value < 100)
        self.progress.setValue(value)
        if text:
            self.status_label.setText(text)

    def set_distortion_info(self, avg_distortion: float, max_distortion: float):
        """设置变形信息"""
        self.distortion_label.setText(
            f"变形量: 平均 {avg_distortion:.4f} | 最大 {max_distortion:.4f}"
        )
