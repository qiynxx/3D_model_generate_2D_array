"""参数设置面板"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QGroupBox, QPushButton, QTabWidget, QSlider,
    QProgressBar, QFileDialog, QMessageBox, QScrollArea, QLineEdit,
    QComboBox
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
    conform_to_surface: bool = True  # 曲面贴合模式
    extension_height: float = 0.5  # 向表面外延伸高度
    # 方形焊盘参数
    pad_enabled: bool = True  # 是否生成方形焊盘
    pad_length: float = 3.0   # 方形长度（沿路径方向）
    pad_width: float = 2.0    # 方形宽度（垂直于路径）


@dataclass
class ExportParams:
    """导出参数"""
    scale: float = 1.0
    output_dir: str = ""


@dataclass
class FPCParams:
    """FPC布局参数"""
    groove_width: float = 1.0  # 走线宽度（与凹槽宽度同步）
    pad_radius: float = 2.0  # IR点焊盘半径
    center_pad_radius: float = 3.0  # 中心焊盘半径


class ParamPanel(QWidget):
    """参数设置面板"""

    # 信号
    groove_params_changed = pyqtSignal(object)  # GrooveParams
    generate_grooves_clicked = pyqtSignal()
    flatten_clicked = pyqtSignal()
    generate_fpc_layout_clicked = pyqtSignal()  # 新增：生成FPC布局图
    validate_flatten_clicked = pyqtSignal()  # 新增：验证展平精度
    play_fitting_animation_clicked = pyqtSignal()  # 新增：播放FPC贴合动画
    export_all_clicked = pyqtSignal()
    # 单独导出信号
    export_single_clicked = pyqtSignal(str)  # 导出类型

    def __init__(self, parent=None):
        super().__init__(parent)

        self.groove_params = GrooveParams()
        self.export_params = ExportParams()
        self.fpc_params = FPCParams()

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
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 内容容器
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(8)

        # 宽度设置
        width_group = QGroupBox("宽度设置")
        width_layout = QVBoxLayout(width_group)
        width_layout.setSpacing(4)

        # 自动宽度
        self.auto_width_check = QCheckBox("自动调整宽度")
        self.auto_width_check.setChecked(True)
        self.auto_width_check.stateChanged.connect(self._on_auto_width_changed)
        width_layout.addWidget(self.auto_width_check)

        # 基础宽度 - 使用滑块
        self.width_label = QLabel("基础宽度: 1.0 mm")
        width_layout.addWidget(self.width_label)
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 100)  # 0.1 - 10.0 mm
        self.width_slider.setValue(10)
        self.width_slider.valueChanged.connect(self._on_width_slider_changed)
        width_layout.addWidget(self.width_slider)
        # 保留隐藏的spin用于存储值
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 10.0)
        self.width_spin.setValue(1.0)
        self.width_spin.setVisible(False)

        # 宽度范围 - 紧凑布局
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("范围:"))
        self.min_width_spin = QDoubleSpinBox()
        self.min_width_spin.setRange(0.1, 5.0)
        self.min_width_spin.setValue(0.5)
        self.min_width_spin.setFixedWidth(60)
        self.min_width_spin.valueChanged.connect(self._on_params_changed)
        range_layout.addWidget(self.min_width_spin)
        range_layout.addWidget(QLabel("-"))
        self.max_width_spin = QDoubleSpinBox()
        self.max_width_spin.setRange(0.5, 20.0)
        self.max_width_spin.setValue(5.0)
        self.max_width_spin.setFixedWidth(60)
        self.max_width_spin.valueChanged.connect(self._on_params_changed)
        range_layout.addWidget(self.max_width_spin)
        range_layout.addWidget(QLabel("mm"))
        range_layout.addStretch()
        width_layout.addLayout(range_layout)

        layout.addWidget(width_group)

        # 深度设置
        depth_group = QGroupBox("深度设置")
        depth_layout = QVBoxLayout(depth_group)
        depth_layout.setSpacing(4)

        # 凹槽深度 - 滑块
        self.depth_label = QLabel("凹槽深度: 0.5 mm")
        depth_layout.addWidget(self.depth_label)
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 200)  # 0.1 - 20.0 mm
        self.depth_slider.setValue(5)
        self.depth_slider.valueChanged.connect(self._on_depth_slider_changed)
        depth_layout.addWidget(self.depth_slider)
        # 保留隐藏的spin
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 20.0)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setVisible(False)

        # 向外延伸 - 滑块
        self.ext_height_label = QLabel("向外延伸: 0.5 mm")
        depth_layout.addWidget(self.ext_height_label)
        self.ext_height_slider = QSlider(Qt.Horizontal)
        self.ext_height_slider.setRange(1, 50)  # 0.01 - 5.0 mm
        self.ext_height_slider.setValue(5)
        self.ext_height_slider.valueChanged.connect(self._on_ext_height_slider_changed)
        depth_layout.addWidget(self.ext_height_slider)
        # 保留隐藏的spin
        self.ext_height_spin = QDoubleSpinBox()
        self.ext_height_spin.setRange(0.01, 5.0)
        self.ext_height_spin.setValue(0.5)
        self.ext_height_spin.setVisible(False)

        layout.addWidget(depth_group)

        # 曲面贴合设置
        self.conform_surface_check = QCheckBox("启用曲面贴合模式")
        self.conform_surface_check.setChecked(True)
        self.conform_surface_check.setToolTip("凹槽边界精确贴合曲面")
        self.conform_surface_check.stateChanged.connect(self._on_params_changed)
        layout.addWidget(self.conform_surface_check)

        # 方形焊盘设置
        pad_group = QGroupBox("方形焊盘凹槽")
        pad_layout = QVBoxLayout(pad_group)
        pad_layout.setSpacing(4)

        self.pad_enabled_check = QCheckBox("生成方形焊盘凹槽")
        self.pad_enabled_check.setChecked(True)
        self.pad_enabled_check.stateChanged.connect(self._on_pad_enabled_changed)
        pad_layout.addWidget(self.pad_enabled_check)

        # 焊盘尺寸 - 滑块
        self.pad_length_label = QLabel("长度: 3.0 mm")
        pad_layout.addWidget(self.pad_length_label)
        self.pad_length_slider = QSlider(Qt.Horizontal)
        self.pad_length_slider.setRange(10, 500)  # 1.0 - 50.0 mm
        self.pad_length_slider.setValue(30)
        self.pad_length_slider.valueChanged.connect(self._on_pad_length_slider_changed)
        pad_layout.addWidget(self.pad_length_slider)
        self.pad_length_spin = QDoubleSpinBox()
        self.pad_length_spin.setRange(1.0, 50.0)
        self.pad_length_spin.setValue(3.0)
        self.pad_length_spin.setVisible(False)

        self.pad_width_label = QLabel("宽度: 2.0 mm")
        pad_layout.addWidget(self.pad_width_label)
        self.pad_width_slider = QSlider(Qt.Horizontal)
        self.pad_width_slider.setRange(10, 500)  # 1.0 - 50.0 mm
        self.pad_width_slider.setValue(20)
        self.pad_width_slider.valueChanged.connect(self._on_pad_width_slider_changed)
        pad_layout.addWidget(self.pad_width_slider)
        self.pad_width_spin = QDoubleSpinBox()
        self.pad_width_spin.setRange(1.0, 50.0)
        self.pad_width_spin.setValue(2.0)
        self.pad_width_spin.setVisible(False)

        layout.addWidget(pad_group)

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

        # 设置滚动区域内容
        scroll_area.setWidget(content_widget)
        return scroll_area

    def _on_width_slider_changed(self, value):
        """宽度滑块变更"""
        width = value / 10.0
        self.width_spin.setValue(width)
        self.width_label.setText(f"基础宽度: {width:.1f} mm")
        self._on_params_changed()

    def _on_depth_slider_changed(self, value):
        """深度滑块变更"""
        depth = value / 10.0
        self.depth_spin.setValue(depth)
        self.depth_label.setText(f"凹槽深度: {depth:.1f} mm")
        self._on_params_changed()

    def _on_ext_height_slider_changed(self, value):
        """延伸高度滑块变更"""
        height = value / 10.0
        self.ext_height_spin.setValue(height)
        self.ext_height_label.setText(f"向外延伸: {height:.1f} mm")
        self._on_params_changed()

    def _on_pad_length_slider_changed(self, value):
        """焊盘长度滑块变更"""
        length = value / 10.0
        self.pad_length_spin.setValue(length)
        self.pad_length_label.setText(f"长度: {length:.1f} mm")
        self._on_params_changed()

    def _on_pad_width_slider_changed(self, value):
        """焊盘宽度滑块变更"""
        width = value / 10.0
        self.pad_width_spin.setValue(width)
        self.pad_width_label.setText(f"宽度: {width:.1f} mm")
        self._on_params_changed()

    def _on_scale_slider_changed(self, value):
        """输出比例滑块变更"""
        scale = value / 10.0
        self.scale_spin.setValue(scale)
        self.scale_label.setText(f"输出比例: {scale:.1f} : 1")

    def _on_pad_radius_slider_changed(self, value):
        """IR焊盘半径滑块变更"""
        radius = value / 10.0
        self.pad_radius_spin.setValue(radius)
        self.pad_radius_label.setText(f"IR焊盘半径: {radius:.1f} mm")
        self._on_fpc_params_changed()

    def _on_center_pad_slider_changed(self, value):
        """中心焊盘半径滑块变更"""
        radius = value / 10.0
        self.center_pad_spin.setValue(radius)
        self.center_pad_label.setText(f"中心焊盘半径: {radius:.1f} mm")
        self._on_fpc_params_changed()

    def _create_flatten_tab(self) -> QWidget:
        """创建展开设置选项卡"""
        # 使用滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(8)

        # 展开方法
        method_group = QGroupBox("展开算法")
        method_layout = QVBoxLayout(method_group)

        self.lscm_radio = QCheckBox("LSCM (最小二乘共形映射)")
        self.lscm_radio.setChecked(True)
        method_layout.addWidget(self.lscm_radio)

        method_layout.addWidget(QLabel("保持角度关系，适合FPC制造"))

        layout.addWidget(method_group)

        # 缩放设置 - 使用滑块
        scale_group = QGroupBox("缩放设置")
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.setSpacing(4)

        self.scale_label = QLabel("输出比例: 1.0 : 1")
        scale_layout.addWidget(self.scale_label)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 100)  # 0.1 - 10.0
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)
        scale_layout.addWidget(self.scale_slider)
        # 隐藏的spin存储值
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setVisible(False)

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

        # FPC布局设置 - 使用滑块
        fpc_group = QGroupBox("FPC图纸设置")
        fpc_layout = QVBoxLayout(fpc_group)
        fpc_layout.setSpacing(4)

        # 走线宽度说明 - 使用凹槽宽度
        fpc_width_info = QLabel("走线宽度: 使用凹槽宽度设置")
        fpc_width_info.setStyleSheet("color: #888; font-style: italic;")
        fpc_layout.addWidget(fpc_width_info)

        # IR焊盘半径 - 滑块
        self.pad_radius_label = QLabel("IR焊盘半径: 2.0 mm")
        fpc_layout.addWidget(self.pad_radius_label)
        self.pad_radius_slider = QSlider(Qt.Horizontal)
        self.pad_radius_slider.setRange(5, 100)  # 0.5 - 10.0 mm
        self.pad_radius_slider.setValue(20)
        self.pad_radius_slider.valueChanged.connect(self._on_pad_radius_slider_changed)
        fpc_layout.addWidget(self.pad_radius_slider)
        # 隐藏的spin
        self.pad_radius_spin = QDoubleSpinBox()
        self.pad_radius_spin.setRange(0.5, 10.0)
        self.pad_radius_spin.setValue(2.0)
        self.pad_radius_spin.setVisible(False)

        # 中心焊盘半径 - 滑块
        self.center_pad_label = QLabel("中心焊盘半径: 3.0 mm")
        fpc_layout.addWidget(self.center_pad_label)
        self.center_pad_slider = QSlider(Qt.Horizontal)
        self.center_pad_slider.setRange(5, 150)  # 0.5 - 15.0 mm
        self.center_pad_slider.setValue(30)
        self.center_pad_slider.valueChanged.connect(self._on_center_pad_slider_changed)
        fpc_layout.addWidget(self.center_pad_slider)
        # 隐藏的spin
        self.center_pad_spin = QDoubleSpinBox()
        self.center_pad_spin.setRange(0.5, 15.0)
        self.center_pad_spin.setValue(3.0)
        self.center_pad_spin.setVisible(False)

        layout.addWidget(fpc_group)

        # 生成FPC图纸按钮
        self.btn_generate_fpc = QPushButton("生成FPC图纸")
        self.btn_generate_fpc.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
        """)
        self.btn_generate_fpc.clicked.connect(lambda: self.generate_fpc_layout_clicked.emit())
        layout.addWidget(self.btn_generate_fpc)

        # 精度验证和动画仿真
        validation_group = QGroupBox("精度验证与动画仿真")
        validation_layout = QVBoxLayout(validation_group)

        # 验证按钮
        self.btn_validate = QPushButton("验证展平精度")
        self.btn_validate.setStyleSheet("""
            QPushButton {
                background-color: #16a085;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1abc9c;
            }
        """)
        self.btn_validate.setToolTip(
            "验证3D路径展开为2D后的精度，\n"
            "检查长度保持和往返映射误差"
        )
        self.btn_validate.clicked.connect(lambda: self.validate_flatten_clicked.emit())
        validation_layout.addWidget(self.btn_validate)

        # 动画仿真按钮
        self.btn_animation = QPushButton("播放FPC贴合动画")
        self.btn_animation.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        self.btn_animation.setToolTip(
            "播放2D FPC逐渐贴合到3D模型表面的动画，\n"
            "用于在3D打印前验证贴合效果"
        )
        self.btn_animation.clicked.connect(lambda: self.play_fitting_animation_clicked.emit())
        validation_layout.addWidget(self.btn_animation)

        validation_info = QLabel("验证展平精度，减少3D打印验证工作量")
        validation_info.setStyleSheet("color: gray; font-size: 10px;")
        validation_layout.addWidget(validation_info)

        layout.addWidget(validation_group)

        # 变形信息
        self.distortion_label = QLabel("变形量: -")
        layout.addWidget(self.distortion_label)

        layout.addStretch()

        scroll_area.setWidget(content_widget)
        return scroll_area

    def _create_export_tab(self) -> QWidget:
        """创建导出设置选项卡"""
        # 使用滚动区域支持更多内容
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # 项目名称
        name_group = QGroupBox("项目名称")
        name_layout = QHBoxLayout(name_group)
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("输入项目名称")
        self.project_name_edit.setText("ir_array_export")
        name_layout.addWidget(self.project_name_edit)
        layout.addWidget(name_group)

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

        # 3D模型文件
        mesh_group = QGroupBox("3D模型")
        mesh_layout = QVBoxLayout(mesh_group)

        stl_row = QHBoxLayout()
        self.export_stl = QCheckBox("原始模型 (STL)")
        self.export_stl.setChecked(True)
        stl_row.addWidget(self.export_stl)
        self.btn_save_stl = QPushButton("单独保存")
        self.btn_save_stl.setFixedWidth(80)
        self.btn_save_stl.clicked.connect(lambda: self._on_single_export("stl"))
        stl_row.addWidget(self.btn_save_stl)
        mesh_layout.addLayout(stl_row)

        groove_stl_row = QHBoxLayout()
        self.export_groove_stl = QCheckBox("带凹槽模型 (STL)")
        self.export_groove_stl.setChecked(True)
        groove_stl_row.addWidget(self.export_groove_stl)
        self.btn_save_groove_stl = QPushButton("单独保存")
        self.btn_save_groove_stl.setFixedWidth(80)
        self.btn_save_groove_stl.clicked.connect(lambda: self._on_single_export("groove_stl"))
        groove_stl_row.addWidget(self.btn_save_groove_stl)
        mesh_layout.addLayout(groove_stl_row)

        layout.addWidget(mesh_group)

        # 2D图纸
        drawing_group = QGroupBox("2D图纸")
        drawing_layout = QVBoxLayout(drawing_group)

        dxf_row = QHBoxLayout()
        self.export_dxf = QCheckBox("2D路径 (DXF)")
        self.export_dxf.setChecked(True)
        dxf_row.addWidget(self.export_dxf)
        self.btn_save_dxf = QPushButton("单独保存")
        self.btn_save_dxf.setFixedWidth(80)
        self.btn_save_dxf.clicked.connect(lambda: self._on_single_export("dxf"))
        dxf_row.addWidget(self.btn_save_dxf)
        drawing_layout.addLayout(dxf_row)

        svg_row = QHBoxLayout()
        self.export_svg = QCheckBox("2D路径 (SVG)")
        self.export_svg.setChecked(False)
        svg_row.addWidget(self.export_svg)
        self.btn_save_svg = QPushButton("单独保存")
        self.btn_save_svg.setFixedWidth(80)
        self.btn_save_svg.clicked.connect(lambda: self._on_single_export("svg"))
        svg_row.addWidget(self.btn_save_svg)
        drawing_layout.addLayout(svg_row)

        fpc_dxf_row = QHBoxLayout()
        self.export_fpc_dxf = QCheckBox("FPC图纸 (DXF)")
        self.export_fpc_dxf.setChecked(True)
        fpc_dxf_row.addWidget(self.export_fpc_dxf)
        self.btn_save_fpc_dxf = QPushButton("单独保存")
        self.btn_save_fpc_dxf.setFixedWidth(80)
        self.btn_save_fpc_dxf.clicked.connect(lambda: self._on_single_export("fpc_dxf"))
        fpc_dxf_row.addWidget(self.btn_save_fpc_dxf)
        drawing_layout.addLayout(fpc_dxf_row)

        fpc_svg_row = QHBoxLayout()
        self.export_fpc_svg = QCheckBox("FPC图纸 (SVG)")
        self.export_fpc_svg.setChecked(True)
        fpc_svg_row.addWidget(self.export_fpc_svg)
        self.btn_save_fpc_svg = QPushButton("单独保存")
        self.btn_save_fpc_svg.setFixedWidth(80)
        self.btn_save_fpc_svg.clicked.connect(lambda: self._on_single_export("fpc_svg"))
        fpc_svg_row.addWidget(self.btn_save_fpc_svg)
        drawing_layout.addLayout(fpc_svg_row)

        # FPC单一轮廓DXF（无交叉线）
        fpc_outline_row = QHBoxLayout()
        self.export_fpc_outline = QCheckBox("FPC轮廓 (DXF)")
        self.export_fpc_outline.setChecked(False)
        self.export_fpc_outline.setToolTip("导出合并后的单一封闭轮廓，无交叉线")
        fpc_outline_row.addWidget(self.export_fpc_outline)
        self.btn_save_fpc_outline = QPushButton("单独保存")
        self.btn_save_fpc_outline.setFixedWidth(80)
        self.btn_save_fpc_outline.clicked.connect(lambda: self._on_single_export("fpc_outline_dxf"))
        fpc_outline_row.addWidget(self.btn_save_fpc_outline)
        drawing_layout.addLayout(fpc_outline_row)

        layout.addWidget(drawing_group)

        # 坐标文件
        coord_group = QGroupBox("坐标文件")
        coord_layout = QVBoxLayout(coord_group)

        world_json_row = QHBoxLayout()
        self.export_world_json = QCheckBox("世界坐标 (JSON)")
        self.export_world_json.setChecked(True)
        world_json_row.addWidget(self.export_world_json)
        self.btn_save_world_json = QPushButton("单独保存")
        self.btn_save_world_json.setFixedWidth(80)
        self.btn_save_world_json.clicked.connect(lambda: self._on_single_export("world_json"))
        world_json_row.addWidget(self.btn_save_world_json)
        coord_layout.addLayout(world_json_row)

        world_csv_row = QHBoxLayout()
        self.export_world_csv = QCheckBox("世界坐标 (CSV)")
        self.export_world_csv.setChecked(True)
        world_csv_row.addWidget(self.export_world_csv)
        self.btn_save_world_csv = QPushButton("单独保存")
        self.btn_save_world_csv.setFixedWidth(80)
        self.btn_save_world_csv.clicked.connect(lambda: self._on_single_export("world_csv"))
        world_csv_row.addWidget(self.btn_save_world_csv)
        coord_layout.addLayout(world_csv_row)

        local_json_row = QHBoxLayout()
        self.export_local_json = QCheckBox("局部坐标 (JSON)")
        self.export_local_json.setChecked(True)
        local_json_row.addWidget(self.export_local_json)
        self.btn_save_local_json = QPushButton("单独保存")
        self.btn_save_local_json.setFixedWidth(80)
        self.btn_save_local_json.clicked.connect(lambda: self._on_single_export("local_json"))
        local_json_row.addWidget(self.btn_save_local_json)
        coord_layout.addLayout(local_json_row)

        local_csv_row = QHBoxLayout()
        self.export_local_csv = QCheckBox("局部坐标 (CSV)")
        self.export_local_csv.setChecked(True)
        local_csv_row.addWidget(self.export_local_csv)
        self.btn_save_local_csv = QPushButton("单独保存")
        self.btn_save_local_csv.setFixedWidth(80)
        self.btn_save_local_csv.clicked.connect(lambda: self._on_single_export("local_csv"))
        local_csv_row.addWidget(self.btn_save_local_csv)
        coord_layout.addLayout(local_csv_row)

        layout.addWidget(coord_group)

        # 配置文件
        config_group = QGroupBox("配置文件")
        config_layout = QVBoxLayout(config_group)

        project_row = QHBoxLayout()
        self.export_project = QCheckBox("项目配置 (JSON)")
        self.export_project.setChecked(True)
        project_row.addWidget(self.export_project)
        self.btn_save_project = QPushButton("单独保存")
        self.btn_save_project.setFixedWidth(80)
        self.btn_save_project.clicked.connect(lambda: self._on_single_export("project"))
        project_row.addWidget(self.btn_save_project)
        config_layout.addLayout(project_row)

        layout.addWidget(config_group)

        # 一键导出按钮
        self.btn_export_all = QPushButton("一键保存所有文件到项目文件夹")
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

        # 提示信息
        hint_label = QLabel("一键保存将创建以项目名称命名的文件夹，包含所有选中的文件")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        layout.addStretch()

        scroll_area.setWidget(content_widget)
        return scroll_area

    def _on_auto_width_changed(self, state: int):
        """自动宽度复选框变更"""
        enabled = state != Qt.Checked
        self.min_width_spin.setEnabled(enabled)
        self.max_width_spin.setEnabled(enabled)
        self._on_params_changed()

    def _on_pad_enabled_changed(self, state: int):
        """方形焊盘启用状态变更"""
        enabled = state == Qt.Checked
        self.pad_length_spin.setEnabled(enabled)
        self.pad_width_spin.setEnabled(enabled)
        self._on_params_changed()

    def _on_params_changed(self):
        """参数变更"""
        self.groove_params.width = self.width_spin.value()
        self.groove_params.depth = self.depth_spin.value()
        self.groove_params.auto_width = self.auto_width_check.isChecked()
        self.groove_params.min_width = self.min_width_spin.value()
        self.groove_params.max_width = self.max_width_spin.value()
        self.groove_params.conform_to_surface = self.conform_surface_check.isChecked()
        self.groove_params.extension_height = self.ext_height_spin.value()
        # 方形焊盘参数
        self.groove_params.pad_enabled = self.pad_enabled_check.isChecked()
        self.groove_params.pad_length = self.pad_length_spin.value()
        self.groove_params.pad_width = self.pad_width_spin.value()

        self.groove_params_changed.emit(self.groove_params)

    def _on_fpc_params_changed(self):
        """FPC参数变更"""
        # 走线宽度使用凹槽宽度设置（保持同步）
        self.fpc_params.groove_width = self.width_spin.value()
        self.fpc_params.pad_radius = self.pad_radius_spin.value()
        self.fpc_params.center_pad_radius = self.center_pad_spin.value()

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

    def _on_single_export(self, export_type: str):
        """单独导出某类型文件"""
        self.export_single_clicked.emit(export_type)

    def get_project_name(self) -> str:
        """获取项目名称"""
        name = self.project_name_edit.text().strip()
        if not name:
            name = "ir_array_export"
        return name

    def get_export_options(self) -> dict:
        """获取导出选项"""
        return {
            'stl': self.export_stl.isChecked(),
            'groove_stl': self.export_groove_stl.isChecked(),
            'dxf': self.export_dxf.isChecked(),
            'svg': self.export_svg.isChecked(),
            'fpc_dxf': self.export_fpc_dxf.isChecked(),
            'fpc_svg': self.export_fpc_svg.isChecked(),
            'fpc_outline_dxf': self.export_fpc_outline.isChecked(),
            'world_json': self.export_world_json.isChecked(),
            'world_csv': self.export_world_csv.isChecked(),
            'local_json': self.export_local_json.isChecked(),
            'local_csv': self.export_local_csv.isChecked(),
            'project': self.export_project.isChecked()
        }

    def get_groove_params(self) -> GrooveParams:
        """获取凹槽参数"""
        # 手动同步所有参数值（避免信号循环）
        self.groove_params.width = self.width_spin.value()
        self.groove_params.depth = self.depth_spin.value()
        self.groove_params.auto_width = self.auto_width_check.isChecked()
        self.groove_params.min_width = self.min_width_spin.value()
        self.groove_params.max_width = self.max_width_spin.value()
        self.groove_params.conform_to_surface = self.conform_surface_check.isChecked()
        self.groove_params.extension_height = self.ext_height_spin.value()
        self.groove_params.pad_enabled = self.pad_enabled_check.isChecked()
        self.groove_params.pad_length = self.pad_length_spin.value()
        self.groove_params.pad_width = self.pad_width_spin.value()
        return self.groove_params

    def get_export_params(self) -> ExportParams:
        """获取导出参数"""
        self.export_params.scale = self.scale_spin.value()
        return self.export_params

    def get_fpc_params(self) -> FPCParams:
        """获取FPC布局参数"""
        self._on_fpc_params_changed()
        return self.fpc_params

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
