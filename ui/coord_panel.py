"""坐标系统管理面板"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QComboBox, QCheckBox,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
from typing import Dict, Optional, Tuple
import numpy as np


class CoordPanel(QWidget):
    """坐标系统管理面板"""

    # 信号
    origin_changed = pyqtSignal(str)  # 原点变更 (point_id)
    axis_changed = pyqtSignal()  # 轴方向变更
    show_axes_changed = pyqtSignal(bool)  # 显示/隐藏坐标轴

    def __init__(self, parent=None):
        super().__init__(parent)

        self.origin_point_id: Optional[str] = None
        self.points: Dict[str, dict] = {}  # 引用点位数据

        # 坐标轴方向（欧拉角，度）
        self.axis_rotation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 标题
        title = QLabel("坐标系统")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # 原点设置
        origin_group = QGroupBox("坐标原点")
        origin_layout = QVBoxLayout(origin_group)

        # 原点选择
        origin_select_layout = QHBoxLayout()
        origin_select_layout.addWidget(QLabel("原点:"))
        self.origin_combo = QComboBox()
        self.origin_combo.addItem("未设置", None)
        self.origin_combo.currentIndexChanged.connect(self._on_origin_changed)
        origin_select_layout.addWidget(self.origin_combo)
        origin_layout.addLayout(origin_select_layout)

        # 原点坐标显示
        self.origin_coord_label = QLabel("原点坐标: -")
        origin_layout.addWidget(self.origin_coord_label)

        # 设为当前选中点
        self.btn_set_origin = QPushButton("设为当前选中点")
        self.btn_set_origin.clicked.connect(self._on_set_origin_clicked)
        origin_layout.addWidget(self.btn_set_origin)

        scroll_layout.addWidget(origin_group)

        # 坐标轴设置
        axis_group = QGroupBox("坐标轴方向")
        axis_layout = QVBoxLayout(axis_group)

        # 显示坐标轴
        self.show_axes_check = QCheckBox("显示坐标轴")
        self.show_axes_check.setChecked(True)
        self.show_axes_check.stateChanged.connect(self._on_show_axes_changed)
        axis_layout.addWidget(self.show_axes_check)

        # X轴旋转（Roll）
        roll_layout = QHBoxLayout()
        roll_layout.addWidget(QLabel("绕X轴 (°):"))
        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setValue(0.0)
        self.roll_spin.setSingleStep(5.0)
        self.roll_spin.valueChanged.connect(self._on_axis_rotation_changed)
        roll_layout.addWidget(self.roll_spin)
        axis_layout.addLayout(roll_layout)

        # Y轴旋转（Pitch）
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("绕Y轴 (°):"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-180.0, 180.0)
        self.pitch_spin.setValue(0.0)
        self.pitch_spin.setSingleStep(5.0)
        self.pitch_spin.valueChanged.connect(self._on_axis_rotation_changed)
        pitch_layout.addWidget(self.pitch_spin)
        axis_layout.addLayout(pitch_layout)

        # Z轴旋转（Yaw）
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("绕Z轴 (°):"))
        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-180.0, 180.0)
        self.yaw_spin.setValue(0.0)
        self.yaw_spin.setSingleStep(5.0)
        self.yaw_spin.valueChanged.connect(self._on_axis_rotation_changed)
        yaw_layout.addWidget(self.yaw_spin)
        axis_layout.addLayout(yaw_layout)

        # 重置按钮
        self.btn_reset_axis = QPushButton("重置轴方向")
        self.btn_reset_axis.clicked.connect(self._on_reset_axis)
        axis_layout.addWidget(self.btn_reset_axis)

        # 对齐到曲面法线
        self.btn_align_normal = QPushButton("对齐到原点法线")
        self.btn_align_normal.clicked.connect(self._on_align_to_normal)
        axis_layout.addWidget(self.btn_align_normal)

        scroll_layout.addWidget(axis_group)

        # 坐标轴长度
        length_group = QGroupBox("坐标轴显示")
        length_layout = QVBoxLayout(length_group)

        axis_len_layout = QHBoxLayout()
        axis_len_layout.addWidget(QLabel("轴长度 (mm):"))
        self.axis_length_spin = QDoubleSpinBox()
        self.axis_length_spin.setRange(0.5, 100.0)
        self.axis_length_spin.setValue(3.0)  # 默认值改小
        self.axis_length_spin.setSingleStep(0.5)
        self.axis_length_spin.valueChanged.connect(self._on_axis_changed)
        axis_len_layout.addWidget(self.axis_length_spin)
        length_layout.addLayout(axis_len_layout)

        # 箭头粗细
        shaft_layout = QHBoxLayout()
        shaft_layout.addWidget(QLabel("箭头粗细:"))
        self.shaft_scale_spin = QDoubleSpinBox()
        self.shaft_scale_spin.setRange(0.5, 5.0)
        self.shaft_scale_spin.setValue(1.0)
        self.shaft_scale_spin.setSingleStep(0.1)
        self.shaft_scale_spin.valueChanged.connect(self._on_axis_changed)
        shaft_layout.addWidget(self.shaft_scale_spin)
        length_layout.addLayout(shaft_layout)

        # 箭头头部大小
        tip_layout = QHBoxLayout()
        tip_layout.addWidget(QLabel("箭头头部:"))
        self.tip_scale_spin = QDoubleSpinBox()
        self.tip_scale_spin.setRange(0.5, 5.0)
        self.tip_scale_spin.setValue(1.0)
        self.tip_scale_spin.setSingleStep(0.1)
        self.tip_scale_spin.valueChanged.connect(self._on_axis_changed)
        tip_layout.addWidget(self.tip_scale_spin)
        length_layout.addLayout(tip_layout)

        scroll_layout.addWidget(length_group)

        # 坐标列表显示
        coord_list_group = QGroupBox("点坐标列表")
        coord_list_layout = QVBoxLayout(coord_list_group)

        # 坐标系选择
        coord_type_layout = QHBoxLayout()
        coord_type_layout.addWidget(QLabel("显示:"))
        self.coord_type_combo = QComboBox()
        self.coord_type_combo.addItem("世界坐标", "world")
        self.coord_type_combo.addItem("局部坐标", "local")
        self.coord_type_combo.currentIndexChanged.connect(self._on_coord_type_changed)
        coord_type_layout.addWidget(self.coord_type_combo)
        coord_list_layout.addLayout(coord_type_layout)

        # 坐标表格
        self.coord_table = QTableWidget()
        self.coord_table.setColumnCount(4)
        self.coord_table.setHorizontalHeaderLabels(["名称", "X", "Y", "Z"])
        self.coord_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.coord_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.coord_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.coord_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.coord_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.coord_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.coord_table.setMinimumHeight(150)
        coord_list_layout.addWidget(self.coord_table)

        # 刷新按钮
        self.btn_refresh_coords = QPushButton("刷新坐标")
        self.btn_refresh_coords.clicked.connect(self._refresh_coordinate_table)
        coord_list_layout.addWidget(self.btn_refresh_coords)

        scroll_layout.addWidget(coord_list_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def update_points(self, points: Dict[str, dict]):
        """更新点位列表"""
        self.points = points

        # 保存当前选择
        current_id = self.origin_combo.currentData()

        # 更新下拉框
        self.origin_combo.blockSignals(True)
        self.origin_combo.clear()
        self.origin_combo.addItem("未设置", None)

        for point_id, info in points.items():
            name = info.get('name', point_id[:8])
            self.origin_combo.addItem(name, point_id)

        # 恢复选择
        if current_id:
            idx = self.origin_combo.findData(current_id)
            if idx >= 0:
                self.origin_combo.setCurrentIndex(idx)

        self.origin_combo.blockSignals(False)

        # 自动刷新坐标表格
        self._refresh_coordinate_table()

    def set_origin_by_id(self, point_id: str):
        """通过ID设置原点"""
        idx = self.origin_combo.findData(point_id)
        if idx >= 0:
            self.origin_combo.setCurrentIndex(idx)

    def get_origin_position(self) -> Optional[np.ndarray]:
        """获取原点位置"""
        if self.origin_point_id and self.origin_point_id in self.points:
            return self.points[self.origin_point_id]['position'].copy()
        return None

    def get_rotation_matrix(self) -> np.ndarray:
        """获取旋转矩阵（欧拉角转旋转矩阵）"""
        roll = np.radians(self.roll_spin.value())
        pitch = np.radians(self.pitch_spin.value())
        yaw = np.radians(self.yaw_spin.value())

        # 旋转矩阵 Rz * Ry * Rx
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        Rx = np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ])

        Ry = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])

        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def get_axis_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取坐标轴方向向量"""
        R = self.get_rotation_matrix()
        x_axis = R @ np.array([1, 0, 0])
        y_axis = R @ np.array([0, 1, 0])
        z_axis = R @ np.array([0, 0, 1])
        return x_axis, y_axis, z_axis

    def get_axis_length(self) -> float:
        """获取坐标轴长度"""
        return self.axis_length_spin.value()

    def get_shaft_scale(self) -> float:
        """获取箭头粗细缩放"""
        return self.shaft_scale_spin.value()

    def get_tip_scale(self) -> float:
        """获取箭头头部缩放"""
        return self.tip_scale_spin.value()

    def is_axes_visible(self) -> bool:
        """是否显示坐标轴"""
        return self.show_axes_check.isChecked()

    def transform_to_local(self, position: np.ndarray) -> Optional[np.ndarray]:
        """将世界坐标转换为局部坐标"""
        origin = self.get_origin_position()
        if origin is None:
            return None

        # 平移到原点
        translated = position - origin

        # 旋转（使用逆旋转矩阵）
        R = self.get_rotation_matrix()
        R_inv = R.T  # 正交矩阵的逆等于转置

        local_pos = R_inv @ translated
        return local_pos

    def get_all_local_coordinates(self) -> Dict[str, Tuple[str, np.ndarray]]:
        """获取所有点的局部坐标

        Returns:
            Dict[str, Tuple[str, np.ndarray]]: {point_id: (name, local_position)}
        """
        result = {}

        if not self.origin_point_id:
            return result

        for point_id, info in self.points.items():
            position = info['position']
            local_pos = self.transform_to_local(position)
            if local_pos is not None:
                result[point_id] = (info.get('name', point_id), local_pos)

        return result

    def _on_origin_changed(self, index: int):
        """原点选择变更"""
        point_id = self.origin_combo.currentData()
        self.origin_point_id = point_id

        if point_id and point_id in self.points:
            pos = self.points[point_id]['position']
            self.origin_coord_label.setText(
                f"原点坐标: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
        else:
            self.origin_coord_label.setText("原点坐标: -")

        self.origin_changed.emit(point_id if point_id else "")

        # 刷新坐标表格（原点变化会影响局部坐标）
        self._refresh_coordinate_table()

    def _on_set_origin_clicked(self):
        """设为当前选中点按钮"""
        # 这个信号会被主窗口处理
        pass  # 实际由主窗口实现

    def _on_show_axes_changed(self, state: int):
        """显示坐标轴变更"""
        self.show_axes_changed.emit(state == Qt.Checked)

    def _on_axis_rotation_changed(self):
        """轴旋转变更"""
        self.axis_rotation = np.array([
            self.roll_spin.value(),
            self.pitch_spin.value(),
            self.yaw_spin.value()
        ])
        self.axis_changed.emit()

        # 刷新坐标表格（旋转变化会影响局部坐标）
        self._refresh_coordinate_table()

    def _on_axis_changed(self):
        """轴参数变更"""
        self.axis_changed.emit()

    def _on_reset_axis(self):
        """重置轴方向"""
        self.roll_spin.blockSignals(True)
        self.pitch_spin.blockSignals(True)
        self.yaw_spin.blockSignals(True)

        self.roll_spin.setValue(0.0)
        self.pitch_spin.setValue(0.0)
        self.yaw_spin.setValue(0.0)

        self.roll_spin.blockSignals(False)
        self.pitch_spin.blockSignals(False)
        self.yaw_spin.blockSignals(False)

        self.axis_rotation = np.array([0.0, 0.0, 0.0])
        self.axis_changed.emit()

    def _on_align_to_normal(self):
        """对齐到原点法线"""
        # 这个需要主窗口提供法线信息
        pass  # 实际由主窗口实现

    def set_axis_rotation_from_normal(self, normal: np.ndarray):
        """根据法线设置坐标轴旋转

        使Z轴对齐到法线方向
        """
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # 计算从Z轴(0,0,1)到法线的旋转
        z_axis = np.array([0, 0, 1])

        # 计算旋转轴（叉积）
        rotation_axis = np.cross(z_axis, normal)
        axis_norm = np.linalg.norm(rotation_axis)

        if axis_norm < 1e-6:
            # Z轴和法线平行
            if np.dot(z_axis, normal) > 0:
                # 同向，不需要旋转
                self._on_reset_axis()
            else:
                # 反向，绕X轴旋转180度
                self.roll_spin.setValue(180.0)
                self.pitch_spin.setValue(0.0)
                self.yaw_spin.setValue(0.0)
                self.axis_changed.emit()
            return

        rotation_axis = rotation_axis / axis_norm

        # 计算旋转角（点积）
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1, 1))

        # 转换为欧拉角（简化处理）
        # 使用罗德里格斯公式
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # 从旋转矩阵提取欧拉角
        pitch = np.arcsin(-R[2, 0])
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0

        self.roll_spin.blockSignals(True)
        self.pitch_spin.blockSignals(True)
        self.yaw_spin.blockSignals(True)

        self.roll_spin.setValue(np.degrees(roll))
        self.pitch_spin.setValue(np.degrees(pitch))
        self.yaw_spin.setValue(np.degrees(yaw))

        self.roll_spin.blockSignals(False)
        self.pitch_spin.blockSignals(False)
        self.yaw_spin.blockSignals(False)

        self.axis_changed.emit()

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            'origin_point_id': self.origin_point_id,
            'axis_rotation': self.axis_rotation.tolist(),
            'axis_length': self.axis_length_spin.value(),
            'shaft_scale': self.shaft_scale_spin.value(),
            'tip_scale': self.tip_scale_spin.value(),
            'show_axes': self.show_axes_check.isChecked()
        }

    def from_dict(self, data: dict):
        """从字典导入"""
        if 'origin_point_id' in data:
            self.origin_point_id = data['origin_point_id']
            idx = self.origin_combo.findData(self.origin_point_id)
            if idx >= 0:
                self.origin_combo.setCurrentIndex(idx)

        if 'axis_rotation' in data:
            rot = data['axis_rotation']
            self.roll_spin.setValue(rot[0])
            self.pitch_spin.setValue(rot[1])
            self.yaw_spin.setValue(rot[2])
            self.axis_rotation = np.array(rot)

        if 'axis_length' in data:
            self.axis_length_spin.setValue(data['axis_length'])

        if 'shaft_scale' in data:
            self.shaft_scale_spin.setValue(data['shaft_scale'])

        if 'tip_scale' in data:
            self.tip_scale_spin.setValue(data['tip_scale'])

        if 'show_axes' in data:
            self.show_axes_check.setChecked(data['show_axes'])

    def _on_coord_type_changed(self, index: int):
        """坐标类型切换"""
        self._refresh_coordinate_table()

    def _refresh_coordinate_table(self):
        """刷新坐标表格"""
        self.coord_table.setRowCount(0)

        if not self.points:
            return

        coord_type = self.coord_type_combo.currentData()
        use_local = coord_type == "local" and self.origin_point_id is not None

        for point_id, info in self.points.items():
            row = self.coord_table.rowCount()
            self.coord_table.insertRow(row)

            name = info.get('name', point_id[:8])
            world_pos = info['position']

            if use_local:
                local_pos = self.transform_to_local(world_pos)
                if local_pos is not None:
                    pos = local_pos
                else:
                    pos = world_pos
            else:
                pos = world_pos

            # 名称
            name_item = QTableWidgetItem(name)
            is_origin = point_id == self.origin_point_id
            is_center = info.get('is_center', False)
            if is_origin:
                name_item.setForeground(QColor(255, 215, 0))  # 金色
                name_item.setText(f"◎ {name}")
            elif is_center:
                name_item.setForeground(QColor(255, 100, 100))  # 红色
                name_item.setText(f"★ {name}")
            self.coord_table.setItem(row, 0, name_item)

            # X, Y, Z 坐标
            for col, val in enumerate(pos):
                item = QTableWidgetItem(f"{val:.3f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.coord_table.setItem(row, col + 1, item)
