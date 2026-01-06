"""点位管理面板"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QLineEdit, QComboBox, QMessageBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import Dict, Optional
import numpy as np


class PointPanel(QWidget):
    """IR点位管理面板"""

    # 信号
    point_selected = pyqtSignal(str)  # 选中的点ID
    point_deleted = pyqtSignal(str)  # 删除的点ID
    center_changed = pyqtSignal(str)  # 中心点变更
    picking_toggled = pyqtSignal(bool)  # 拾取模式切换
    paths_requested = pyqtSignal()  # 请求生成路径
    group_changed = pyqtSignal(str, str)  # (point_id, group_name)
    pad_size_changed = pyqtSignal(str, float, float)  # (point_id, length, width)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.points: Dict[str, dict] = {}  # id -> point_info
        self.current_center_id: Optional[str] = None
        self.origin_point_id: Optional[str] = None  # 坐标原点ID

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("IR点位管理")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # 工具栏
        toolbar = QHBoxLayout()

        self.btn_add = QPushButton("添加点")
        self.btn_add.setCheckable(True)
        self.btn_add.clicked.connect(self._on_add_clicked)
        toolbar.addWidget(self.btn_add)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        toolbar.addWidget(self.btn_delete)

        layout.addLayout(toolbar)

        # 点列表
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # 点信息
        info_group = QGroupBox("点信息")
        info_layout = QVBoxLayout(info_group)

        # 名称
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_changed)
        name_layout.addWidget(self.name_edit)
        info_layout.addLayout(name_layout)

        # 分组
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("分组:"))
        self.group_combo = QComboBox()
        self.group_combo.setEditable(True)
        self.group_combo.addItems(["default", "group_1", "group_2"])
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        group_layout.addWidget(self.group_combo)
        info_layout.addLayout(group_layout)

        # 坐标显示
        self.coord_label = QLabel("坐标: -")
        info_layout.addWidget(self.coord_label)

        # 设为中心按钮
        self.btn_set_center = QPushButton("设为中心连接点")
        self.btn_set_center.clicked.connect(self._on_set_center_clicked)
        info_layout.addWidget(self.btn_set_center)

        # 焊盘尺寸设置
        pad_group = QGroupBox("焊盘尺寸")
        pad_layout = QVBoxLayout(pad_group)

        # 焊盘长度
        pad_length_layout = QHBoxLayout()
        pad_length_layout.addWidget(QLabel("长度 (mm):"))
        self.pad_length_spin = QDoubleSpinBox()
        self.pad_length_spin.setRange(1.0, 20.0)
        self.pad_length_spin.setValue(3.0)
        self.pad_length_spin.setSingleStep(0.5)
        self.pad_length_spin.valueChanged.connect(self._on_pad_size_changed)
        pad_length_layout.addWidget(self.pad_length_spin)
        pad_layout.addLayout(pad_length_layout)

        # 焊盘宽度
        pad_width_layout = QHBoxLayout()
        pad_width_layout.addWidget(QLabel("宽度 (mm):"))
        self.pad_width_spin = QDoubleSpinBox()
        self.pad_width_spin.setRange(1.0, 20.0)
        self.pad_width_spin.setValue(2.0)
        self.pad_width_spin.setSingleStep(0.5)
        self.pad_width_spin.valueChanged.connect(self._on_pad_size_changed)
        pad_width_layout.addWidget(self.pad_width_spin)
        pad_layout.addLayout(pad_width_layout)

        info_layout.addWidget(pad_group)

        layout.addWidget(info_group)

        # 生成路径按钮
        self.btn_generate = QPushButton("生成连接路径")
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.btn_generate.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.btn_generate)

        # 统计信息
        self.stats_label = QLabel("点数: 0 | 中心点: 未设置")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def add_point(
        self,
        point_id: str,
        position: np.ndarray,
        name: str = "",
        is_center: bool = False,
        group: str = "default",
        pad_length: float = 3.0,
        pad_width: float = 2.0
    ):
        """添加点到列表"""
        self.points[point_id] = {
            'id': point_id,
            'position': position,
            'name': name if name else f"IR_{point_id[:4]}",
            'is_center': is_center,
            'group': group,
            'pad_length': pad_length,
            'pad_width': pad_width
        }

        item = QListWidgetItem()
        self._update_item_text(item, point_id)
        item.setData(Qt.UserRole, point_id)
        self.list_widget.addItem(item)

        if is_center:
            self.current_center_id = point_id

        self._update_stats()

    def remove_point(self, point_id: str):
        """从列表移除点"""
        if point_id in self.points:
            del self.points[point_id]

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == point_id:
                self.list_widget.takeItem(i)
                break

        if self.current_center_id == point_id:
            self.current_center_id = None

        self._update_stats()

    def set_center_point(self, point_id: str):
        """设置中心点"""
        # 取消旧的中心点
        if self.current_center_id and self.current_center_id in self.points:
            self.points[self.current_center_id]['is_center'] = False

        self.current_center_id = point_id

        if point_id in self.points:
            self.points[point_id]['is_center'] = True

        # 更新列表显示
        self._refresh_list()
        self._update_stats()

    def get_selected_point_id(self) -> Optional[str]:
        """获取当前选中的点ID"""
        items = self.list_widget.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None

    def select_point_by_id(self, point_id: str):
        """通过ID选中点（从3D视图同步）"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == point_id:
                # 阻止信号循环
                self.list_widget.blockSignals(True)
                self.list_widget.setCurrentItem(item)
                self.list_widget.blockSignals(False)

                # 更新信息显示
                if point_id in self.points:
                    info = self.points[point_id]
                    self.name_edit.setText(info['name'])
                    self.group_combo.setCurrentText(info['group'])
                    pos = info['position']
                    self.coord_label.setText(f"坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

                    # 更新焊盘尺寸显示
                    self.pad_length_spin.blockSignals(True)
                    self.pad_width_spin.blockSignals(True)
                    self.pad_length_spin.setValue(info.get('pad_length', 3.0))
                    self.pad_width_spin.setValue(info.get('pad_width', 2.0))
                    self.pad_length_spin.blockSignals(False)
                    self.pad_width_spin.blockSignals(False)
                break

    def _update_item_text(self, item: QListWidgetItem, point_id: str):
        """更新列表项文本"""
        info = self.points.get(point_id, {})
        name = info.get('name', point_id)
        group = info.get('group', 'default')
        is_center = info.get('is_center', False)
        is_origin = (point_id == self.origin_point_id)

        # 构建前缀标记
        prefix = ""
        if is_origin:
            prefix += "◎ "  # 原点标记
        if is_center:
            prefix += "★ "  # 中心点标记

        text = f"{prefix}{name} [{group}]"
        item.setText(text)

        # 设置颜色（原点用金色，中心点用红色）
        if is_origin:
            item.setForeground(Qt.yellow)  # 坐标原点用黄色
        elif is_center:
            item.setForeground(Qt.red)
        else:
            item.setForeground(Qt.white)

    def _refresh_list(self):
        """刷新列表显示"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            point_id = item.data(Qt.UserRole)
            self._update_item_text(item, point_id)

    def _update_stats(self):
        """更新统计信息"""
        count = len(self.points)
        center_name = "未设置"
        if self.current_center_id and self.current_center_id in self.points:
            center_name = self.points[self.current_center_id]['name']

        # 计算参与路径生成的点数（排除原点）
        active_count = count
        if self.origin_point_id and self.origin_point_id in self.points:
            active_count -= 1

        self.stats_label.setText(f"点数: {count} (路径点: {active_count}) | 中心点: {center_name}")

    def _on_add_clicked(self, checked: bool):
        """添加按钮点击"""
        self.picking_toggled.emit(checked)
        if checked:
            self.btn_add.setText("点击模型添加...")
            self.btn_add.setStyleSheet("background-color: #e74c3c;")
        else:
            self.btn_add.setText("添加点")
            self.btn_add.setStyleSheet("")

    def stop_picking(self):
        """停止拾取模式"""
        self.btn_add.setChecked(False)
        self.btn_add.setText("添加点")
        self.btn_add.setStyleSheet("")

    def _on_delete_clicked(self):
        """删除按钮点击"""
        point_id = self.get_selected_point_id()
        if point_id:
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除点 {self.points[point_id]['name']} 吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.point_deleted.emit(point_id)

    def _on_selection_changed(self):
        """选择变更"""
        point_id = self.get_selected_point_id()
        if point_id and point_id in self.points:
            info = self.points[point_id]
            self.name_edit.setText(info['name'])
            self.group_combo.setCurrentText(info['group'])
            pos = info['position']
            self.coord_label.setText(f"坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

            # 更新焊盘尺寸显示
            self.pad_length_spin.blockSignals(True)
            self.pad_width_spin.blockSignals(True)
            self.pad_length_spin.setValue(info.get('pad_length', 3.0))
            self.pad_width_spin.setValue(info.get('pad_width', 2.0))
            self.pad_length_spin.blockSignals(False)
            self.pad_width_spin.blockSignals(False)

            self.point_selected.emit(point_id)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击设置为中心点"""
        point_id = item.data(Qt.UserRole)
        if point_id:
            self._set_as_center(point_id)

    def _on_set_center_clicked(self):
        """设为中心按钮点击"""
        point_id = self.get_selected_point_id()
        if point_id:
            self._set_as_center(point_id)

    def _set_as_center(self, point_id: str):
        """设置为中心点"""
        self.set_center_point(point_id)
        self.center_changed.emit(point_id)

    def _on_name_changed(self):
        """名称变更"""
        point_id = self.get_selected_point_id()
        if point_id and point_id in self.points:
            self.points[point_id]['name'] = self.name_edit.text()
            self._refresh_list()

    def _on_group_changed(self, group: str):
        """分组变更"""
        point_id = self.get_selected_point_id()
        if point_id and point_id in self.points:
            self.points[point_id]['group'] = group
            self._refresh_list()
            self.group_changed.emit(point_id, group)

    def _on_pad_size_changed(self):
        """焊盘尺寸变更"""
        point_id = self.get_selected_point_id()
        if point_id and point_id in self.points:
            length = self.pad_length_spin.value()
            width = self.pad_width_spin.value()
            self.points[point_id]['pad_length'] = length
            self.points[point_id]['pad_width'] = width
            self.pad_size_changed.emit(point_id, length, width)

    def _on_generate_clicked(self):
        """生成路径按钮点击"""
        if not self.current_center_id:
            QMessageBox.warning(self, "警告", "请先设置中心连接点")
            return
        if len(self.points) < 2:
            QMessageBox.warning(self, "警告", "至少需要2个点才能生成路径")
            return
        self.paths_requested.emit()

    def clear_all(self):
        """清除所有点"""
        self.points.clear()
        self.list_widget.clear()
        self.current_center_id = None
        self._update_stats()

    def get_point_pad_sizes(self) -> Dict[str, tuple]:
        """获取所有点的焊盘尺寸

        Returns:
            Dict[str, tuple]: {point_id: (pad_length, pad_width)}
        """
        result = {}
        for point_id, info in self.points.items():
            result[point_id] = (
                info.get('pad_length', 3.0),
                info.get('pad_width', 2.0)
            )
        return result

    def set_origin_point(self, point_id: Optional[str]):
        """设置坐标原点

        Args:
            point_id: 原点ID，None表示清除原点
        """
        self.origin_point_id = point_id
        self._refresh_list()
        self._update_stats()
