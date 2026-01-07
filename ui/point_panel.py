"""点位管理面板"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QLineEdit, QComboBox, QMessageBox,
    QDoubleSpinBox, QAbstractItemView
)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import Dict, Optional, List, Tuple
from enum import Enum
import numpy as np


class PathMode(Enum):
    """路径生成模式"""
    STAR = "star"       # 星形连接：中心点到各个IR点
    SERIAL = "serial"   # 串联连接：一根线连接所有点


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
    path_mode_changed = pyqtSignal(str)  # 路径模式变更
    serial_order_changed = pyqtSignal(list)  # 串联顺序变更

    def __init__(self, parent=None):
        super().__init__(parent)

        self.points: Dict[str, dict] = {}  # id -> point_info
        self.current_center_id: Optional[str] = None
        self.origin_point_id: Optional[str] = None  # 坐标原点ID
        self.current_path_mode: str = PathMode.STAR.value  # 当前路径模式

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

        # 路径模式设置
        path_mode_group = QGroupBox("路径模式")
        path_mode_layout = QVBoxLayout(path_mode_group)

        self.path_mode_combo = QComboBox()
        self.path_mode_combo.addItem("星形连接 (中心点到各IR点)", PathMode.STAR.value)
        self.path_mode_combo.addItem("串联连接 (一根线连接所有点)", PathMode.SERIAL.value)
        self.path_mode_combo.currentIndexChanged.connect(self._on_path_mode_changed)
        path_mode_layout.addWidget(self.path_mode_combo)

        # 设为中心按钮（仅星形模式显示）
        self.btn_set_center = QPushButton("设为中心连接点")
        self.btn_set_center.clicked.connect(self._on_set_center_clicked)
        path_mode_layout.addWidget(self.btn_set_center)

        layout.addWidget(path_mode_group)

        # 串联排序设置（仅在串联模式下显示）
        self.serial_order_group = QGroupBox("串联顺序 (拖拽排序)")
        serial_order_layout = QVBoxLayout(self.serial_order_group)

        self.serial_order_list = QListWidget()
        self.serial_order_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.serial_order_list.setDefaultDropAction(Qt.MoveAction)
        self.serial_order_list.setMinimumHeight(80)
        self.serial_order_list.setMaximumHeight(120)
        self.serial_order_list.model().rowsMoved.connect(self._on_serial_order_changed)
        serial_order_layout.addWidget(self.serial_order_list)

        serial_hint = QLabel("拖拽调整点位连接顺序")
        serial_hint.setStyleSheet("color: gray; font-size: 10px;")
        serial_order_layout.addWidget(serial_hint)

        # 默认隐藏，仅在串联模式下显示
        self.serial_order_group.setVisible(False)
        layout.addWidget(self.serial_order_group)

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

        # 计算参与路径生成的点数（排除原点）
        active_count = count
        if self.origin_point_id and self.origin_point_id in self.points:
            active_count -= 1

        # 根据路径模式显示不同信息
        if self.current_path_mode == PathMode.SERIAL.value:
            self.stats_label.setText(f"点数: {count} (路径点: {active_count}) | 模式: 串联")
        else:
            center_name = "未设置"
            if self.current_center_id and self.current_center_id in self.points:
                center_name = self.points[self.current_center_id]['name']
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

    def _on_path_mode_changed(self, index: int):
        """路径模式变更"""
        mode_value = self.path_mode_combo.itemData(index)
        self.current_path_mode = mode_value

        # 串联模式：隐藏设置中心按钮，显示排序列表
        is_serial = (mode_value == PathMode.SERIAL.value)
        self.btn_set_center.setVisible(not is_serial)
        self.serial_order_group.setVisible(is_serial)

        # 更新统计信息
        self._update_stats()

        # 发送信号
        self.path_mode_changed.emit(mode_value)

    def _on_serial_order_changed(self):
        """串联顺序变更（拖拽后触发）"""
        order = self.get_serial_order()
        self.serial_order_changed.emit(order)

    def _on_generate_clicked(self):
        """生成路径按钮点击"""
        # 串联模式不需要中心点
        if self.current_path_mode == PathMode.STAR.value:
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

    def get_path_mode(self) -> str:
        """获取当前路径模式"""
        return self.current_path_mode

    def set_path_mode(self, mode: str):
        """设置路径模式"""
        for i in range(self.path_mode_combo.count()):
            if self.path_mode_combo.itemData(i) == mode:
                self.path_mode_combo.setCurrentIndex(i)
                break

    def update_serial_order_list(self, points: List[Tuple[str, str]]):
        """
        更新串联排序列表

        Args:
            points: [(point_id, point_name), ...] 点列表
        """
        self.serial_order_list.clear()
        for point_id, point_name in points:
            item = QListWidgetItem(point_name)
            item.setData(Qt.UserRole, point_id)
            self.serial_order_list.addItem(item)

    def get_serial_order(self) -> List[str]:
        """获取当前串联顺序（点ID列表）"""
        order = []
        for i in range(self.serial_order_list.count()):
            item = self.serial_order_list.item(i)
            point_id = item.data(Qt.UserRole)
            order.append(point_id)
        return order

    def set_serial_order(self, order: List[str], all_points: List[Tuple[str, str]]):
        """
        设置串联顺序

        Args:
            order: 点ID顺序列表
            all_points: [(point_id, point_name), ...] 所有点列表
        """
        # 创建ID到名称的映射
        id_to_name = {pid: name for pid, name in all_points}

        self.serial_order_list.clear()

        # 按指定顺序添加点
        added = set()
        for point_id in order:
            if point_id in id_to_name and point_id not in added:
                item = QListWidgetItem(id_to_name[point_id])
                item.setData(Qt.UserRole, point_id)
                self.serial_order_list.addItem(item)
                added.add(point_id)

        # 添加不在顺序中的点
        for point_id, point_name in all_points:
            if point_id not in added:
                item = QListWidgetItem(point_name)
                item.setData(Qt.UserRole, point_id)
                self.serial_order_list.addItem(item)
