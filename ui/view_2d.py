"""2D展开视图组件 - 显示展开后的路径"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont, QPolygonF
from typing import Optional, List, Tuple, Dict


class View2D(QWidget):
    """2D路径展开视图"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 路径数据
        self.paths_2d: Dict[str, np.ndarray] = {}  # point_id -> 2D路径
        self.ir_points_2d: Dict[str, Tuple[np.ndarray, str, bool]] = {}  # point_id -> (pos, name, is_center)
        self.center_2d: Optional[np.ndarray] = None
        self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # FPC布局数据
        self.groove_outlines: Dict[str, np.ndarray] = {}  # 凹槽轮廓
        self.ir_pads: Dict[str, np.ndarray] = {}  # IR点焊盘轮廓
        self.center_pad: Optional[np.ndarray] = None  # 中心焊盘轮廓
        self.merged_outline: Optional[np.ndarray] = None  # 合并后的总轮廓

        # 视图参数
        self.view_scale = 1.0
        self.view_offset = np.array([0.0, 0.0])
        self.show_paths = True
        self.show_points = True
        self.show_grid = True
        self.show_groove_outlines = True  # 是否显示凹槽轮廓
        self.show_fpc_layout = False  # 是否显示FPC布局模式

        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #1a1a2e;")

    def set_flatten_result(self, paths_2d: Dict[str, np.ndarray],
                           ir_points_2d: Dict[str, Tuple[np.ndarray, str, bool]],
                           center_2d: np.ndarray,
                           bounds: Tuple[np.ndarray, np.ndarray]):
        """设置展开结果"""
        self.paths_2d = {k: v.copy() for k, v in paths_2d.items()}
        self.ir_points_2d = {k: (v[0].copy(), v[1], v[2]) for k, v in ir_points_2d.items()}
        self.center_2d = center_2d.copy()
        self.bounds = (bounds[0].copy(), bounds[1].copy())

        self._auto_fit()
        self.update()

    def clear(self):
        """清除数据"""
        self.paths_2d = {}
        self.ir_points_2d = {}
        self.center_2d = None
        self.bounds = None
        # 清除FPC布局数据
        self.groove_outlines = {}
        self.ir_pads = {}
        self.center_pad = None
        self.merged_outline = None
        self.show_fpc_layout = False
        self.update()

    def set_fpc_layout(self, groove_outlines: Dict[str, np.ndarray],
                       ir_pads: Dict[str, np.ndarray],
                       center_pad: np.ndarray,
                       merged_outline: Optional[np.ndarray],
                       bounds: Tuple[np.ndarray, np.ndarray]):
        """设置FPC布局数据"""
        self.groove_outlines = {k: v.copy() for k, v in groove_outlines.items()}
        self.ir_pads = {k: v.copy() for k, v in ir_pads.items()}
        self.center_pad = center_pad.copy() if center_pad is not None else None
        self.merged_outline = merged_outline.copy() if merged_outline is not None else None
        self.bounds = (bounds[0].copy(), bounds[1].copy())
        self.show_fpc_layout = True
        self._auto_fit()
        self.update()

    def toggle_fpc_layout(self, show: bool):
        """切换FPC布局显示"""
        self.show_fpc_layout = show
        self.update()

    def _auto_fit(self):
        """自动缩放以适应视图"""
        if self.bounds is None:
            return

        min_xy, max_xy = self.bounds
        data_size = max_xy - min_xy

        if data_size[0] < 1e-6 or data_size[1] < 1e-6:
            return

        # 计算缩放
        widget_size = np.array([self.width() - 60, self.height() - 60])
        scale_x = widget_size[0] / data_size[0]
        scale_y = widget_size[1] / data_size[1]
        self.view_scale = min(scale_x, scale_y) * 0.85

        # 计算偏移使其居中
        center_data = (min_xy + max_xy) / 2
        center_widget = np.array([self.width() / 2, self.height() / 2])
        self.view_offset = center_widget - center_data * self.view_scale

    def _to_screen(self, point_2d: np.ndarray) -> QPointF:
        """2D坐标转屏幕坐标"""
        screen = point_2d * self.view_scale + self.view_offset
        # Y轴翻转
        return QPointF(screen[0], self.height() - screen[1])

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor('#1a1a2e'))

        if not self.paths_2d and not self.groove_outlines:
            # 显示提示
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 12))
            painter.drawText(self.rect(), Qt.AlignCenter,
                           "生成路径后点击\"展开\"按钮\n\n"
                           "路径将展开为2D平面图\n"
                           "保持实际长度")
            return

        # 绘制网格
        if self.show_grid:
            self._draw_grid(painter)

        # 如果是FPC布局模式，绘制凹槽轮廓
        if self.show_fpc_layout and self.groove_outlines:
            self._draw_fpc_layout(painter)
        else:
            # 绘制路径
            if self.show_paths:
                self._draw_paths(painter)

        # 绘制IR点
        if self.show_points:
            self._draw_ir_points(painter)

        # 绘制比例尺
        self._draw_scale_bar(painter)

        # 绘制图例
        self._draw_legend(painter)

    def _draw_grid(self, painter: QPainter):
        """绘制背景网格"""
        if self.bounds is None:
            return

        pen = QPen(QColor(50, 50, 70))
        pen.setWidthF(0.5)
        painter.setPen(pen)

        min_xy, max_xy = self.bounds
        grid_step = 10.0  # 10mm网格

        # 垂直线
        x = np.floor(min_xy[0] / grid_step) * grid_step
        while x <= max_xy[0]:
            p1 = self._to_screen(np.array([x, min_xy[1]]))
            p2 = self._to_screen(np.array([x, max_xy[1]]))
            painter.drawLine(p1, p2)
            x += grid_step

        # 水平线
        y = np.floor(min_xy[1] / grid_step) * grid_step
        while y <= max_xy[1]:
            p1 = self._to_screen(np.array([min_xy[0], y]))
            p2 = self._to_screen(np.array([max_xy[0], y]))
            painter.drawLine(p1, p2)
            y += grid_step

    def _draw_paths(self, painter: QPainter):
        """绘制路径"""
        colors = [
            '#f39c12',  # 橙色
            '#3498db',  # 蓝色
            '#9b59b6',  # 紫色
            '#1abc9c',  # 青色
            '#e91e63',  # 粉色
            '#00bcd4',  # 天蓝
            '#ff5722',  # 深橙
            '#8bc34a',  # 浅绿
        ]

        for idx, (point_id, path) in enumerate(self.paths_2d.items()):
            if len(path) < 2:
                continue

            color = QColor(colors[idx % len(colors)])
            pen = QPen(color)
            pen.setWidthF(2.5)
            painter.setPen(pen)

            qpath = QPainterPath()
            qpath.moveTo(self._to_screen(path[0]))
            for i in range(1, len(path)):
                qpath.lineTo(self._to_screen(path[i]))
            painter.drawPath(qpath)

    def _draw_fpc_layout(self, painter: QPainter):
        """绘制FPC布局（带宽度的凹槽轮廓）"""
        groove_colors = [
            '#f39c12',  # 橙色
            '#3498db',  # 蓝色
            '#9b59b6',  # 紫色
            '#1abc9c',  # 青色
            '#e91e63',  # 粉色
            '#00bcd4',  # 天蓝
            '#ff5722',  # 深橙
            '#8bc34a',  # 浅绿
        ]

        # 绘制凹槽轮廓（填充）
        for idx, (point_id, outline) in enumerate(self.groove_outlines.items()):
            if len(outline) < 3:
                continue

            color = QColor(groove_colors[idx % len(groove_colors)])
            color.setAlpha(100)  # 半透明填充

            # 绘制填充的多边形
            polygon = QPolygonF()
            for pt in outline:
                polygon.append(self._to_screen(pt))

            painter.setPen(QPen(QColor(groove_colors[idx % len(groove_colors)]), 1.5))
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

        # 绘制IR点焊盘
        for point_id, pad in self.ir_pads.items():
            if len(pad) < 3:
                continue

            color = QColor('#2ecc71')
            color.setAlpha(150)

            polygon = QPolygonF()
            for pt in pad:
                polygon.append(self._to_screen(pt))

            painter.setPen(QPen(QColor('#2ecc71'), 1.5))
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

        # 绘制中心焊盘
        if self.center_pad is not None and len(self.center_pad) >= 3:
            color = QColor('#e74c3c')
            color.setAlpha(150)

            polygon = QPolygonF()
            for pt in self.center_pad:
                polygon.append(self._to_screen(pt))

            painter.setPen(QPen(QColor('#e74c3c'), 1.5))
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

        # 绘制合并后的总轮廓（虚线）
        if self.merged_outline is not None and len(self.merged_outline) >= 3:
            pen = QPen(QColor('#ffffff'))
            pen.setWidthF(2.0)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            polygon = QPolygonF()
            for pt in self.merged_outline:
                polygon.append(self._to_screen(pt))
            # 闭合轮廓
            polygon.append(self._to_screen(self.merged_outline[0]))

            painter.drawPolygon(polygon)

    def _draw_ir_points(self, painter: QPainter):
        """绘制IR点"""
        for point_id, (pos, name, is_center) in self.ir_points_2d.items():
            screen_pos = self._to_screen(pos)

            # 绘制点
            if is_center:
                radius = 8
                color = QColor('#e74c3c')  # 红色中心点
            else:
                radius = 6
                color = QColor('#2ecc71')  # 绿色IR点

            painter.setPen(QPen(Qt.white, 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(screen_pos, radius, radius)

            # 绘制标签
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 9, QFont.Bold))
            painter.drawText(int(screen_pos.x() + 12), int(screen_pos.y() + 4), name)

    def _draw_scale_bar(self, painter: QPainter):
        """绘制比例尺"""
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Arial', 10))

        # 10mm比例尺
        scale_length_mm = 10.0
        scale_length_px = scale_length_mm * self.view_scale

        # 绘制比例尺
        x = 20
        y = self.height() - 30
        painter.drawLine(x, y, int(x + scale_length_px), y)
        painter.drawLine(x, y - 5, x, y + 5)
        painter.drawLine(int(x + scale_length_px), y - 5, int(x + scale_length_px), y + 5)
        painter.drawText(x, y - 10, f"{scale_length_mm:.0f} mm")

    def _draw_legend(self, painter: QPainter):
        """绘制图例"""
        painter.setFont(QFont('Arial', 9))

        x = self.width() - 120
        y = 20

        # 中心点
        painter.setBrush(QBrush(QColor('#e74c3c')))
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(x, y, 10, 10)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x + 15, y + 9, "中心点")

        # IR点
        y += 20
        painter.setBrush(QBrush(QColor('#2ecc71')))
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(x, y, 10, 10)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x + 15, y + 9, "IR点")

        if self.show_fpc_layout and self.groove_outlines:
            # 凹槽轮廓
            y += 20
            color = QColor('#f39c12')
            color.setAlpha(100)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#f39c12'), 1))
            painter.drawRect(x, y, 10, 10)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 15, y + 9, "凹槽轮廓")

            # 外边框
            y += 20
            pen = QPen(QColor('#ffffff'))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, 10, 10)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 15, y + 9, "FPC外形")
        else:
            # 路径
            y += 20
            painter.setPen(QPen(QColor('#f39c12'), 2))
            painter.drawLine(x, y + 5, x + 10, y + 5)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 15, y + 9, "走线路径")

    def resizeEvent(self, event):
        """窗口大小改变"""
        self._auto_fit()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.view_scale *= factor
        self.update()

    def mousePressEvent(self, event):
        """鼠标按下"""
        if event.button() == Qt.MiddleButton:
            self._drag_start = event.pos()
            self._offset_start = self.view_offset.copy()

    def mouseMoveEvent(self, event):
        """鼠标移动"""
        if hasattr(self, '_drag_start') and event.buttons() & Qt.MiddleButton:
            delta = event.pos() - self._drag_start
            self.view_offset = self._offset_start + np.array([delta.x(), -delta.y()])
            self.update()

    def get_paths_for_export(self) -> List[np.ndarray]:
        """获取用于导出的路径（实际尺寸mm）"""
        return list(self.paths_2d.values())

    def get_points_for_export(self) -> List[Tuple[np.ndarray, str]]:
        """获取用于导出的点（实际尺寸mm）"""
        return [(pos, name) for pos, name, _ in self.ir_points_2d.values()]

    def get_fpc_layout_for_export(self) -> Dict:
        """获取FPC布局数据用于导出"""
        return {
            'groove_outlines': list(self.groove_outlines.values()),
            'ir_pads': list(self.ir_pads.values()),
            'center_pad': self.center_pad,
            'merged_outline': self.merged_outline
        }

    def has_fpc_layout(self) -> bool:
        """检查是否有FPC布局数据"""
        return len(self.groove_outlines) > 0
