"""DXF导入模块 - 解析FPC DXF并反向映射到3D曲面

功能：
1. 解析本软件导出的FPC DXF文件
2. 提取IR点位置、中心点、沟槽轮廓等信息
3. 将2D点关系反向映射到3D曲面
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

from .path_planner_2d import (
    build_tangent_plane_basis,
    project_to_surface,
    tangent_plane_to_3d
)


@dataclass
class DXFImportResult:
    """DXF导入解析结果"""
    # IR点焊盘的2D中心坐标 {标签名 -> 2D坐标}
    ir_pads_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    # 中心点焊盘的2D中心坐标
    center_pad_2d: Optional[np.ndarray] = None
    # 沟槽轮廓 {标签名 -> 轮廓点数组}
    groove_outlines_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    # 所有沟槽轮廓列表（用于未能匹配标签的情况）
    all_groove_outlines: List[np.ndarray] = field(default_factory=list)
    # 标签文本信息 {标签名 -> 2D位置}
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    # 缩放因子
    scale: float = 1.0
    # 边界
    bounds: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.zeros(2), np.zeros(2))
    )


class DXFImporter:
    """DXF文件解析器"""

    def __init__(self):
        if not HAS_EZDXF:
            raise ImportError("ezdxf库未安装，无法解析DXF文件")

    def parse_fpc_dxf(self, filepath: str) -> Optional[DXFImportResult]:
        """
        解析FPC DXF文件

        Args:
            filepath: DXF文件路径

        Returns:
            DXFImportResult 或 None（解析失败时）
        """
        try:
            doc = ezdxf.readfile(filepath)
            msp = doc.modelspace()

            result = DXFImportResult()

            # 收集各图层的实体
            ir_pad_polygons = []  # IR焊盘轮廓
            center_pad_polygon = None  # 中心焊盘轮廓
            groove_polygons = []  # 沟槽轮廓
            label_texts = []  # 标签文本

            # 遍历所有实体
            for entity in msp:
                layer = entity.dxf.layer.upper()

                if entity.dxftype() == 'LWPOLYLINE':
                    # 获取多边形顶点
                    points = np.array([(p[0], p[1]) for p in entity.get_points()])
                    if len(points) < 3:
                        continue

                    if layer == 'IR_PADS':
                        ir_pad_polygons.append(points)
                    elif layer == 'CENTER_PAD':
                        center_pad_polygon = points
                    elif layer == 'GROOVES':
                        groove_polygons.append(points)
                        result.all_groove_outlines.append(points)

                elif entity.dxftype() == 'TEXT':
                    if layer == 'LABELS':
                        text = entity.dxf.text
                        pos = np.array([entity.dxf.insert.x, entity.dxf.insert.y])
                        label_texts.append((text, pos))
                        result.labels[text] = pos

            # 计算中心焊盘的质心
            if center_pad_polygon is not None:
                result.center_pad_2d = self._compute_centroid(center_pad_polygon)

            # 计算各IR焊盘的质心并匹配标签
            pad_centroids = []
            for pad in ir_pad_polygons:
                centroid = self._compute_centroid(pad)
                pad_centroids.append(centroid)

            # 匹配标签到焊盘（使用最近距离）
            for label, label_pos in label_texts:
                if label.upper().startswith('P') or label.upper().startswith('IR'):
                    # 找到离这个标签最近的焊盘
                    min_dist = float('inf')
                    nearest_centroid = None

                    for centroid in pad_centroids:
                        dist = np.linalg.norm(centroid - label_pos)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_centroid = centroid

                    if nearest_centroid is not None and min_dist < 10:  # 10mm阈值
                        result.ir_pads_2d[label] = nearest_centroid

            # 如果没有匹配到任何标签，使用默认命名
            if not result.ir_pads_2d and pad_centroids:
                for i, centroid in enumerate(pad_centroids):
                    result.ir_pads_2d[f"P{i+1}"] = centroid

            # 计算边界
            all_points = []
            for points in ir_pad_polygons:
                all_points.extend(points)
            if center_pad_polygon is not None:
                all_points.extend(center_pad_polygon)
            for points in groove_polygons:
                all_points.extend(points)

            if all_points:
                all_points = np.array(all_points)
                result.bounds = (
                    np.min(all_points, axis=0),
                    np.max(all_points, axis=0)
                )

            # 尝试匹配沟槽到点
            self._match_grooves_to_points(result, groove_polygons)

            return result

        except Exception as e:
            print(f"解析DXF文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_centroid(self, polygon: np.ndarray) -> np.ndarray:
        """计算多边形质心"""
        # 使用简单的平均值（对于凸多边形足够精确）
        return np.mean(polygon, axis=0)

    def _match_grooves_to_points(
        self,
        result: DXFImportResult,
        groove_polygons: List[np.ndarray]
    ):
        """尝试将沟槽轮廓匹配到对应的IR点"""
        if result.center_pad_2d is None:
            return

        center = result.center_pad_2d

        for label, pad_pos in result.ir_pads_2d.items():
            # 找到连接中心点和这个IR点的沟槽
            direction = pad_pos - center
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            best_match = None
            best_score = float('inf')

            for groove in groove_polygons:
                # 计算沟槽质心
                groove_center = np.mean(groove, axis=0)

                # 检查沟槽是否在中心点和IR点之间
                to_groove = groove_center - center
                distance_along = np.dot(to_groove, direction)
                distance_to_line = np.linalg.norm(
                    to_groove - distance_along * direction
                )

                # 沟槽应该在中心点和IR点之间，且接近连线
                ir_distance = np.linalg.norm(pad_pos - center)
                if 0 < distance_along < ir_distance and distance_to_line < 5:
                    score = distance_to_line
                    if score < best_score:
                        best_score = score
                        best_match = groove

            if best_match is not None:
                result.groove_outlines_2d[label] = best_match


class GeodesicPolarMapper:
    """测地线极坐标映射器 - 从2D相对坐标反向计算3D曲面位置"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh

    def map_2d_to_3d(
        self,
        reference_point_3d: np.ndarray,
        target_offset_2d: np.ndarray,
        reference_normal: np.ndarray,
        num_steps: int = 20
    ) -> np.ndarray:
        """
        从参考点出发，根据2D偏移计算目标点在3D曲面上的位置

        Args:
            reference_point_3d: 参考点的3D位置（已投影到曲面）
            target_offset_2d: 目标点相对于参考点的2D偏移 (dx, dy)
            reference_normal: 参考点的法向量
            num_steps: 曲面步进的步数（默认20步，足够大多数情况）

        Returns:
            目标点的3D位置
        """
        # 计算2D偏移的距离和方向
        distance = np.linalg.norm(target_offset_2d)
        if distance < 1e-6:
            return reference_point_3d.copy()

        # 构建切平面坐标系
        x_axis, y_axis, _ = build_tangent_plane_basis(reference_normal)

        # 将2D方向转换为3D初始方向
        direction_2d = target_offset_2d / distance
        direction_3d = direction_2d[0] * x_axis + direction_2d[1] * y_axis
        direction_3d = direction_3d / (np.linalg.norm(direction_3d) + 1e-10)

        # 根据距离调整步数（距离短则步数少）
        adaptive_steps = max(5, min(num_steps, int(distance * 2)))

        # 沿曲面步进
        return self._walk_on_surface(
            reference_point_3d,
            direction_3d,
            distance,
            adaptive_steps
        )

    def _walk_on_surface(
        self,
        start: np.ndarray,
        initial_direction: np.ndarray,
        distance: float,
        num_steps: int = 100
    ) -> np.ndarray:
        """
        沿曲面从起点向指定方向步进指定距离

        Args:
            start: 起点3D坐标（已在曲面上）
            initial_direction: 初始方向（3D单位向量）
            distance: 要步进的总距离
            num_steps: 步进次数

        Returns:
            终点的3D坐标
        """
        if distance < 1e-6:
            return start.copy()

        step_size = distance / num_steps
        current = start.copy()
        direction = initial_direction.copy()

        for _ in range(num_steps):
            # 获取当前点的曲面法向量
            try:
                _, _, face_idx = self.mesh.nearest.on_surface([current])
                normal = self.mesh.face_normals[int(face_idx[0])]
            except:
                # 如果查询失败，使用最近顶点的法向量
                distances = np.linalg.norm(self.mesh.vertices - current, axis=1)
                nearest_idx = np.argmin(distances)
                normal = self.mesh.vertex_normals[nearest_idx]

            # 将方向投影到切平面（保持在曲面上）
            tangent_dir = direction - np.dot(direction, normal) * normal
            tangent_norm = np.linalg.norm(tangent_dir)
            if tangent_norm > 1e-10:
                tangent_dir = tangent_dir / tangent_norm
            else:
                # 方向与法向量平行，无法继续
                break

            # 步进
            next_pos = current + tangent_dir * step_size

            # 投影回曲面
            current = project_to_surface(self.mesh, next_pos)

            # 更新方向（平行传输）
            direction = tangent_dir

        return current

    def map_multiple_points(
        self,
        reference_point_3d: np.ndarray,
        reference_normal: np.ndarray,
        offsets_2d: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        批量映射多个点

        Args:
            reference_point_3d: 参考点的3D位置
            reference_normal: 参考点的法向量
            offsets_2d: 各点相对于参考点的2D偏移 {label -> (dx, dy)}

        Returns:
            各点的3D位置 {label -> 3D坐标}
        """
        results = {}
        for label, offset in offsets_2d.items():
            results[label] = self.map_2d_to_3d(
                reference_point_3d,
                offset,
                reference_normal
            )
        return results


class DXFImportManager(QObject):
    """DXF导入管理器 - 管理导入状态和实时更新"""

    # 信号
    points_updated = pyqtSignal(dict)  # 发射更新后的点位置 {label -> 3D坐标}
    import_completed = pyqtSignal()  # 导入完成
    import_cancelled = pyqtSignal()  # 导入取消
    status_message = pyqtSignal(str)  # 状态消息

    def __init__(self, mesh: trimesh.Trimesh, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.polar_mapper = GeodesicPolarMapper(mesh)

        # 导入数据
        self.import_result: Optional[DXFImportResult] = None
        self.relative_offsets_2d: Dict[str, np.ndarray] = {}  # 各点相对中心的2D偏移

        # 参考点状态
        self.reference_point_3d: Optional[np.ndarray] = None
        self.reference_normal: Optional[np.ndarray] = None
        self.reference_label: str = "Center"

        # 计算出的3D点位置
        self.points_3d: Dict[str, np.ndarray] = {}

        # 更新节流
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._do_update)
        self._pending_position: Optional[np.ndarray] = None
        self._pending_rotation: Optional[float] = None
        self._pending_offset: Optional[np.ndarray] = None

        # 旋转角度（弧度）
        self.rotation_angle: float = 0.0
        # 2D偏移（相对于初始位置）
        self.offset_2d: np.ndarray = np.zeros(2)  # (x, y) 偏移量

    def load_dxf(self, filepath: str) -> bool:
        """
        加载DXF文件

        Args:
            filepath: DXF文件路径

        Returns:
            是否加载成功
        """
        try:
            importer = DXFImporter()
            self.import_result = importer.parse_fpc_dxf(filepath)

            if self.import_result is None:
                return False

            if self.import_result.center_pad_2d is None:
                self.status_message.emit("DXF文件中未找到中心焊盘")
                return False

            if not self.import_result.ir_pads_2d:
                self.status_message.emit("DXF文件中未找到IR点焊盘")
                return False

            # 计算各点相对于中心点的2D偏移
            center_2d = self.import_result.center_pad_2d
            self.relative_offsets_2d.clear()

            for label, pos_2d in self.import_result.ir_pads_2d.items():
                self.relative_offsets_2d[label] = pos_2d - center_2d

            self.status_message.emit(
                f"已加载 {len(self.import_result.ir_pads_2d)} 个IR点"
            )
            return True

        except Exception as e:
            self.status_message.emit(f"加载DXF失败: {e}")
            return False

    def set_reference_point(self, position_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        设置参考点位置并计算所有点的3D位置

        Args:
            position_3d: 参考点的3D位置

        Returns:
            所有点的3D位置 {label -> 3D坐标}
        """
        # 投影到曲面
        self.reference_point_3d = project_to_surface(self.mesh, position_3d)

        # 获取法向量
        try:
            _, _, face_idx = self.mesh.nearest.on_surface([self.reference_point_3d])
            self.reference_normal = self.mesh.face_normals[int(face_idx[0])]
        except:
            distances = np.linalg.norm(
                self.mesh.vertices - self.reference_point_3d, axis=1
            )
            nearest_idx = np.argmin(distances)
            self.reference_normal = self.mesh.vertex_normals[nearest_idx]

        # 计算所有点的位置
        return self._compute_all_points()

    def update_reference_point(self, position_3d: np.ndarray):
        """
        更新参考点位置（带节流）

        Args:
            position_3d: 新的参考点位置
        """
        self._pending_position = position_3d
        if not self._update_timer.isActive():
            self._update_timer.start(50)  # 50ms节流

    def _do_update(self):
        """执行实际的更新"""
        need_update = False

        # 处理pending position
        if self._pending_position is not None:
            # 投影到曲面
            self.reference_point_3d = project_to_surface(self.mesh, self._pending_position)
            # 获取法向量
            try:
                _, _, face_idx = self.mesh.nearest.on_surface([self.reference_point_3d])
                self.reference_normal = self.mesh.face_normals[int(face_idx[0])]
            except:
                pass
            self._pending_position = None
            need_update = True

        # 处理pending rotation
        if hasattr(self, '_pending_rotation') and self._pending_rotation is not None:
            self.rotation_angle = self._pending_rotation
            self._pending_rotation = None
            need_update = True

        # 处理pending offset
        if hasattr(self, '_pending_offset') and self._pending_offset is not None:
            self.offset_2d = self._pending_offset
            self._pending_offset = None
            need_update = True

        # 如果有任何更新，重新计算所有点
        if need_update and self.reference_point_3d is not None:
            points = self._compute_all_points()
            self.points_updated.emit(points)

    def set_rotation(self, angle_degrees: float):
        """
        设置旋转角度（带节流）

        Args:
            angle_degrees: 旋转角度（度）
        """
        self._pending_rotation = np.radians(angle_degrees)
        if not self._update_timer.isActive():
            self._update_timer.start(50)  # 50ms节流

    def set_offset_2d(self, x_offset: float, y_offset: float):
        """
        设置2D偏移量（带节流）

        Args:
            x_offset: X方向偏移（mm）
            y_offset: Y方向偏移（mm）
        """
        self._pending_offset = np.array([x_offset, y_offset])
        if not self._update_timer.isActive():
            self._update_timer.start(50)  # 50ms节流

    def _compute_all_points(self) -> Dict[str, np.ndarray]:
        """计算所有点的3D位置"""
        if self.reference_point_3d is None or self.reference_normal is None:
            return {}

        self.points_3d.clear()

        # 构建切平面坐标系
        x_axis, y_axis, _ = build_tangent_plane_basis(self.reference_normal)

        # 应用2D偏移到参考点（在切平面上移动）
        offset_3d = self.offset_2d[0] * x_axis + self.offset_2d[1] * y_axis
        adjusted_ref_point = project_to_surface(self.mesh, self.reference_point_3d + offset_3d)

        # 获取调整后参考点的法向量
        try:
            _, _, face_idx = self.mesh.nearest.on_surface([adjusted_ref_point])
            adjusted_normal = self.mesh.face_normals[int(face_idx[0])]
        except:
            adjusted_normal = self.reference_normal

        # 添加调整后的中心点
        self.points_3d[self.reference_label] = adjusted_ref_point.copy()

        # 应用旋转到偏移量
        cos_a = np.cos(self.rotation_angle)
        sin_a = np.sin(self.rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # 计算各IR点的位置
        for label, offset_2d in self.relative_offsets_2d.items():
            # 应用旋转
            rotated_offset = rotation_matrix @ offset_2d

            # 映射到3D（使用调整后的参考点和法向量）
            pos_3d = self.polar_mapper.map_2d_to_3d(
                adjusted_ref_point,
                rotated_offset,
                adjusted_normal
            )
            self.points_3d[label] = pos_3d

        return self.points_3d

    def get_groove_outlines(self) -> List[np.ndarray]:
        """获取沟槽轮廓数据"""
        if self.import_result is None:
            return []
        return self.import_result.all_groove_outlines

    def get_point_count(self) -> int:
        """获取导入的点数量"""
        if self.import_result is None:
            return 0
        return len(self.import_result.ir_pads_2d)

    def cancel_import(self):
        """取消导入"""
        self.import_result = None
        self.relative_offsets_2d.clear()
        self.points_3d.clear()
        self.reference_point_3d = None
        self.reference_normal = None
        self.import_cancelled.emit()

    def finalize_import(self):
        """确认导入"""
        self.import_completed.emit()
