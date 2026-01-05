"""凹槽生成模块"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import splprep, splev


@dataclass
class GrooveParams:
    """凹槽参数"""
    width: float = 1.0  # 凹槽宽度 (mm)
    depth: float = 0.5  # 凹槽深度 (mm)
    extension_height: float = 0.5  # 向上延伸高度（确保切穿表面）
    smooth_bottom: bool = True  # 是否平滑底部
    auto_width: bool = True  # 是否根据汇聚线路自动调整宽度
    min_width: float = 0.5  # 最小宽度
    max_width: float = 5.0  # 最大宽度
    resolution: int = 20  # 路径采样密度


class GrooveGenerator:
    """凹槽生成器"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.params = GrooveParams()

    def set_params(self, **kwargs):
        """设置凹槽参数"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def _smooth_path_bottom(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        depth: float
    ) -> np.ndarray:
        """
        计算平滑的凹槽底部中心线

        方法：
        1. 计算原始底部点
        2. 使用移动平均或样条插值平滑
        3. 保持整体深度一致

        Args:
            path_points: 路径点
            path_normals: 法向量
            depth: 凹槽深度

        Returns:
            平滑后的底部中心点
        """
        n = len(path_points)
        if n < 4:
            # 点太少，直接返回原始底部
            return path_points - depth * path_normals

        # 计算原始底部点
        bottom_points = path_points - depth * path_normals

        # 使用移动平均平滑
        window = min(5, n // 2)
        if window < 2:
            return bottom_points

        smoothed = bottom_points.copy()
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            smoothed[i] = np.mean(bottom_points[start:end], axis=0)

        # 保持端点
        smoothed[0] = bottom_points[0]
        smoothed[-1] = bottom_points[-1]

        # 可选：使用样条进一步平滑
        try:
            # 计算弧长参数
            diffs = np.diff(smoothed, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])

            if cumulative[-1] > 1e-6:
                t = cumulative / cumulative[-1]

                # 分别对x, y, z拟合样条
                tck_x, _ = splprep([t, smoothed[:, 0]], s=0.1, k=min(3, n-1))
                tck_y, _ = splprep([t, smoothed[:, 1]], s=0.1, k=min(3, n-1))
                tck_z, _ = splprep([t, smoothed[:, 2]], s=0.1, k=min(3, n-1))

                # 重新采样
                t_new = np.linspace(0, 1, n)
                x_new = splev(t_new, tck_x)[1]
                y_new = splev(t_new, tck_y)[1]
                z_new = splev(t_new, tck_z)[1]

                smoothed = np.column_stack([x_new, y_new, z_new])
        except:
            # 样条失败，使用移动平均的结果
            pass

        return smoothed

    def generate_groove_profile(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        沿路径生成凹槽几何体

        凹槽横截面为矩形:
        - 宽度方向: 沿曲面切向（路径切向×法向）
        - 深度方向: 沿曲面法向（向内）
        - 高度方向: 沿曲面法向（向外延伸确保切穿）

        Args:
            path_points: 路径点坐标 (N, 3)
            path_normals: 路径点法向量 (N, 3)
            width: 凹槽宽度，None则使用默认参数
            depth: 凹槽深度，None则使用默认参数

        Returns:
            凹槽的3D网格
        """
        if len(path_points) < 2:
            return None

        w = width if width is not None else self.params.width
        d = depth if depth is not None else self.params.depth
        ext_h = self.params.extension_height

        n = len(path_points)

        # 归一化法向量
        normals = path_normals.copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        normals = normals / norms

        # 计算路径切向量
        tangents = np.zeros_like(path_points)
        tangents[0] = path_points[1] - path_points[0]
        tangents[-1] = path_points[-1] - path_points[-2]
        for i in range(1, n - 1):
            tangents[i] = path_points[i + 1] - path_points[i - 1]

        # 归一化切向量
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        t_norms[t_norms < 1e-10] = 1.0
        tangents = tangents / t_norms

        # 计算侧向量: 必须同时垂直于法向量和切向量
        # side = tangent × normal，这样side在曲面上
        side_vectors = np.cross(tangents, normals)
        s_norms = np.linalg.norm(side_vectors, axis=1, keepdims=True)

        # 处理退化情况（切向与法向平行）
        for i in range(n):
            if s_norms[i] < 1e-6:
                # 切向与法向几乎平行，使用替代方向
                if abs(normals[i, 0]) < 0.9:
                    alt = np.array([1.0, 0.0, 0.0])
                else:
                    alt = np.array([0.0, 1.0, 0.0])
                side_vectors[i] = np.cross(alt, normals[i])
                s_norms[i] = np.linalg.norm(side_vectors[i])

        s_norms[s_norms < 1e-10] = 1.0
        side_vectors = side_vectors / s_norms

        # 确保side_vectors真正垂直于normals（修正任何数值误差）
        for i in range(n):
            # 移除side在normal方向的分量
            dot = np.dot(side_vectors[i], normals[i])
            side_vectors[i] = side_vectors[i] - dot * normals[i]
            side_vectors[i] = side_vectors[i] / (np.linalg.norm(side_vectors[i]) + 1e-10)

        # 平滑side向量以避免扭曲
        if n > 3:
            smoothed_side = side_vectors.copy()
            for _ in range(3):
                for i in range(1, n - 1):
                    avg = (side_vectors[i-1] + side_vectors[i] + side_vectors[i+1]) / 3
                    # 保持垂直于法向量
                    avg = avg - np.dot(avg, normals[i]) * normals[i]
                    norm_avg = np.linalg.norm(avg)
                    if norm_avg > 1e-6:
                        smoothed_side[i] = avg / norm_avg
                side_vectors = smoothed_side.copy()

        # 计算平滑的底部中心线
        if self.params.smooth_bottom:
            bottom_center = self._smooth_path_bottom(path_points, normals, d)
        else:
            bottom_center = path_points - d * normals

        # 计算顶部中心线（向上延伸）
        top_center = path_points + ext_h * normals

        # 生成凹槽顶点 - 矩形横截面
        half_w = w / 2

        vertices = []
        faces = []

        for i in range(n):
            normal = normals[i]
            side = side_vectors[i]
            top_p = top_center[i]
            bot_p = bottom_center[i]

            # 矩形横截面的4个顶点
            # 顶部边（向外延伸）
            v0 = top_p + half_w * side  # 顶左
            v1 = top_p - half_w * side  # 顶右
            # 底部边（向内挖入）
            v2 = bot_p + half_w * side  # 底左
            v3 = bot_p - half_w * side  # 底右

            base_idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])

            # 连接相邻截面形成面
            if i > 0:
                prev_base = base_idx - 4
                # 顶面
                faces.append([prev_base, prev_base + 1, base_idx + 1])
                faces.append([prev_base, base_idx + 1, base_idx])
                # 左侧面
                faces.append([prev_base, base_idx, base_idx + 2])
                faces.append([prev_base, base_idx + 2, prev_base + 2])
                # 右侧面
                faces.append([prev_base + 1, prev_base + 3, base_idx + 3])
                faces.append([prev_base + 1, base_idx + 3, base_idx + 1])
                # 底面
                faces.append([prev_base + 2, base_idx + 2, base_idx + 3])
                faces.append([prev_base + 2, base_idx + 3, prev_base + 3])

        # 封闭两端
        if n >= 2:
            # 起始端
            faces.append([0, 2, 3])
            faces.append([0, 3, 1])
            # 结束端
            end_base = (n - 1) * 4
            faces.append([end_base, end_base + 1, end_base + 3])
            faces.append([end_base, end_base + 3, end_base + 2])

        if not vertices:
            return None

        vertices = np.array(vertices)
        faces = np.array(faces)

        groove_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        groove_mesh.fix_normals()

        return groove_mesh

    def subtract_groove_from_mesh(
        self,
        groove: trimesh.Trimesh
    ) -> Optional[trimesh.Trimesh]:
        """
        从原始网格中减去凹槽

        Args:
            groove: 凹槽网格

        Returns:
            带凹槽的新网格
        """
        try:
            # 使用trimesh的布尔运算
            result = self.mesh.difference(groove, engine='blender')
            return result
        except Exception as e:
            print(f"布尔运算失败: {e}")
            # 备选方案：返回合并的网格供可视化
            return trimesh.util.concatenate([self.mesh, groove])

    def generate_all_grooves(
        self,
        paths: List[Tuple[np.ndarray, np.ndarray]],
        merge_at_intersections: bool = True
    ) -> List[trimesh.Trimesh]:
        """
        生成所有路径的凹槽

        Args:
            paths: 路径列表，每个元素为 (path_points, path_normals)
            merge_at_intersections: 是否在交汇处合并加宽

        Returns:
            凹槽网格列表
        """
        grooves = []

        for path_points, path_normals in paths:
            groove = self.generate_groove_profile(path_points, path_normals)
            if groove is not None:
                grooves.append(groove)

        return grooves

    def compute_adaptive_width(
        self,
        point: np.ndarray,
        all_paths: List[np.ndarray],
        base_width: float
    ) -> float:
        """
        计算自适应宽度（根据汇聚线路数量）

        Args:
            point: 查询点
            all_paths: 所有路径点
            base_width: 基础宽度

        Returns:
            调整后的宽度
        """
        # 统计在该点附近的路径数量
        threshold = base_width * 2
        count = 0

        for path in all_paths:
            distances = np.linalg.norm(path - point, axis=1)
            if np.min(distances) < threshold:
                count += 1

        # 根据汇聚数量调整宽度
        width = base_width * count
        return np.clip(width, self.params.min_width, self.params.max_width)
