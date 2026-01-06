"""凹槽生成模块"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import splprep, splev


@dataclass
class GrooveParams:
    """凹槽参数"""
    width: float = 1.0  # 路径凹槽宽度 (mm)
    depth: float = 0.5  # 凹槽深度 (mm)
    extension_height: float = 0.02  # 向上延伸高度（很小，仅确保能切到表面）
    smooth_bottom: bool = True  # 是否平滑底部
    auto_width: bool = True  # 是否根据汇聚线路自动调整宽度
    min_width: float = 0.5  # 最小宽度
    max_width: float = 5.0  # 最大宽度
    resolution: int = 20  # 路径采样密度
    conform_to_surface: bool = True  # 是否生成曲面贴合的凹槽
    # 方形焊盘凹槽参数
    pad_length: float = 3.0  # 方形凹槽长度 (mm) - 沿路径方向
    pad_width: float = 2.0   # 方形凹槽宽度 (mm) - 垂直于路径方向
    pad_enabled: bool = True  # 是否生成方形焊盘凹槽


class GrooveGenerator:
    """凹槽生成器"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.params = GrooveParams()
        # 预计算模型厚度信息
        self._estimate_mesh_thickness()

    def _estimate_mesh_thickness(self):
        """估算模型的平均厚度"""
        # 采样一些表面点，向内发射射线
        samples = min(100, len(self.mesh.vertices))
        indices = np.random.choice(len(self.mesh.vertices), samples, replace=False)

        thicknesses = []
        for idx in indices:
            point = self.mesh.vertices[idx]
            normal = self.mesh.vertex_normals[idx]

            # 确保法向量指向外部
            to_center = self.mesh.centroid - point
            if np.dot(normal, to_center) > 0:
                normal = -normal

            # 向内发射射线
            ray_origin = point - 0.01 * normal
            ray_direction = -normal

            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )

            if len(locations) > 0:
                distances = np.linalg.norm(locations - point, axis=1)
                valid = distances[distances > 0.05]
                if len(valid) > 0:
                    thicknesses.append(np.min(valid))

        if thicknesses:
            self.estimated_thickness = np.median(thicknesses)
            print(f"估算模型厚度: {self.estimated_thickness:.2f}mm")
        else:
            self.estimated_thickness = float('inf')
            print("无法估算模型厚度（可能是实心模型）")

    def set_params(self, **kwargs):
        """设置凹槽参数"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def _compute_parallel_transport_frames(
        self,
        path_points: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """
        使用平行传输计算沿路径的一致侧向量

        平行传输保持向量在曲面上"平行"移动，避免扇形发散
        """
        n = len(path_points)
        side_vectors = np.zeros_like(path_points)

        # 计算第一个点的侧向量
        tangent0 = path_points[1] - path_points[0]
        tangent0 = tangent0 / (np.linalg.norm(tangent0) + 1e-10)

        # 初始侧向量 = tangent × normal
        side0 = np.cross(tangent0, normals[0])
        side_norm = np.linalg.norm(side0)
        if side_norm < 1e-6:
            # 退化情况：使用替代方向
            if abs(normals[0, 0]) < 0.9:
                alt = np.array([1.0, 0.0, 0.0])
            else:
                alt = np.array([0.0, 1.0, 0.0])
            side0 = np.cross(alt, normals[0])
            side0 = side0 / (np.linalg.norm(side0) + 1e-10)
        else:
            side0 = side0 / side_norm

        side_vectors[0] = side0

        # 沿路径平行传输
        for i in range(1, n):
            prev_side = side_vectors[i - 1]
            curr_normal = normals[i]

            # 将前一个侧向量投影到当前法向量的切平面上
            # 移除与法向量平行的分量
            dot = np.dot(prev_side, curr_normal)
            projected = prev_side - dot * curr_normal
            proj_norm = np.linalg.norm(projected)

            if proj_norm > 1e-6:
                side_vectors[i] = projected / proj_norm
            else:
                # 严重退化，使用计算方法
                if i < n - 1:
                    tangent = path_points[i + 1] - path_points[i]
                else:
                    tangent = path_points[i] - path_points[i - 1]
                tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
                side = np.cross(tangent, curr_normal)
                side_vectors[i] = side / (np.linalg.norm(side) + 1e-10)

        return side_vectors

    def _project_points_to_surface(
        self,
        points: np.ndarray,
        search_distance: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点投影到曲面上

        Args:
            points: 要投影的点 (N, 3)
            search_distance: 搜索距离

        Returns:
            (投影后的点, 投影点的法向量)
        """
        projected_points = np.zeros_like(points)
        projected_normals = np.zeros_like(points)

        for i, point in enumerate(points):
            # 使用trimesh的最近点查询
            closest, distance, face_idx = self.mesh.nearest.on_surface([point])
            projected_points[i] = closest[0]

            # 获取该面的法向量
            if face_idx[0] >= 0:
                face = self.mesh.faces[face_idx[0]]
                # 计算面法向量
                v0, v1, v2 = self.mesh.vertices[face]
                e1 = v1 - v0
                e2 = v2 - v0
                face_normal = np.cross(e1, e2)
                norm = np.linalg.norm(face_normal)
                if norm > 1e-10:
                    projected_normals[i] = face_normal / norm
                else:
                    # 回退到顶点法向量
                    projected_normals[i] = self.mesh.vertex_normals[face[0]]
            else:
                # 回退到最近顶点的法向量
                dists = np.linalg.norm(self.mesh.vertices - point, axis=1)
                nearest_idx = np.argmin(dists)
                projected_normals[i] = self.mesh.vertex_normals[nearest_idx]

        # 确保法向量指向外部
        mesh_center = self.mesh.centroid
        for i in range(len(projected_normals)):
            to_center = mesh_center - projected_points[i]
            if np.dot(projected_normals[i], to_center) > 0:
                projected_normals[i] = -projected_normals[i]

        return projected_points, projected_normals

    def generate_groove_profile_curved(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        生成曲面贴合的凹槽几何体

        凹槽的顶部边界精确位于曲面上，实现与FPC的完美贴合

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
        half_w = w / 2.0

        n = len(path_points)

        # 归一化法向量
        normals = path_normals.copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        normals = normals / norms

        # 确保法向量指向模型外部
        mesh_center = self.mesh.centroid
        for i in range(n):
            to_center = mesh_center - path_points[i]
            if np.dot(normals[i], to_center) > 0:
                normals[i] = -normals[i]

        # 限制凹槽深度，防止打穿
        max_safe_depth = self.estimated_thickness * 0.7 if self.estimated_thickness < float('inf') else d
        if d > max_safe_depth and max_safe_depth > 0.1:
            print(f"警告: 凹槽深度 {d:.2f}mm 超过安全限制，调整为 {max_safe_depth:.2f}mm")
            d = max_safe_depth

        # 使用平行传输计算一致的侧向量
        side_vectors = self._compute_parallel_transport_frames(path_points, normals)

        # 计算初始的左右边界点（未投影到曲面）
        left_points_init = path_points + half_w * side_vectors
        right_points_init = path_points - half_w * side_vectors

        # 将左右边界点投影到曲面上
        left_top, left_normals = self._project_points_to_surface(left_points_init)
        right_top, right_normals = self._project_points_to_surface(right_points_init)

        # 中心路径也投影到曲面（确保在曲面上）
        center_top, center_normals = self._project_points_to_surface(path_points)

        # 计算底部点：从曲面上的顶部点沿法向量向内偏移
        # 使用投影后的法向量，确保凹槽底部与顶部平行于曲面
        left_bottom = left_top - d * left_normals
        right_bottom = right_top - d * right_normals
        center_bottom = center_top - d * center_normals

        # 向上微小延伸以确保能完全切穿表面
        left_top_ext = left_top + ext_h * left_normals
        right_top_ext = right_top + ext_h * right_normals

        # 生成网格顶点和面
        # 横截面结构（梯形，更好地贴合曲面）：
        #   v0 (左上) ---- v1 (右上)
        #      |              |
        #   v2 (左下) ---- v3 (右下)
        vertices = []
        faces = []

        for i in range(n):
            # 使用延伸后的顶部点（确保切穿表面）和底部点
            v0 = left_top_ext[i]   # 左上
            v1 = right_top_ext[i]  # 右上
            v2 = left_bottom[i]    # 左下
            v3 = right_bottom[i]   # 右下

            base_idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])

            # 连接相邻截面形成面
            if i > 0:
                prev_base = base_idx - 4
                # 顶面（凹槽开口面 - 这里会被布尔运算移除）
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

    def generate_groove_profile(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        沿路径生成凹槽几何体

        根据 conform_to_surface 参数选择生成方式：
        - True: 使用曲面贴合算法，凹槽顶部与曲面完美贴合
        - False: 使用平行传输算法，保持凹槽宽度一致

        Args:
            path_points: 路径点坐标 (N, 3)
            path_normals: 路径点法向量 (N, 3)
            width: 凹槽宽度，None则使用默认参数
            depth: 凹槽深度，None则使用默认参数

        Returns:
            凹槽的3D网格
        """
        # 如果启用曲面贴合，使用新算法
        if self.params.conform_to_surface:
            return self.generate_groove_profile_curved(
                path_points, path_normals, width, depth
            )

        # 否则使用原始的平行传输算法
        return self._generate_groove_profile_flat(
            path_points, path_normals, width, depth
        )

    def _generate_groove_profile_flat(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        原始的平面凹槽生成算法（使用平行传输保持宽度一致）

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

        # 确保法向量指向模型外部
        mesh_center = self.mesh.centroid
        for i in range(n):
            to_center = mesh_center - path_points[i]
            if np.dot(normals[i], to_center) > 0:
                normals[i] = -normals[i]

        # 限制凹槽深度，防止打穿
        max_safe_depth = self.estimated_thickness * 0.7 if self.estimated_thickness < float('inf') else d
        if d > max_safe_depth and max_safe_depth > 0.1:
            print(f"警告: 凹槽深度 {d:.2f}mm 超过安全限制，调整为 {max_safe_depth:.2f}mm")
            d = max_safe_depth

        # 使用平行传输计算一致的侧向量
        side_vectors = self._compute_parallel_transport_frames(path_points, normals)

        # 计算顶部和底部中心线
        top_center = path_points + ext_h * normals
        bottom_center = path_points - d * normals

        # 生成凹槽顶点 - 矩形横截面
        half_w = w / 2

        vertices = []
        faces = []

        for i in range(n):
            side = side_vectors[i]
            top_p = top_center[i]
            bot_p = bottom_center[i]

            # 矩形横截面的4个顶点
            v0 = top_p + half_w * side  # 顶左
            v1 = top_p - half_w * side  # 顶右
            v2 = bot_p + half_w * side  # 底左
            v3 = bot_p - half_w * side  # 底右

            base_idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])

            # 连接相邻截面形成面
            if i > 0:
                prev_base = base_idx - 4
                # 顶面（凹槽开口面）
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

    def generate_pad_groove(
        self,
        center_point: np.ndarray,
        normal: np.ndarray,
        incoming_direction: np.ndarray,
        pad_length: Optional[float] = None,
        pad_width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        在IR点位置生成方形焊盘凹槽

        方形凹槽的方向由incoming_direction决定，确保路径垂直进入方形边缘

        Args:
            center_point: 方形中心点坐标 (3,)
            normal: 该点的表面法向量 (3,)
            incoming_direction: 路径进入的方向向量 (3,) - 从路径指向中心点
            pad_length: 方形长度（沿路径方向），None则使用默认参数
            pad_width: 方形宽度（垂直于路径），None则使用默认参数
            depth: 凹槽深度，None则使用默认参数

        Returns:
            方形凹槽的3D网格
        """
        pl = pad_length if pad_length is not None else self.params.pad_length
        pw = pad_width if pad_width is not None else self.params.pad_width
        d = depth if depth is not None else self.params.depth
        ext_h = self.params.extension_height

        # 归一化法向量
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # 确保法向量指向外部
        mesh_center = self.mesh.centroid
        to_center = mesh_center - center_point
        if np.dot(normal, to_center) > 0:
            normal = -normal

        # 限制深度
        max_safe_depth = self.estimated_thickness * 0.7 if self.estimated_thickness < float('inf') else d
        if d > max_safe_depth and max_safe_depth > 0.1:
            d = max_safe_depth

        # 计算方形的两个轴向量
        # forward: 沿路径进入方向（方形的"长"方向）
        # side: 垂直于路径（方形的"宽"方向）

        # 将incoming_direction投影到切平面上
        incoming_dir = incoming_direction / (np.linalg.norm(incoming_direction) + 1e-10)
        # 移除法向量分量，得到切平面上的方向
        forward = incoming_dir - np.dot(incoming_dir, normal) * normal
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            # 退化情况：使用任意切向量
            if abs(normal[0]) < 0.9:
                forward = np.array([1.0, 0.0, 0.0])
            else:
                forward = np.array([0.0, 1.0, 0.0])
            forward = forward - np.dot(forward, normal) * normal
        forward = forward / (np.linalg.norm(forward) + 1e-10)

        # side向量垂直于forward和normal
        side = np.cross(normal, forward)
        side = side / (np.linalg.norm(side) + 1e-10)

        # 将中心点投影到曲面
        projected, proj_normals = self._project_points_to_surface(center_point.reshape(1, 3))
        center_on_surface = projected[0]
        surface_normal = proj_normals[0]

        # 计算方形的四个角点（在曲面上）
        half_l = pl / 2.0
        half_w = pw / 2.0

        # 四个角的初始位置（未投影）
        corners_init = np.array([
            center_on_surface + half_l * forward + half_w * side,   # 前右
            center_on_surface + half_l * forward - half_w * side,   # 前左
            center_on_surface - half_l * forward - half_w * side,   # 后左
            center_on_surface - half_l * forward + half_w * side,   # 后右
        ])

        # 将角点投影到曲面
        corners_top, corners_normals = self._project_points_to_surface(corners_init)

        # 计算底部角点
        corners_bottom = corners_top - d * corners_normals

        # 向上延伸以确保切穿表面
        corners_top_ext = corners_top + ext_h * corners_normals

        # 生成方形凹槽网格
        # 顶点顺序：4个顶部角点 + 4个底部角点
        vertices = []
        for i in range(4):
            vertices.append(corners_top_ext[i])
        for i in range(4):
            vertices.append(corners_bottom[i])

        vertices = np.array(vertices)

        # 面定义（注意顺序确保法向量朝外）
        faces = [
            # 顶面（会被布尔运算移除）
            [0, 1, 2], [0, 2, 3],
            # 底面
            [4, 6, 5], [4, 7, 6],
            # 前面
            [0, 4, 5], [0, 5, 1],
            # 后面
            [2, 6, 7], [2, 7, 3],
            # 左面
            [1, 5, 6], [1, 6, 2],
            # 右面
            [3, 7, 4], [3, 4, 0],
        ]

        faces = np.array(faces)

        pad_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        pad_mesh.fix_normals()

        return pad_mesh

    def generate_path_with_perpendicular_end(
        self,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        pad_length: Optional[float] = None,
        width: Optional[float] = None,
        depth: Optional[float] = None
    ) -> Optional[trimesh.Trimesh]:
        """
        生成路径凹槽，末端调整为垂直进入方形焊盘

        路径的起始点是IR点位置，会被截断到方形焊盘边缘，
        并确保最后一段路径垂直进入方形

        Args:
            path_points: 路径点坐标 (N, 3)，第一个点是IR点
            path_normals: 路径点法向量 (N, 3)
            pad_length: 方形焊盘长度，用于计算截断位置
            width: 凹槽宽度
            depth: 凹槽深度

        Returns:
            路径凹槽的3D网格
        """
        if len(path_points) < 2:
            return None

        pl = pad_length if pad_length is not None else self.params.pad_length

        # 计算需要截断的距离（从IR点开始截断半个焊盘长度）
        truncate_dist = pl / 2.0

        # 计算累积弧长
        segments = np.diff(path_points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

        # 找到截断点
        if cumulative_lengths[-1] <= truncate_dist:
            # 路径太短，无法截断
            return None

        # 找到截断后的起始索引
        start_idx = np.searchsorted(cumulative_lengths, truncate_dist)
        if start_idx >= len(path_points):
            return None

        # 在截断点处插值
        if start_idx > 0:
            remaining = truncate_dist - cumulative_lengths[start_idx - 1]
            segment_len = segment_lengths[start_idx - 1]
            if segment_len > 1e-10:
                t = remaining / segment_len
                new_start = path_points[start_idx - 1] + t * segments[start_idx - 1]
                new_normal = path_normals[start_idx - 1] * (1 - t) + path_normals[start_idx] * t
                new_normal = new_normal / (np.linalg.norm(new_normal) + 1e-10)

                # 构建新的路径点数组
                truncated_points = np.vstack([new_start, path_points[start_idx:]])
                truncated_normals = np.vstack([new_normal, path_normals[start_idx:]])
            else:
                truncated_points = path_points[start_idx:]
                truncated_normals = path_normals[start_idx:]
        else:
            truncated_points = path_points
            truncated_normals = path_normals

        if len(truncated_points) < 2:
            return None

        # 确保起始段垂直于方形边缘
        # 调整前几个点使路径垂直进入
        ir_point = path_points[0]  # 原始IR点位置
        first_point = truncated_points[0]  # 截断后的第一个点

        # 计算从截断点到IR点的方向（应该是路径进入方形的方向）
        entry_dir = ir_point - first_point
        entry_dir = entry_dir / (np.linalg.norm(entry_dir) + 1e-10)

        # 添加一小段垂直进入的路径
        perpendicular_length = min(1.0, pl / 4.0)  # 垂直段长度
        perpendicular_point = first_point + perpendicular_length * entry_dir
        perpendicular_normal = truncated_normals[0]

        # 重新构建路径
        final_points = np.vstack([perpendicular_point, truncated_points])
        final_normals = np.vstack([perpendicular_normal, truncated_normals])

        # 使用现有方法生成凹槽
        return self.generate_groove_profile(final_points, final_normals, width, depth)

    def generate_complete_groove(
        self,
        ir_point: np.ndarray,
        ir_normal: np.ndarray,
        path_points: np.ndarray,
        path_normals: np.ndarray,
        width: Optional[float] = None,
        depth: Optional[float] = None,
        pad_length: Optional[float] = None,
        pad_width: Optional[float] = None
    ) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
        """
        生成完整的凹槽：方形焊盘 + 路径凹槽

        Args:
            ir_point: IR点位置
            ir_normal: IR点法向量
            path_points: 完整路径点（从IR点到中心点）
            path_normals: 路径法向量
            width: 路径凹槽宽度
            depth: 凹槽深度
            pad_length: 方形焊盘长度（None则使用默认参数）
            pad_width: 方形焊盘宽度（None则使用默认参数）

        Returns:
            (方形焊盘网格, 路径凹槽网格)
        """
        pad_mesh = None
        path_mesh = None

        # 使用传入的尺寸或默认参数
        pl = pad_length if pad_length is not None else self.params.pad_length
        pw = pad_width if pad_width is not None else self.params.pad_width

        # 计算路径进入方向
        if len(path_points) >= 2:
            # 从第二个点指向第一个点（IR点）的方向
            incoming_dir = path_points[0] - path_points[1]
        else:
            incoming_dir = np.array([1.0, 0.0, 0.0])

        # 生成方形焊盘凹槽
        if self.params.pad_enabled:
            pad_mesh = self.generate_pad_groove(
                ir_point, ir_normal, incoming_dir,
                pl, pw, depth
            )

        # 生成路径凹槽（截断并垂直进入）
        if len(path_points) >= 2:
            if self.params.pad_enabled:
                # 路径需要截断到焊盘边缘
                path_mesh = self.generate_path_with_perpendicular_end(
                    path_points, path_normals,
                    pl, width, depth
                )
            else:
                # 不需要截断
                path_mesh = self.generate_groove_profile(
                    path_points, path_normals, width, depth
                )

        return pad_mesh, path_mesh
