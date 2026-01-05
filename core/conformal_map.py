"""共形映射曲面展开模块"""
import numpy as np
import trimesh
from typing import Tuple, List, Optional
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass


@dataclass
class FlattenResult:
    """展开结果"""
    uv_coords: np.ndarray  # UV坐标 (N, 2)
    faces: np.ndarray  # 面索引
    scale: float  # 缩放因子
    distortion: np.ndarray  # 各面的变形量


class ConformalMapper:
    """共形映射展开器（LSCM算法实现）"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.n_vertices = len(self.vertices)
        self.n_faces = len(self.faces)

    def find_boundary_vertices(self) -> np.ndarray:
        """找到边界顶点"""
        edges = self.mesh.edges_unique
        edge_faces = self.mesh.edges_unique_inverse

        # 统计每条边被多少面共享
        edge_count = np.zeros(len(edges), dtype=int)
        for face_idx in range(self.n_faces):
            for edge_idx in self.mesh.faces_unique_edges[face_idx]:
                edge_count[edge_idx] += 1

        # 边界边只被一个面共享
        boundary_edges = edges[edge_count == 1]
        boundary_vertices = np.unique(boundary_edges.flatten())

        return boundary_vertices

    def compute_lscm(
        self,
        fixed_vertices: Optional[List[int]] = None,
        fixed_positions: Optional[np.ndarray] = None
    ) -> FlattenResult:
        """
        计算最小二乘共形映射

        Args:
            fixed_vertices: 固定的顶点索引
            fixed_positions: 固定顶点的UV坐标

        Returns:
            展开结果
        """
        if fixed_vertices is None:
            # 默认固定边界的两个最远点
            boundary = self.find_boundary_vertices()
            if len(boundary) >= 2:
                # 找边界上最远的两点
                boundary_pts = self.vertices[boundary]
                dists = np.linalg.norm(
                    boundary_pts[:, None] - boundary_pts[None, :],
                    axis=2
                )
                i, j = np.unravel_index(np.argmax(dists), dists.shape)
                fixed_vertices = [boundary[i], boundary[j]]
                fixed_positions = np.array([[0, 0], [1, 0]])
            else:
                # 使用任意两个顶点
                fixed_vertices = [0, 1]
                fixed_positions = np.array([[0, 0], [1, 0]])

        # 构建LSCM方程组
        # 对于每个三角形，共形条件：(u2-u0) + i(v2-v0) = c * ((u1-u0) + i(v1-v0))
        # 其中 c = (p2-p0) / (p1-p0) 在复平面上

        n_free = self.n_vertices - len(fixed_vertices)
        free_vertices = [v for v in range(self.n_vertices) if v not in fixed_vertices]
        vertex_to_free = {v: i for i, v in enumerate(free_vertices)}

        # 构建稀疏矩阵
        # 每个三角形贡献2行（实部和虚部）
        rows = []
        cols = []
        data = []
        rhs = np.zeros(2 * self.n_faces)

        for f_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            p0, p1, p2 = self.vertices[face]

            # 计算局部坐标系
            e1 = p1 - p0
            e2 = p2 - p0
            e1_len = np.linalg.norm(e1)
            if e1_len < 1e-10:
                continue

            e1_normalized = e1 / e1_len

            # 投影到局部2D坐标
            x1 = e1_len
            x2 = np.dot(e2, e1_normalized)
            y2 = np.linalg.norm(e2 - x2 * e1_normalized)

            if y2 < 1e-10:
                continue

            # LSCM系数
            a = (x1 - x2) / y2
            b = x2 / y2
            c = -x1 / y2

            # 实部方程
            row_r = 2 * f_idx
            # 虚部方程
            row_i = 2 * f_idx + 1

            for v, coeff_u, coeff_v in [
                (v0, -a - 1, -c),
                (v1, a, c + 1),
                (v2, 1 - b, b - 1)
            ]:
                if v in fixed_vertices:
                    fix_idx = fixed_vertices.index(v)
                    uv = fixed_positions[fix_idx]
                    rhs[row_r] -= coeff_u * uv[0] - coeff_v * uv[1]
                    rhs[row_i] -= coeff_u * uv[1] + coeff_v * uv[0]
                else:
                    free_idx = vertex_to_free[v]
                    # u 分量
                    rows.append(row_r)
                    cols.append(free_idx)
                    data.append(coeff_u)
                    rows.append(row_i)
                    cols.append(free_idx)
                    data.append(coeff_v)
                    # v 分量
                    rows.append(row_r)
                    cols.append(n_free + free_idx)
                    data.append(-coeff_v)
                    rows.append(row_i)
                    cols.append(n_free + free_idx)
                    data.append(coeff_u)

        if not rows:
            # 无法求解，返回简单投影
            return self._simple_projection()

        A = csr_matrix((data, (rows, cols)), shape=(2 * self.n_faces, 2 * n_free))

        # 最小二乘求解 A^T A x = A^T b
        AtA = A.T @ A
        Atb = A.T @ rhs

        try:
            solution = spsolve(AtA, Atb)
        except Exception:
            return self._simple_projection()

        # 组装UV坐标
        uv = np.zeros((self.n_vertices, 2))
        for i, v in enumerate(free_vertices):
            uv[v, 0] = solution[i]
            uv[v, 1] = solution[n_free + i]
        for i, v in enumerate(fixed_vertices):
            uv[v] = fixed_positions[i]

        # 计算变形
        distortion = self._compute_distortion(uv)

        # 计算缩放因子
        scale = self._compute_scale(uv)

        return FlattenResult(
            uv_coords=uv,
            faces=self.faces.copy(),
            scale=scale,
            distortion=distortion
        )

    def _simple_projection(self) -> FlattenResult:
        """简单投影展开（备选方案）"""
        # 计算主成分方向
        centered = self.vertices - self.vertices.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 投影到前两个主成分
        uv = centered @ eigenvectors[:, -2:]

        distortion = np.zeros(self.n_faces)
        scale = 1.0

        return FlattenResult(
            uv_coords=uv,
            faces=self.faces.copy(),
            scale=scale,
            distortion=distortion
        )

    def _compute_distortion(self, uv: np.ndarray) -> np.ndarray:
        """计算各面的角度变形"""
        distortion = np.zeros(self.n_faces)

        for f_idx, face in enumerate(self.faces):
            # 3D角度
            p = self.vertices[face]
            angles_3d = self._compute_triangle_angles(p)

            # 2D角度
            q = uv[face]
            angles_2d = self._compute_triangle_angles_2d(q)

            # 变形量
            distortion[f_idx] = np.sum(np.abs(angles_3d - angles_2d))

        return distortion

    def _compute_triangle_angles(self, points: np.ndarray) -> np.ndarray:
        """计算3D三角形的角度"""
        angles = np.zeros(3)
        for i in range(3):
            v1 = points[(i + 1) % 3] - points[i]
            v2 = points[(i + 2) % 3] - points[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles[i] = np.arccos(np.clip(cos_angle, -1, 1))
        return angles

    def _compute_triangle_angles_2d(self, points: np.ndarray) -> np.ndarray:
        """计算2D三角形的角度"""
        angles = np.zeros(3)
        for i in range(3):
            v1 = points[(i + 1) % 3] - points[i]
            v2 = points[(i + 2) % 3] - points[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles[i] = np.arccos(np.clip(cos_angle, -1, 1))
        return angles

    def _compute_scale(self, uv: np.ndarray) -> float:
        """计算3D到2D的缩放因子"""
        area_3d = self.mesh.area
        area_2d = 0

        for face in self.faces:
            p = uv[face]
            # 2D三角形面积
            area_2d += 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) -
                (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )

        if area_2d < 1e-10:
            return 1.0

        return np.sqrt(area_3d / area_2d)


def transform_path_to_2d(
    path_3d: np.ndarray,
    mesh: trimesh.Trimesh,
    uv_coords: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    将3D路径转换到2D展开坐标

    Args:
        path_3d: 3D路径点
        mesh: 原始网格
        uv_coords: UV坐标
        scale: 缩放因子

    Returns:
        2D路径点
    """
    path_2d = []

    for point in path_3d:
        # 找到最近的顶点
        distances = np.linalg.norm(mesh.vertices - point, axis=1)
        nearest_idx = np.argmin(distances)
        path_2d.append(uv_coords[nearest_idx] * scale)

    return np.array(path_2d)
