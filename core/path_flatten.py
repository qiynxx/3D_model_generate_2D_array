"""路径展开模块 - 使用保形映射将3D曲面路径展开为2D平面

核心思路：
1. 使用LSCM（最小二乘共形映射）展开整个曲面得到UV坐标
2. 对于每个3D路径点，找到它所在的三角面，计算重心坐标
3. 使用重心坐标在2D UV空间中插值得到对应位置

这种方法保证：
- 展开保持角度关系（共形性）
- 3D路径在曲面上的相对位置关系在2D中被保持
- 展开后的2D PCB形状可以正确贴合回3D曲面
"""
import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class PathFlattenResult:
    """路径展开结果"""
    paths_2d: Dict[str, np.ndarray]  # point_id -> 2D路径点
    ir_points_2d: Dict[str, Tuple[np.ndarray, str, bool]]  # point_id -> (2D坐标, 名称, 是否中心)
    center_2d: np.ndarray  # 中心点2D坐标
    scale: float  # 缩放因子 (mm)
    total_bounds: Tuple[np.ndarray, np.ndarray]  # (min, max) 边界
    uv_coords: Optional[np.ndarray] = None  # UV坐标（用于调试）
    distortion: Optional[np.ndarray] = None  # 变形量


def compute_lscm_uv(mesh: trimesh.Trimesh, center_position: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    使用LSCM算法计算网格的UV展开坐标

    Args:
        mesh: 网格模型
        center_position: 中心点位置（用于确定展开方向）

    Returns:
        (uv_coords, scale_factor) UV坐标数组和缩放因子
    """
    vertices = mesh.vertices
    faces = mesh.faces
    n_vertices = len(vertices)
    n_faces = len(faces)

    print(f"\n  LSCM计算: {n_vertices} 顶点, {n_faces} 面")

    # 找边界顶点
    boundary_vertices = find_boundary_vertices(mesh)
    print(f"  边界顶点数: {len(boundary_vertices)}")

    # 如果没有边界（闭合曲面），使用基于中心点的局部展开
    if len(boundary_vertices) < 2:
        print(f"  检测到闭合曲面，使用局部切平面展开...")
        return unfold_from_center(mesh, center_position)

    # 选择固定点 - 使用边界上距离最远的两个点
    if len(boundary_vertices) >= 2:
        boundary_pts = vertices[boundary_vertices]
        dists = np.linalg.norm(
            boundary_pts[:, None] - boundary_pts[None, :],
            axis=2
        )
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        fixed_vertices = [boundary_vertices[i], boundary_vertices[j]]

        # 使用实际3D距离来设置固定位置
        actual_dist = dists[i, j]
        fixed_positions = np.array([[0.0, 0.0], [actual_dist, 0.0]])
        print(f"  固定顶点: {fixed_vertices[0]}, {fixed_vertices[1]}")
        print(f"  固定顶点3D距离: {actual_dist:.4f}")
        print(f"  固定位置: {fixed_positions}")
    else:
        # 没有足够的边界点，使用简单投影
        print("  警告: 边界点不足，使用简单投影")
        return simple_projection_uv(mesh)

    # 构建LSCM方程
    n_free = n_vertices - len(fixed_vertices)
    free_vertices = [v for v in range(n_vertices) if v not in fixed_vertices]
    vertex_to_free = {v: i for i, v in enumerate(free_vertices)}

    rows = []
    cols = []
    data = []
    rhs = np.zeros(2 * n_faces)

    skipped_faces = 0
    for f_idx, face in enumerate(faces):
        v0, v1, v2 = face
        p0, p1, p2 = vertices[face]

        e1 = p1 - p0
        e2 = p2 - p0
        e1_len = np.linalg.norm(e1)
        if e1_len < 1e-10:
            skipped_faces += 1
            continue

        e1_normalized = e1 / e1_len
        x1 = e1_len
        x2 = np.dot(e2, e1_normalized)
        y2 = np.linalg.norm(e2 - x2 * e1_normalized)

        if y2 < 1e-10:
            skipped_faces += 1
            continue

        a = (x1 - x2) / y2
        b = x2 / y2
        c = -x1 / y2

        row_r = 2 * f_idx
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
                rows.append(row_r)
                cols.append(free_idx)
                data.append(coeff_u)
                rows.append(row_i)
                cols.append(free_idx)
                data.append(coeff_v)
                rows.append(row_r)
                cols.append(n_free + free_idx)
                data.append(-coeff_v)
                rows.append(row_i)
                cols.append(n_free + free_idx)
                data.append(coeff_u)

    print(f"  跳过的退化面: {skipped_faces}")

    if not rows:
        print("  警告: 无有效方程，使用简单投影")
        return simple_projection_uv(mesh)

    A = csr_matrix((data, (rows, cols)), shape=(2 * n_faces, 2 * n_free))
    AtA = A.T @ A
    Atb = A.T @ rhs

    print(f"  矩阵大小: {A.shape}, 非零元素: {A.nnz}")

    try:
        solution = spsolve(AtA, Atb)
        print(f"  求解完成, 解向量范围: [{solution.min():.4f}, {solution.max():.4f}]")
    except Exception as e:
        print(f"  求解失败: {e}, 使用简单投影")
        return simple_projection_uv(mesh)

    # 组装UV坐标
    uv = np.zeros((n_vertices, 2))
    for i, v in enumerate(free_vertices):
        uv[v, 0] = solution[i]
        uv[v, 1] = solution[n_free + i]
    for i, v in enumerate(fixed_vertices):
        uv[v] = fixed_positions[i]

    print(f"  初始UV范围: X=[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}], Y=[{uv[:,1].min():.4f}, {uv[:,1].max():.4f}]")

    # 计算缩放因子（如果固定点已使用真实距离，可能不需要额外缩放）
    scale = compute_scale_factor(mesh, uv)

    # 检查是否需要额外缩放（如果UV尺寸已经接近3D尺寸，scale应该接近1）
    uv_range_x = uv[:,0].max() - uv[:,0].min()
    uv_range_y = uv[:,1].max() - uv[:,1].min()

    # 只有当缩放因子差异很大时才应用
    if abs(scale - 1.0) > 0.1:
        uv = uv * scale
        print(f"  应用缩放 {scale:.4f}")
    else:
        print(f"  缩放因子接近1.0，不需要额外缩放")
        scale = 1.0

    print(f"  最终UV范围: X=[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}], Y=[{uv[:,1].min():.4f}, {uv[:,1].max():.4f}]")

    return uv, scale


def unfold_from_center(mesh: trimesh.Trimesh, center_position: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    从中心点展开闭合曲面 - 使用测地线距离和角度

    对于没有边界的闭合曲面，使用中心点作为参考，
    计算每个顶点相对于中心点的测地线距离和角度来展开。
    """
    print(f"  使用中心点展开法...")

    vertices = mesh.vertices
    n_vertices = len(vertices)

    # 如果没有提供中心点，使用网格中心
    if center_position is None:
        center_position = vertices.mean(axis=0)

    # 找到最近中心点的顶点
    center_distances = np.linalg.norm(vertices - center_position, axis=1)
    center_vertex = np.argmin(center_distances)
    center_pos = vertices[center_vertex]

    # 获取中心点的法向量
    center_normal = mesh.vertex_normals[center_vertex]

    # 构建切平面坐标系
    normal = center_normal / (np.linalg.norm(center_normal) + 1e-10)
    if abs(normal[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, normal)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

    print(f"  中心顶点: {center_vertex}")
    print(f"  中心位置: {center_pos}")
    print(f"  法向量: {normal}")

    # 计算每个顶点到中心的测地线距离（使用Dijkstra）
    geodesic_distances = compute_geodesic_distances(mesh, center_vertex)

    # 计算每个顶点相对于中心的角度（投影到切平面）
    angles = np.zeros(n_vertices)
    for i in range(n_vertices):
        offset = vertices[i] - center_pos
        # 投影到切平面
        x = np.dot(offset, x_axis)
        y = np.dot(offset, y_axis)
        angles[i] = np.arctan2(y, x)

    # 使用极坐标展开，但使用测地线距离作为半径
    uv = np.zeros((n_vertices, 2))
    for i in range(n_vertices):
        r = geodesic_distances[i]  # 使用测地线距离
        theta = angles[i]
        uv[i, 0] = r * np.cos(theta)
        uv[i, 1] = r * np.sin(theta)

    # 不需要额外缩放，因为测地线距离已经是真实的曲面距离
    scale = 1.0

    uv_range_x = uv[:,0].max() - uv[:,0].min()
    uv_range_y = uv[:,1].max() - uv[:,1].min()
    print(f"  展开范围: {uv_range_x:.4f} x {uv_range_y:.4f}")

    return uv, scale


def compute_geodesic_distances(mesh: trimesh.Trimesh, source_vertex: int) -> np.ndarray:
    """
    计算从源顶点到所有其他顶点的测地线距离（使用Dijkstra算法）
    """
    import heapq

    n_vertices = len(mesh.vertices)
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length

    # 构建图
    graph = {i: [] for i in range(n_vertices)}
    for (v1, v2), length in zip(edges, edge_lengths):
        graph[v1].append((v2, length))
        graph[v2].append((v1, length))

    # Dijkstra算法
    distances = np.full(n_vertices, np.inf)
    distances[source_vertex] = 0
    pq = [(0.0, source_vertex)]
    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph[u]:
            if v not in visited:
                new_dist = dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return distances


def find_boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """找到边界顶点"""
    edges = mesh.edges_unique

    edge_count = {}
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_vertices = set()
    for (v1, v2), count in edge_count.items():
        if count == 1:
            boundary_vertices.add(v1)
            boundary_vertices.add(v2)

    return np.array(list(boundary_vertices))


def simple_projection_uv(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """简单投影展开（备选方案）- 使用PCA投影到最大方差平面"""
    print(f"  使用简单PCA投影...")

    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 使用最大的两个特征向量作为投影平面
    # eigh 返回的是升序排列，所以 -2 和 -1 是最大的两个
    uv = centered @ eigenvectors[:, -2:]

    # 计算当前UV范围
    uv_range_x = uv[:, 0].max() - uv[:, 0].min()
    uv_range_y = uv[:, 1].max() - uv[:, 1].min()

    print(f"  原始投影范围: {uv_range_x:.4f} x {uv_range_y:.4f}")

    # 使用边长比例计算更准确的缩放
    edges = mesh.edges_unique
    n_samples = min(500, len(edges))
    indices = np.linspace(0, len(edges)-1, n_samples, dtype=int)

    length_3d_sum = 0.0
    length_2d_sum = 0.0

    for v1, v2 in edges[indices]:
        len_3d = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
        len_2d = np.linalg.norm(uv[v1] - uv[v2])
        if len_3d > 1e-10 and len_2d > 1e-10:
            length_3d_sum += len_3d
            length_2d_sum += len_2d

    if length_2d_sum > 1e-10:
        scale = length_3d_sum / length_2d_sum
    else:
        # 回退到面积比例
        uv_area = max(uv_range_x * uv_range_y, 1e-10)
        scale = np.sqrt(mesh.area / uv_area)

    uv_scaled = uv * scale
    final_range_x = uv_scaled[:, 0].max() - uv_scaled[:, 0].min()
    final_range_y = uv_scaled[:, 1].max() - uv_scaled[:, 1].min()

    print(f"  缩放后范围: {final_range_x:.4f} x {final_range_y:.4f}")
    print(f"  缩放因子: {scale:.4f}")

    return uv_scaled, scale


def compute_scale_factor(mesh: trimesh.Trimesh, uv: np.ndarray) -> float:
    """计算3D到2D的缩放因子 - 基于边长对比，更稳定"""
    edges = mesh.edges_unique

    # 采样边来计算平均缩放比
    n_samples = min(1000, len(edges))
    if n_samples < 10:
        # 太少的边，使用全部
        sample_edges = edges
    else:
        # 均匀采样
        indices = np.linspace(0, len(edges)-1, n_samples, dtype=int)
        sample_edges = edges[indices]

    length_3d_total = 0.0
    length_2d_total = 0.0
    valid_count = 0

    for v1, v2 in sample_edges:
        len_3d = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
        len_2d = np.linalg.norm(uv[v1] - uv[v2])

        if len_3d > 1e-10 and len_2d > 1e-10:
            length_3d_total += len_3d
            length_2d_total += len_2d
            valid_count += 1

    if valid_count < 3 or length_2d_total < 1e-10:
        print(f"  警告: 无法计算缩放因子，使用默认值1.0")
        return 1.0

    scale = length_3d_total / length_2d_total
    print(f"  边长缩放因子: {scale:.4f} (基于{valid_count}条边)")
    return scale


def find_containing_face(
    mesh: trimesh.Trimesh,
    point: np.ndarray
) -> Tuple[int, Tuple[float, float, float]]:
    """
    找到包含给定点的三角面，并计算重心坐标

    Returns:
        (face_index, barycentric_coords)
    """
    # 先找最近的顶点
    distances = np.linalg.norm(mesh.vertices - point, axis=1)
    nearest_vertex = np.argmin(distances)

    # 找包含该顶点的所有面
    vertex_faces = np.where(np.any(mesh.faces == nearest_vertex, axis=1))[0]

    best_face = -1
    best_bary = (1.0, 0.0, 0.0)
    best_dist = float('inf')

    for face_idx in vertex_faces:
        face = mesh.faces[face_idx]
        v0, v1, v2 = mesh.vertices[face]

        # 计算重心坐标
        bary = compute_barycentric(point, v0, v1, v2)

        # 检查是否在三角形内（或附近）
        u, v, w = bary
        if u >= -0.1 and v >= -0.1 and w >= -0.1:
            # 计算到三角形的距离
            projected = u * v0 + v * v1 + w * v2
            dist = np.linalg.norm(point - projected)
            if dist < best_dist:
                best_dist = dist
                best_face = face_idx
                best_bary = bary

    # 如果没找到合适的面，使用最近顶点所在的第一个面
    if best_face == -1 and len(vertex_faces) > 0:
        best_face = vertex_faces[0]
        face = mesh.faces[best_face]
        v0, v1, v2 = mesh.vertices[face]
        best_bary = compute_barycentric(point, v0, v1, v2)

    return best_face, best_bary


def compute_barycentric(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> Tuple[float, float, float]:
    """计算点的重心坐标"""
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0

    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return (1.0, 0.0, 0.0)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u, v, w)


def transform_point_3d_to_2d(
    point_3d: np.ndarray,
    mesh: trimesh.Trimesh,
    uv_coords: np.ndarray
) -> np.ndarray:
    """
    将3D点通过重心坐标插值转换为2D UV坐标

    使用trimesh的nearest.on_surface来可靠地找到包含点的面
    """
    try:
        # 使用trimesh找到最近的面和投影点
        closest, distance, face_idx = mesh.nearest.on_surface([point_3d])
        face_idx = int(face_idx[0])
        closest_point = closest[0]

        if face_idx < 0 or face_idx >= len(mesh.faces):
            # 回退：使用最近顶点
            distances = np.linalg.norm(mesh.vertices - point_3d, axis=1)
            nearest_idx = np.argmin(distances)
            return uv_coords[nearest_idx].copy()

        face = mesh.faces[face_idx]
        v0, v1, v2 = mesh.vertices[face]

        # 使用投影后的点计算重心坐标（更准确）
        bary = compute_barycentric(closest_point, v0, v1, v2)

        # 限制重心坐标在合理范围内
        u, v, w = bary
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        total = u + v + w
        if total > 1e-10:
            u, v, w = u/total, v/total, w/total
        else:
            u, v, w = 1.0, 0.0, 0.0

        uv0, uv1, uv2 = uv_coords[face]
        point_2d = u * uv0 + v * uv1 + w * uv2

        return point_2d

    except Exception as e:
        print(f"  transform_point_3d_to_2d 错误: {e}")
        # 回退：使用最近顶点
        distances = np.linalg.norm(mesh.vertices - point_3d, axis=1)
        nearest_idx = np.argmin(distances)
        return uv_coords[nearest_idx].copy()


def flatten_paths_with_conformal_map(
    paths_3d: Dict[str, np.ndarray],
    ir_points: Dict[str, Tuple[np.ndarray, str, bool]],
    center_position: np.ndarray,
    mesh: trimesh.Trimesh
) -> PathFlattenResult:
    """
    使用保形映射（LSCM）展开3D路径为2D

    方法：
    1. 计算整个曲面的LSCM UV坐标
    2. 对每个3D路径点，通过重心坐标插值得到2D位置
    3. 保持路径的拓扑关系和相对位置

    Args:
        paths_3d: 路径字典 {point_id: 3D路径点数组}
        ir_points: IR点信息 {point_id: (position, name, is_center)}
        center_position: 中心点3D坐标
        mesh: 网格模型

    Returns:
        PathFlattenResult
    """
    print("\n" + "="*60)
    print("开始LSCM保形映射展开")
    print("="*60)

    print(f"\n网格信息:")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.faces)}")
    print(f"  表面积: {mesh.area:.2f}")

    # 计算网格的3D尺寸作为参考
    bounds_3d = mesh.bounds
    size_3d = np.linalg.norm(bounds_3d[1] - bounds_3d[0])
    print(f"  3D模型对角线尺寸: {size_3d:.2f}")

    # 计算LSCM UV坐标（传入中心点用于闭合曲面展开）
    uv_coords, scale = compute_lscm_uv(mesh, center_position)

    # 验证UV坐标是否合理
    uv_range_x = uv_coords[:,0].max() - uv_coords[:,0].min()
    uv_range_y = uv_coords[:,1].max() - uv_coords[:,1].min()
    uv_size = max(uv_range_x, uv_range_y)

    print(f"\nLSCM计算完成:")
    print(f"  UV坐标范围: X=[{uv_coords[:,0].min():.4f}, {uv_coords[:,0].max():.4f}]")
    print(f"  UV坐标范围: Y=[{uv_coords[:,1].min():.4f}, {uv_coords[:,1].max():.4f}]")
    print(f"  UV尺寸: {uv_range_x:.4f} x {uv_range_y:.4f}")
    print(f"  缩放因子: {scale:.4f}")

    # 如果UV尺寸远小于3D尺寸，说明LSCM可能失败了
    if uv_size < size_3d * 0.01:
        print(f"\n  警告: UV尺寸 ({uv_size:.4f}) 远小于3D尺寸 ({size_3d:.2f})")
        print(f"  尝试使用简单投影方法...")
        uv_coords, scale = simple_projection_uv(mesh)
        uv_range_x = uv_coords[:,0].max() - uv_coords[:,0].min()
        uv_range_y = uv_coords[:,1].max() - uv_coords[:,1].min()
        print(f"  简单投影UV尺寸: {uv_range_x:.4f} x {uv_range_y:.4f}")

    paths_2d = {}
    ir_points_2d = {}
    all_points = []
    correction = 1.0  # 初始化校正因子

    # 转换中心点
    center_2d = transform_point_3d_to_2d(center_position, mesh, uv_coords)
    all_points.append(center_2d.copy())
    print(f"\n中心点:")
    print(f"  3D位置: {center_position}")
    print(f"  2D位置: {center_2d}")

    # 转换路径
    print(f"\n路径转换:")
    length_ratios = []

    for point_id, path_3d in paths_3d.items():
        if len(path_3d) < 2:
            continue

        path_2d = []
        for point in path_3d:
            point_2d = transform_point_3d_to_2d(point, mesh, uv_coords)
            path_2d.append(point_2d)
            all_points.append(point_2d.copy())

        path_2d = np.array(path_2d)
        paths_2d[point_id] = path_2d

        # 计算路径长度
        path_3d_length = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))
        path_2d_length = np.sum(np.linalg.norm(np.diff(path_2d, axis=0), axis=1))

        if path_3d_length > 1e-6 and path_2d_length > 1e-6:
            ratio = path_2d_length / path_3d_length
            length_ratios.append(ratio)

        print(f"\n  路径 {point_id[:8]}:")
        print(f"    3D: {len(path_3d)} 点, 长度 {path_3d_length:.2f}")
        print(f"    3D 起点: {path_3d[0]}")
        print(f"    3D 终点: {path_3d[-1]}")
        print(f"    2D: {len(path_2d)} 点, 长度 {path_2d_length:.4f}")
        print(f"    2D 起点: {path_2d[0]}")
        print(f"    2D 终点: {path_2d[-1]}")
        print(f"    长度比例: {path_2d_length/path_3d_length:.4f}" if path_3d_length > 0 else "    长度比例: N/A")

    # 检查2D/3D长度比例是否合理（应该接近1.0）
    if length_ratios:
        avg_ratio = np.mean(length_ratios)
        print(f"\n  平均长度比例: {avg_ratio:.4f}")

        # 如果比例偏差太大（不在0.5-2.0范围内），进行校正
        if avg_ratio < 0.5 or avg_ratio > 2.0:
            correction = 1.0 / avg_ratio
            print(f"  警告: 2D/3D长度比例异常，应用校正因子 {correction:.4f}")

            # 校正所有2D坐标
            for point_id in paths_2d:
                paths_2d[point_id] = paths_2d[point_id] * correction

            # 校正中心点
            center_2d = center_2d * correction

            # 重新收集所有点用于边界计算
            all_points = [center_2d.copy()]
            for path_2d in paths_2d.values():
                all_points.extend(path_2d.tolist())
    else:
        avg_ratio = 1.0

    # 如果没有路径，默认correction为1
    if not length_ratios:
        correction = 1.0

    # 转换IR点（应用相同的校正）
    print(f"\nIR点转换:")
    for point_id, (pos_3d, name, is_center) in ir_points.items():
        pos_2d = transform_point_3d_to_2d(pos_3d, mesh, uv_coords)
        # 应用校正因子
        if correction != 1.0:
            pos_2d = pos_2d * correction
        ir_points_2d[point_id] = (pos_2d.copy(), name, is_center)
        all_points.append(pos_2d.copy())
        print(f"  {name}: 3D={pos_3d} -> 2D={pos_2d}")

    # 计算边界
    if len(all_points) > 0:
        all_points_array = np.array(all_points)
        min_bounds = all_points_array.min(axis=0) - 5
        max_bounds = all_points_array.max(axis=0) + 5
        print(f"\n2D边界: [{min_bounds}] - [{max_bounds}]")
    else:
        min_bounds = np.array([-10, -10])
        max_bounds = np.array([10, 10])

    print("="*60 + "\n")

    return PathFlattenResult(
        paths_2d=paths_2d,
        ir_points_2d=ir_points_2d,
        center_2d=center_2d,
        scale=1.0,  # UV已经缩放到真实尺寸
        total_bounds=(min_bounds, max_bounds),
        uv_coords=uv_coords
    )


def flatten_paths_preserve_geometry(
    paths_3d: Dict[str, np.ndarray],
    ir_points: Dict[str, Tuple[np.ndarray, str, bool]],
    center_position: np.ndarray,
    center_normal: Optional[np.ndarray] = None,
    mesh: Optional[trimesh.Trimesh] = None
) -> PathFlattenResult:
    """
    保持几何关系的路径展开

    如果提供了mesh，使用保形映射
    否则使用弧长保持的展开（向后兼容）
    """
    if mesh is not None:
        return flatten_paths_with_conformal_map(
            paths_3d, ir_points, center_position, mesh
        )

    # 回退到弧长保持的展开
    return flatten_paths_with_arc_length(
        paths_3d, ir_points, center_position, center_normal
    )


def flatten_paths_with_arc_length(
    paths_3d: Dict[str, np.ndarray],
    ir_points: Dict[str, Tuple[np.ndarray, str, bool]],
    center_position: np.ndarray,
    center_normal: Optional[np.ndarray] = None
) -> PathFlattenResult:
    """
    弧长保持的展开（不使用mesh时的回退方案）

    保持每段的长度，按角度排列路径
    """
    paths_2d = {}
    ir_points_2d = {}
    center_2d = np.array([0.0, 0.0])

    if len(paths_3d) == 0:
        return PathFlattenResult(
            paths_2d={},
            ir_points_2d={},
            center_2d=center_2d,
            scale=1.0,
            total_bounds=(np.array([-10, -10]), np.array([10, 10]))
        )

    if center_normal is None:
        center_normal = np.array([0.0, 0.0, 1.0])

    # 构建切平面坐标系
    normal = center_normal / (np.linalg.norm(center_normal) + 1e-10)
    if abs(normal[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, normal)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    y_axis = np.cross(normal, x_axis)

    all_points = [center_2d.copy()]

    # 计算每条路径的初始角度
    path_angles = {}
    for point_id, path_3d in paths_3d.items():
        if len(path_3d) < 2:
            continue
        # 路径从IR点到中心点，取最后一段的方向
        direction = path_3d[0] - center_position
        proj = direction - np.dot(direction, normal) * normal
        proj_norm = np.linalg.norm(proj)
        if proj_norm > 1e-10:
            proj = proj / proj_norm
            x = np.dot(proj, x_axis)
            y = np.dot(proj, y_axis)
            path_angles[point_id] = np.arctan2(y, x)
        else:
            path_angles[point_id] = 0.0

    # 展开每条路径
    for point_id, path_3d in paths_3d.items():
        if len(path_3d) < 2:
            continue

        # 从中心点开始展开（反转路径）
        path_from_center = path_3d[::-1]
        start_angle = path_angles.get(point_id, 0.0)

        path_2d = [center_2d.copy()]
        current_angle = start_angle

        for i in range(1, len(path_from_center)):
            segment_length = np.linalg.norm(path_from_center[i] - path_from_center[i-1])

            # 计算方向变化
            if i >= 2:
                prev_dir = path_from_center[i-1] - path_from_center[i-2]
                curr_dir = path_from_center[i] - path_from_center[i-1]
                prev_len = np.linalg.norm(prev_dir)
                curr_len = np.linalg.norm(curr_dir)

                if prev_len > 1e-10 and curr_len > 1e-10:
                    prev_dir = prev_dir / prev_len
                    curr_dir = curr_dir / curr_len
                    dot = np.clip(np.dot(prev_dir, curr_dir), -1, 1)
                    angle_change = np.arccos(dot)

                    # 使用切平面投影确定转向方向
                    prev_proj = prev_dir - np.dot(prev_dir, normal) * normal
                    curr_proj = curr_dir - np.dot(curr_dir, normal) * normal
                    cross = np.cross(prev_proj, curr_proj)
                    sign = np.sign(np.dot(cross, normal))
                    if sign == 0:
                        sign = 1

                    current_angle += sign * angle_change

            new_point = path_2d[-1] + segment_length * np.array([
                np.cos(current_angle),
                np.sin(current_angle)
            ])
            path_2d.append(new_point)

        # 反转回来（从IR点到中心点）
        path_2d = np.array(path_2d[::-1])
        paths_2d[point_id] = path_2d
        all_points.extend(path_2d.tolist())

        # IR点的2D位置
        ir_info = ir_points.get(point_id)
        if ir_info and len(path_2d) > 0:
            ir_points_2d[point_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])

    # 添加中心点
    for point_id, (pos, name, is_center) in ir_points.items():
        if is_center:
            ir_points_2d[point_id] = (center_2d.copy(), name, True)

    # 计算边界
    all_points_array = np.array(all_points)
    min_bounds = all_points_array.min(axis=0) - 5
    max_bounds = all_points_array.max(axis=0) + 5

    return PathFlattenResult(
        paths_2d=paths_2d,
        ir_points_2d=ir_points_2d,
        center_2d=center_2d,
        scale=1.0,
        total_bounds=(min_bounds, max_bounds)
    )


# 保留旧函数名作为别名
def flatten_paths_with_curvature(*args, **kwargs):
    return flatten_paths_preserve_geometry(*args, **kwargs)


@dataclass
class FPCLayoutResult:
    """FPC布局结果"""
    groove_outlines: Dict[str, np.ndarray]  # point_id -> 凹槽轮廓（闭合多边形）
    merged_outline: Optional[np.ndarray]  # 合并后的总轮廓
    center_pad: np.ndarray  # 中心焊盘轮廓
    ir_pads: Dict[str, np.ndarray]  # point_id -> IR点焊盘轮廓（圆形或方形）
    rectangular_pads: Dict[str, np.ndarray]  # point_id -> 方形焊盘轮廓
    total_bounds: Tuple[np.ndarray, np.ndarray]  # 边界


def generate_path_offset(path_2d: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据路径生成左右偏移线

    Args:
        path_2d: 2D路径点 (N, 2)
        width: 凹槽宽度

    Returns:
        (left_offset, right_offset) 左侧和右侧偏移线
    """
    n = len(path_2d)
    if n < 2:
        return np.array([]), np.array([])

    half_width = width / 2.0

    # 计算每个点的法向量（垂直于路径方向）
    normals = np.zeros((n, 2))

    for i in range(n):
        if i == 0:
            tangent = path_2d[1] - path_2d[0]
        elif i == n - 1:
            tangent = path_2d[-1] - path_2d[-2]
        else:
            tangent = path_2d[i + 1] - path_2d[i - 1]

        # 归一化切向量
        length = np.linalg.norm(tangent)
        if length < 1e-10:
            normals[i] = np.array([0.0, 1.0])
        else:
            tangent = tangent / length
            # 法向量 = 切向量旋转90度
            normals[i] = np.array([-tangent[1], tangent[0]])

    # 生成左右偏移线
    left_offset = path_2d + half_width * normals
    right_offset = path_2d - half_width * normals

    return left_offset, right_offset


def create_groove_outline(path_2d: np.ndarray, width: float) -> np.ndarray:
    """
    根据路径和宽度创建凹槽轮廓（闭合多边形）

    Args:
        path_2d: 2D路径点
        width: 凹槽宽度

    Returns:
        闭合轮廓点（逆时针方向）
    """
    left, right = generate_path_offset(path_2d, width)

    if len(left) == 0:
        return np.array([])

    # 创建闭合轮廓：左侧正向 + 右侧反向
    outline = np.vstack([left, right[::-1]])

    return outline


def create_circular_pad(center: np.ndarray, radius: float, n_points: int = 32) -> np.ndarray:
    """
    创建圆形焊盘轮廓

    Args:
        center: 中心点坐标
        radius: 半径
        n_points: 采样点数

    Returns:
        圆形轮廓点
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return circle


def create_rectangular_pad(center: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    创建矩形焊盘轮廓

    Args:
        center: 中心点坐标
        width: 宽度
        height: 高度

    Returns:
        矩形轮廓点（逆时针）
    """
    hw = width / 2
    hh = height / 2
    return np.array([
        [center[0] - hw, center[1] - hh],
        [center[0] + hw, center[1] - hh],
        [center[0] + hw, center[1] + hh],
        [center[0] - hw, center[1] + hh]
    ])


def create_oriented_rectangular_pad(
    center: np.ndarray,
    length: float,
    width: float,
    direction: np.ndarray
) -> np.ndarray:
    """
    创建带方向的矩形焊盘轮廓

    Args:
        center: 中心点坐标 (2,)
        length: 长度（沿方向）
        width: 宽度（垂直于方向）
        direction: 方向向量 (2,)

    Returns:
        矩形轮廓点（逆时针）
    """
    # 归一化方向向量
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        # 默认方向
        forward = np.array([1.0, 0.0])
    else:
        forward = direction / dir_norm

    # 垂直方向
    side = np.array([-forward[1], forward[0]])

    half_l = length / 2.0
    half_w = width / 2.0

    # 四个角点
    corners = np.array([
        center + half_l * forward + half_w * side,   # 前右
        center + half_l * forward - half_w * side,   # 前左
        center - half_l * forward - half_w * side,   # 后左
        center - half_l * forward + half_w * side,   # 后右
    ])

    return corners


def merge_overlapping_outlines(outlines: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    合并重叠的轮廓

    使用简单的凸包方法或点集并集
    对于复杂情况可能需要使用shapely库
    """
    if not outlines:
        return None

    # 收集所有点
    all_points = []
    for outline in outlines:
        if len(outline) > 0:
            all_points.extend(outline.tolist())

    if not all_points:
        return None

    all_points = np.array(all_points)

    # 简单方法：使用凸包
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(all_points)
        return all_points[hull.vertices]
    except:
        # 如果凸包失败，返回边界框
        min_pt = all_points.min(axis=0)
        max_pt = all_points.max(axis=0)
        return np.array([
            [min_pt[0], min_pt[1]],
            [max_pt[0], min_pt[1]],
            [max_pt[0], max_pt[1]],
            [min_pt[0], max_pt[1]]
        ])


def generate_fpc_layout(
    flatten_result: PathFlattenResult,
    groove_width: float = 1.0,
    pad_radius: float = 2.0,
    center_pad_radius: float = 3.0,
    rectangular_pad_enabled: bool = False,
    rectangular_pad_length: float = 3.0,
    rectangular_pad_width: float = 2.0,
    point_pad_sizes: Optional[Dict[str, Tuple[float, float]]] = None
) -> FPCLayoutResult:
    """
    根据展开结果生成FPC布局图

    Args:
        flatten_result: 路径展开结果
        groove_width: 凹槽（走线）宽度
        pad_radius: IR点焊盘半径（圆形焊盘时使用）
        center_pad_radius: 中心焊盘半径（圆形焊盘时使用）
        rectangular_pad_enabled: 是否使用方形焊盘
        rectangular_pad_length: 方形焊盘长度（默认值）
        rectangular_pad_width: 方形焊盘宽度（默认值）
        point_pad_sizes: 每个点的单独焊盘尺寸 {point_id: (length, width)}

    Returns:
        FPC布局结果
    """
    groove_outlines = {}
    ir_pads = {}
    rectangular_pads = {}
    all_outline_points = []

    # 为每条路径生成凹槽轮廓
    for point_id, path_2d in flatten_result.paths_2d.items():
        if len(path_2d) >= 2:
            outline = create_groove_outline(path_2d, groove_width)
            if len(outline) > 0:
                groove_outlines[point_id] = outline
                all_outline_points.extend(outline.tolist())

    # 生成IR点焊盘
    for point_id, (pos, name, is_center) in flatten_result.ir_points_2d.items():
        if is_center:
            continue  # 中心点单独处理

        if rectangular_pad_enabled:
            # 获取该点的焊盘尺寸
            if point_pad_sizes and point_id in point_pad_sizes:
                pad_l, pad_w = point_pad_sizes[point_id]
            else:
                pad_l, pad_w = rectangular_pad_length, rectangular_pad_width

            # 计算路径方向（从IR点到中心点）
            path_2d = flatten_result.paths_2d.get(point_id)
            if path_2d is not None and len(path_2d) >= 2:
                # 路径方向：从第一个点（IR点）到第二个点
                direction = path_2d[1] - path_2d[0]
            else:
                # 默认方向：指向中心点
                direction = flatten_result.center_2d - pos

            # 生成带方向的方形焊盘
            rect_pad = create_oriented_rectangular_pad(
                pos, pad_l, pad_w, direction
            )
            rectangular_pads[point_id] = rect_pad
            all_outline_points.extend(rect_pad.tolist())

            # 同时也存储到 ir_pads（用于兼容性）
            ir_pads[point_id] = rect_pad
        else:
            # 使用圆形焊盘
            pad = create_circular_pad(pos, pad_radius)
            ir_pads[point_id] = pad
            all_outline_points.extend(pad.tolist())

    # 生成中心焊盘
    if rectangular_pad_enabled:
        # 中心点也使用方形焊盘
        # 获取中心点的焊盘尺寸
        center_point_id = None
        for point_id, (pos, name, is_center) in flatten_result.ir_points_2d.items():
            if is_center:
                center_point_id = point_id
                break

        if point_pad_sizes and center_point_id and center_point_id in point_pad_sizes:
            center_pad_l, center_pad_w = point_pad_sizes[center_point_id]
        else:
            # 默认中心点焊盘尺寸（稍大一些）
            center_pad_l = rectangular_pad_length * 1.5
            center_pad_w = rectangular_pad_width * 1.5

        # 中心点的方向：使用第一条路径的方向作为参考，或默认向右
        center_direction = np.array([1.0, 0.0])
        if flatten_result.paths_2d:
            first_path = list(flatten_result.paths_2d.values())[0]
            if len(first_path) >= 2:
                # 方向从中心点向外
                center_direction = first_path[-2] - first_path[-1]

        center_pad = create_oriented_rectangular_pad(
            flatten_result.center_2d, center_pad_l, center_pad_w, center_direction
        )
    else:
        center_pad = create_circular_pad(flatten_result.center_2d, center_pad_radius)

    all_outline_points.extend(center_pad.tolist())

    # 合并所有轮廓
    all_outlines = list(groove_outlines.values()) + list(ir_pads.values()) + [center_pad]
    merged_outline = merge_overlapping_outlines(all_outlines)

    # 计算边界
    if all_outline_points:
        all_points = np.array(all_outline_points)
        min_bounds = all_points.min(axis=0) - 5
        max_bounds = all_points.max(axis=0) + 5
    else:
        min_bounds = np.array([-10, -10])
        max_bounds = np.array([10, 10])

    return FPCLayoutResult(
        groove_outlines=groove_outlines,
        merged_outline=merged_outline,
        center_pad=center_pad,
        ir_pads=ir_pads,
        rectangular_pads=rectangular_pads,
        total_bounds=(min_bounds, max_bounds)
    )
