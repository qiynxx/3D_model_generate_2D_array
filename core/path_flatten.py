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
from matplotlib.tri import Triangulation


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

    # 始终应用缩放因子以确保精确的尺寸匹配
    # 移除了0.1阈值限制，因为即使是小的缩放差异也会导致FPC贴合问题
    if abs(scale - 1.0) > 1e-6:  # 只有当缩放因子不是精确的1.0时才应用
        uv = uv * scale
        print(f"  应用缩放 {scale:.4f}")
    else:
        print(f"  缩放因子为1.0，无需缩放")
        scale = 1.0

    print(f"  最终UV范围: X=[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}], Y=[{uv[:,1].min():.4f}, {uv[:,1].max():.4f}]")

    return uv, scale


def unfold_from_center(mesh: trimesh.Trimesh, center_position: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    从中心点展开闭合曲面 - 使用ABF（基于角度的展开）方法

    对于闭合曲面，使用局部保角展开。对于每个三角形，
    保持其三个角度不变，从中心向外逐步展开。
    """
    print(f"  使用基于角度的展开法（ABF-like）...")

    vertices = mesh.vertices
    n_vertices = len(vertices)
    faces = mesh.faces
    n_faces = len(faces)

    # 如果没有提供中心点，使用网格中心
    if center_position is None:
        center_position = vertices.mean(axis=0)

    # 找到最近中心点的顶点和面
    center_distances = np.linalg.norm(vertices - center_position, axis=1)
    center_vertex = np.argmin(center_distances)

    print(f"  中心顶点: {center_vertex}")
    print(f"  中心位置: {vertices[center_vertex]}")

    # 预计算每个面的三个角度和边长
    face_angles = np.zeros((n_faces, 3))  # 每个面的三个角
    face_edge_lengths = np.zeros((n_faces, 3))  # 每个面的三个边长

    for fi, face in enumerate(faces):
        v0, v1, v2 = vertices[face]

        # 边向量
        e0 = v1 - v0  # 对着顶点2
        e1 = v2 - v1  # 对着顶点0
        e2 = v0 - v2  # 对着顶点1

        # 边长
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        face_edge_lengths[fi] = [l0, l1, l2]

        # 角度（使用余弦定理）
        # 角0在顶点0，对边是e1（长度l1）
        cos0 = (l2**2 + l0**2 - l1**2) / (2 * l2 * l0 + 1e-10)
        cos1 = (l0**2 + l1**2 - l2**2) / (2 * l0 * l1 + 1e-10)
        cos2 = (l1**2 + l2**2 - l0**2) / (2 * l1 * l2 + 1e-10)

        face_angles[fi, 0] = np.arccos(np.clip(cos0, -1, 1))
        face_angles[fi, 1] = np.arccos(np.clip(cos1, -1, 1))
        face_angles[fi, 2] = np.arccos(np.clip(cos2, -1, 1))

    # 构建顶点-面邻接表
    vertex_faces = {i: [] for i in range(n_vertices)}
    for face_idx, face in enumerate(faces):
        for local_idx, v in enumerate(face):
            vertex_faces[v].append((face_idx, local_idx))

    # 构建边-面邻接表
    edge_faces = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append((face_idx, i))

    # 初始化UV坐标
    uv = np.full((n_vertices, 2), np.nan)
    uv[center_vertex] = [0.0, 0.0]

    # 找到中心顶点的第一个相邻面
    first_face_idx, local_idx = vertex_faces[center_vertex][0]
    first_face = faces[first_face_idx]

    # 放置第一个三角形
    v0_idx = first_face[local_idx]  # 中心顶点
    v1_idx = first_face[(local_idx + 1) % 3]
    v2_idx = first_face[(local_idx + 2) % 3]

    # 获取边长
    l01 = np.linalg.norm(vertices[v1_idx] - vertices[v0_idx])
    l02 = np.linalg.norm(vertices[v2_idx] - vertices[v0_idx])

    # 获取角度
    angle0 = face_angles[first_face_idx, local_idx]

    # 放置第一个三角形（v0在原点，v1在x正方向）
    uv[v0_idx] = [0.0, 0.0]
    uv[v1_idx] = [l01, 0.0]
    uv[v2_idx] = [l02 * np.cos(angle0), l02 * np.sin(angle0)]

    # 已处理的面
    processed_faces = {first_face_idx}

    # 使用BFS展开其他面
    import heapq
    face_queue = []

    # 添加第一个面的相邻面
    for i in range(3):
        v1, v2 = first_face[i], first_face[(i + 1) % 3]
        edge = (min(v1, v2), max(v1, v2))
        for neighbor_face_idx, _ in edge_faces.get(edge, []):
            if neighbor_face_idx not in processed_faces:
                face_center = vertices[faces[neighbor_face_idx]].mean(axis=0)
                dist = np.linalg.norm(face_center - center_position)
                heapq.heappush(face_queue, (dist, neighbor_face_idx, edge))

    while face_queue:
        _, face_idx, shared_edge = heapq.heappop(face_queue)

        if face_idx in processed_faces:
            continue

        face = faces[face_idx]

        # 找到共享边和第三个顶点
        v_shared = [v for v in face if v in shared_edge]
        v_new = [v for v in face if v not in shared_edge][0]

        if len(v_shared) != 2:
            continue

        v_s1, v_s2 = v_shared

        # 检查共享边的两个顶点是否都有UV
        if np.isnan(uv[v_s1][0]) or np.isnan(uv[v_s2][0]):
            # 重新加入队列，稍后处理
            face_center = vertices[face].mean(axis=0)
            dist = np.linalg.norm(face_center - center_position) + 100
            heapq.heappush(face_queue, (dist, face_idx, shared_edge))
            continue

        # 计算新顶点的UV坐标
        # 获取边长
        len_s1_s2 = np.linalg.norm(vertices[v_s2] - vertices[v_s1])
        len_s1_new = np.linalg.norm(vertices[v_new] - vertices[v_s1])
        len_s2_new = np.linalg.norm(vertices[v_new] - vertices[v_s2])

        # 共享边在UV空间的方向
        uv_s1 = uv[v_s1]
        uv_s2 = uv[v_s2]
        edge_dir = uv_s2 - uv_s1
        edge_len = np.linalg.norm(edge_dir)

        if edge_len < 1e-10:
            uv[v_new] = uv_s1 + [len_s1_new, 0.0]
        else:
            edge_dir = edge_dir / edge_len

            # 计算新顶点相对于共享边的位置
            # 使用余弦定理计算角度
            cos_angle = (len_s1_new**2 + len_s1_s2**2 - len_s2_new**2) / (2 * len_s1_new * len_s1_s2 + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            # 垂直于边的方向
            perp = np.array([-edge_dir[1], edge_dir[0]])

            # 新顶点的UV坐标
            dx = len_s1_new * cos_angle
            dy = len_s1_new * np.sin(angle)

            # 确定方向：新顶点应该在共享边的另一侧（与之前展开的面相反）
            # 使用叉积判断
            new_uv_pos = uv_s1 + dx * edge_dir + dy * perp
            new_uv_neg = uv_s1 + dx * edge_dir - dy * perp

            # 找到在这个边上已经展开的另一个面的第三个顶点
            other_v = None
            for other_face_idx, _ in edge_faces.get(shared_edge, []):
                if other_face_idx in processed_faces:
                    other_face = faces[other_face_idx]
                    other_v = [v for v in other_face if v not in shared_edge]
                    if other_v:
                        other_v = other_v[0]
                        break

            if other_v is not None and not np.isnan(uv[other_v][0]):
                # 选择与已有顶点不同侧的位置
                other_uv = uv[other_v]
                dist_pos = np.linalg.norm(new_uv_pos - other_uv)
                dist_neg = np.linalg.norm(new_uv_neg - other_uv)

                if dist_pos > dist_neg:
                    uv[v_new] = new_uv_pos
                else:
                    uv[v_new] = new_uv_neg
            else:
                # 默认选择正方向
                uv[v_new] = new_uv_pos

        processed_faces.add(face_idx)

        # 添加相邻面到队列
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            for neighbor_face_idx, _ in edge_faces.get(edge, []):
                if neighbor_face_idx not in processed_faces:
                    face_center = vertices[faces[neighbor_face_idx]].mean(axis=0)
                    dist = np.linalg.norm(face_center - center_position)
                    heapq.heappush(face_queue, (dist, neighbor_face_idx, edge))

    # 处理可能遗漏的顶点
    for i in range(n_vertices):
        if np.isnan(uv[i][0]):
            valid_mask = ~np.isnan(uv[:, 0])
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                dists = np.linalg.norm(vertices[valid_indices] - vertices[i], axis=1)
                nearest = valid_indices[np.argmin(dists)]
                uv[i] = uv[nearest]
            else:
                uv[i] = [0.0, 0.0]

    scale = 1.0

    uv_range_x = uv[:, 0].max() - uv[:, 0].min()
    uv_range_y = uv[:, 1].max() - uv[:, 1].min()
    print(f"  展开范围: {uv_range_x:.4f} x {uv_range_y:.4f}")

    return uv, scale


def _compute_angle_from_lengths(a, b, c):
    """根据三边长度计算角A（对边为a）的角度"""
    cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def flatten_paths_direct(
    paths_3d: Dict[str, np.ndarray],
    ir_points: Dict[str, Tuple[np.ndarray, str, bool]],
    center_position: np.ndarray,
    mesh: trimesh.Trimesh
) -> PathFlattenResult:
    """
    凹槽带状展开算法 - 只展开路径周围的窄带区域

    不展开整个曲面，而是将每条路径作为一个独立的带状区域展开：
    1. 沿路径计算累积弧长
    2. 保持每段的长度和转角
    3. 凹槽宽度方向直接保持（窄带近似）

    这种方法：
    - 精确保持路径长度
    - 保持路径的弯曲形状
    - 避免闭合曲面展开的问题
    """
    print("\n" + "="*60)
    print("使用凹槽带状展开算法")
    print("="*60)

    paths_2d = {}
    ir_points_2d = {}
    center_2d = np.array([0.0, 0.0])
    all_points = [center_2d.copy()]

    if len(paths_3d) == 0:
        return PathFlattenResult(
            paths_2d={},
            ir_points_2d={},
            center_2d=center_2d,
            scale=1.0,
            total_bounds=(np.array([-10, -10]), np.array([10, 10]))
        )

    # 获取中心点的表面法向量
    try:
        closest, distance, face_idx = mesh.nearest.on_surface([center_position])
        center_face = mesh.faces[int(face_idx[0])]
        v0, v1, v2 = mesh.vertices[center_face]
        e1 = v1 - v0
        e2 = v2 - v0
        center_normal = np.cross(e1, e2)
        center_normal = center_normal / (np.linalg.norm(center_normal) + 1e-10)
    except:
        center_normal = np.array([0.0, 0.0, 1.0])

    # 构建局部坐标系
    if abs(center_normal[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, center_normal)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    y_axis = np.cross(center_normal, x_axis)

    # 计算每条路径在中心点的切线方向角度
    path_angles = {}
    for point_id, path_3d in paths_3d.items():
        if len(path_3d) < 2:
            continue

        # 使用路径到达中心时的切线方向
        last_segment = path_3d[-1] - path_3d[-2]
        outward_direction = -last_segment

        # 投影到中心点的切平面
        proj = outward_direction - np.dot(outward_direction, center_normal) * center_normal
        proj_len = np.linalg.norm(proj)

        if proj_len > 1e-10:
            proj = proj / proj_len
            x = np.dot(proj, x_axis)
            y = np.dot(proj, y_axis)
            path_angles[point_id] = np.arctan2(y, x)
        else:
            path_angles[point_id] = 0.0

    print(f"\n路径数: {len(paths_3d)}")
    print(f"路径角度:")
    for pid, angle in path_angles.items():
        print(f"  {pid[:8]}: {np.degrees(angle):.1f}°")

    # ============ 分两阶段展开 ============
    # 阶段1：先展开主要路径（星形或串联），确定IR点2D坐标
    # 阶段2：再展开自定义路径，调整端点以匹配已确定的坐标

    custom_paths = {}  # 保存自定义路径，稍后处理

    # 对串联段路径进行排序，确保按顺序处理
    sorted_path_items = []
    serial_seg_items = []
    for point_id, path_3d in paths_3d.items():
        if "_to_" in point_id:
            custom_paths[point_id] = path_3d
        elif point_id.startswith("serial_seg_"):
            serial_seg_items.append((point_id, path_3d))
        else:
            sorted_path_items.append((point_id, path_3d))

    # 串联段按编号排序
    serial_seg_items.sort(key=lambda x: int(x[0].replace("serial_seg_", "")))
    sorted_path_items.extend(serial_seg_items)

    # 阶段1：展开主要路径
    print(f"\n=== 阶段1：展开主要路径 ===")

    # 记录上一段的终点2D位置，用于连接后续段
    last_segment_end_2d = None
    last_segment_end_3d = None

    for point_id, path_3d in sorted_path_items:
        if len(path_3d) < 2:
            continue

        initial_angle = path_angles.get(point_id, 0.0)

        # 路径从IR点到中心点，反转为从中心开始
        path_from_center = path_3d[::-1]

        # 使用带状展开方法
        path_2d = _unfold_groove_strip(path_from_center, mesh, initial_angle)

        # 反转回来（从IR点到中心点）
        path_2d = path_2d[::-1]

        # 对于串联段，检查是否需要调整位置以与前一段连接
        if point_id.startswith("serial_seg_") and last_segment_end_2d is not None:
            # 检查当前段的起点是否与上一段的终点接近（共享点）
            current_start_3d = path_3d[0]
            dist_to_last_end = np.linalg.norm(current_start_3d - last_segment_end_3d)
            if dist_to_last_end < 5.0:  # 5mm内认为是共享点
                # 平移整个路径使起点对齐
                offset = last_segment_end_2d - path_2d[0]
                path_2d = path_2d + offset
                print(f"    调整段 {point_id} 以连接前一段，偏移: {offset}")

        paths_2d[point_id] = path_2d
        all_points.extend(path_2d.tolist())

        # 记录当前段的终点
        last_segment_end_2d = path_2d[-1].copy()
        last_segment_end_3d = path_3d[-1].copy()

        # 验证长度
        path_3d_length = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))
        path_2d_length = np.sum(np.linalg.norm(np.diff(path_2d, axis=0), axis=1))
        ratio = path_2d_length / path_3d_length if path_3d_length > 0 else 0

        print(f"\n  主路径 {point_id[:16]}:")
        print(f"    3D长度: {path_3d_length:.4f}")
        print(f"    2D长度: {path_2d_length:.4f}")
        print(f"    长度比例: {ratio:.6f} (误差: {abs(ratio-1)*100:.4f}%)")

        # IR点2D位置
        if point_id == "serial" or point_id.startswith("serial_seg_"):
            # 串联模式：为每个IR点找到路径上最近的点
            for ir_id, (ir_pos_3d, ir_name, ir_is_center) in ir_points.items():
                if ir_is_center:
                    continue  # 中心点单独处理
                if ir_id in ir_points_2d:
                    continue  # 已经处理过的点跳过
                # 找到3D路径上离IR点最近的点
                distances = np.linalg.norm(path_3d - ir_pos_3d, axis=1)
                min_distance = np.min(distances)
                # 只有当点足够接近路径时才使用这个路径的2D坐标
                if min_distance < 5.0:  # 5mm阈值，可调整
                    nearest_idx = np.argmin(distances)
                    # 使用对应的2D坐标
                    ir_points_2d[ir_id] = (path_2d[nearest_idx].copy(), ir_name, ir_is_center)
                    print(f"    IR点 {ir_name}: 2D={path_2d[nearest_idx]} (距离路径 {min_distance:.2f}mm)")
        else:
            # 星形模式：路径ID就是IR点ID
            ir_info = ir_points.get(point_id)
            if ir_info:
                ir_points_2d[point_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])
                print(f"    IR点 {ir_info[1]}: 2D={path_2d[0]}")

    # 添加中心点（在处理自定义路径之前）
    for point_id, (pos, name, is_center) in ir_points.items():
        if is_center:
            ir_points_2d[point_id] = (center_2d.copy(), name, True)
            print(f"  中心点 {name}: 2D={center_2d}")

    # 阶段2：展开自定义路径
    print(f"\n=== 阶段2：展开自定义路径 ({len(custom_paths)} 条) ===")
    for point_id, path_3d in custom_paths.items():
        if len(path_3d) < 2:
            continue

        parts = point_id.split("_to_")
        if len(parts) != 2:
            continue
        from_id, to_id = parts

        print(f"\n  自定义路径 {point_id[:24]}:")
        print(f"    起点ID: {from_id[:8]}, 终点ID: {to_id[:8]}")

        # 检查端点是否已有2D坐标
        from_has_2d = from_id in ir_points_2d
        to_has_2d = to_id in ir_points_2d

        print(f"    起点已确定: {from_has_2d}, 终点已确定: {to_has_2d}")

        # 先正常展开路径
        initial_angle = path_angles.get(point_id, 0.0)
        path_from_center = path_3d[::-1]
        path_2d = _unfold_groove_strip(path_from_center, mesh, initial_angle)
        path_2d = path_2d[::-1]

        # 根据已确定的端点调整路径
        if from_has_2d and to_has_2d:
            # 两端都已确定，线性变换路径使端点匹配
            from_2d_target = ir_points_2d[from_id][0]
            to_2d_target = ir_points_2d[to_id][0]
            path_2d = _adjust_path_endpoints(path_2d, from_2d_target, to_2d_target)
            print(f"    调整端点: {path_2d[0]} -> {path_2d[-1]}")
        elif from_has_2d:
            # 只有起点已确定，平移路径
            from_2d_target = ir_points_2d[from_id][0]
            offset = from_2d_target - path_2d[0]
            path_2d = path_2d + offset
            # 设置终点
            if to_id in ir_points and to_id not in ir_points_2d:
                ir_info = ir_points[to_id]
                ir_points_2d[to_id] = (path_2d[-1].copy(), ir_info[1], ir_info[2])
            print(f"    平移到起点: {path_2d[0]}")
        elif to_has_2d:
            # 只有终点已确定，平移路径
            to_2d_target = ir_points_2d[to_id][0]
            offset = to_2d_target - path_2d[-1]
            path_2d = path_2d + offset
            # 设置起点
            if from_id in ir_points and from_id not in ir_points_2d:
                ir_info = ir_points[from_id]
                ir_points_2d[from_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])
            print(f"    平移到终点: {path_2d[-1]}")
        else:
            # 两端都未确定，使用路径本身的坐标
            if from_id in ir_points:
                ir_info = ir_points[from_id]
                ir_points_2d[from_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])
            if to_id in ir_points:
                ir_info = ir_points[to_id]
                ir_points_2d[to_id] = (path_2d[-1].copy(), ir_info[1], ir_info[2])
            print(f"    使用原始坐标: {path_2d[0]} -> {path_2d[-1]}")

        paths_2d[point_id] = path_2d
        all_points.extend(path_2d.tolist())

        # 验证长度
        path_3d_length = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))
        path_2d_length = np.sum(np.linalg.norm(np.diff(path_2d, axis=0), axis=1))
        print(f"    3D长度: {path_3d_length:.4f}, 2D长度: {path_2d_length:.4f}")

    # 计算边界
    if len(all_points) > 0:
        all_points_array = np.array(all_points)
        min_bounds = all_points_array.min(axis=0) - 5
        max_bounds = all_points_array.max(axis=0) + 5
        print(f"\n2D边界: [{min_bounds}] - [{max_bounds}]")
    else:
        min_bounds = np.array([-10, -10])
        max_bounds = np.array([10, 10])

    # 为动画生成UV坐标
    print(f"\n生成动画用UV坐标...")
    uv_coords = _generate_radial_uv_for_animation(mesh, center_position, center_normal, x_axis, y_axis)
    print(f"  UV坐标生成完成: {len(uv_coords)} 顶点")

    print("="*60 + "\n")

    return PathFlattenResult(
        paths_2d=paths_2d,
        ir_points_2d=ir_points_2d,
        center_2d=center_2d,
        scale=1.0,
        total_bounds=(min_bounds, max_bounds),
        uv_coords=uv_coords
    )


def _adjust_path_endpoints(
    path_2d: np.ndarray,
    from_target: np.ndarray,
    to_target: np.ndarray
) -> np.ndarray:
    """
    调整2D路径的端点以匹配目标坐标

    使用仿射变换，保持路径形状的同时调整端点位置

    Args:
        path_2d: 原始2D路径点数组
        from_target: 起点目标坐标
        to_target: 终点目标坐标

    Returns:
        调整后的2D路径
    """
    if len(path_2d) < 2:
        return path_2d

    # 原始端点
    from_orig = path_2d[0]
    to_orig = path_2d[-1]

    # 原始向量
    orig_vec = to_orig - from_orig
    orig_length = np.linalg.norm(orig_vec)

    # 目标向量
    target_vec = to_target - from_target
    target_length = np.linalg.norm(target_vec)

    if orig_length < 1e-10 or target_length < 1e-10:
        # 退化情况：简单平移
        return path_2d + (from_target - from_orig)

    # 计算缩放和旋转
    scale = target_length / orig_length

    # 计算旋转角度
    orig_angle = np.arctan2(orig_vec[1], orig_vec[0])
    target_angle = np.arctan2(target_vec[1], target_vec[0])
    rotation = target_angle - orig_angle

    # 创建旋转矩阵
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

    # 变换路径：先平移到原点，旋转，缩放，再平移到目标起点
    adjusted_path = []
    for point in path_2d:
        # 平移到原点
        p = point - from_orig
        # 旋转
        p = rot_matrix @ p
        # 缩放
        p = p * scale
        # 平移到目标起点
        p = p + from_target
        adjusted_path.append(p)

    return np.array(adjusted_path)


def _unfold_groove_strip(
    path_3d: np.ndarray,
    mesh: trimesh.Trimesh,
    initial_angle: float
) -> np.ndarray:
    """
    展开单条凹槽带状区域

    将3D路径展开为2D，保持：
    1. 每段的精确长度
    2. 相邻段之间的转角（测地曲率）

    Args:
        path_3d: 3D路径点（从中心开始）
        mesh: 3D网格（用于获取表面法向量）
        initial_angle: 初始方向角度

    Returns:
        path_2d: 2D路径点数组
    """
    n_points = len(path_3d)
    if n_points < 2:
        return np.array([[0.0, 0.0]])

    path_2d = np.zeros((n_points, 2))
    path_2d[0] = [0.0, 0.0]  # 中心点在原点

    # 当前2D方向角度
    current_angle = initial_angle

    for i in range(1, n_points):
        # 计算这一段的3D长度
        segment_3d = path_3d[i] - path_3d[i-1]
        segment_length = np.linalg.norm(segment_3d)

        if segment_length < 1e-10:
            path_2d[i] = path_2d[i-1].copy()
            continue

        # 计算转向角度
        if i >= 2:
            turn_angle = _compute_geodesic_turn_angle(
                path_3d[i-2], path_3d[i-1], path_3d[i], mesh
            )
            current_angle += turn_angle

        # 计算新的2D位置
        dx = segment_length * np.cos(current_angle)
        dy = segment_length * np.sin(current_angle)
        path_2d[i] = path_2d[i-1] + np.array([dx, dy])

    return path_2d


def _compute_geodesic_turn_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    mesh: trimesh.Trimesh
) -> float:
    """
    计算路径在p1点的测地转向角度

    测地转向角度是在曲面切平面上测量的转向角度。

    Args:
        p0, p1, p2: 三个连续的路径点
        mesh: 3D网格

    Returns:
        turn_angle: 转向角度（弧度，正值为左转，负值为右转）
    """
    # 前一段和当前段的方向
    prev_segment = p1 - p0
    curr_segment = p2 - p1

    prev_length = np.linalg.norm(prev_segment)
    curr_length = np.linalg.norm(curr_segment)

    if prev_length < 1e-10 or curr_length < 1e-10:
        return 0.0

    prev_dir = prev_segment / prev_length
    curr_dir = curr_segment / curr_length

    # 获取p1点的表面法向量
    try:
        _, _, face_idx = mesh.nearest.on_surface([p1])
        face = mesh.faces[int(face_idx[0])]
        v0, v1, v2 = mesh.vertices[face]
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
    except:
        # 回退：使用三点平面的法向量
        normal = np.cross(prev_segment, curr_segment)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-10:
            normal = normal / normal_len
        else:
            return 0.0

    # 将方向向量投影到切平面
    prev_proj = prev_dir - np.dot(prev_dir, normal) * normal
    curr_proj = curr_dir - np.dot(curr_dir, normal) * normal

    prev_proj_len = np.linalg.norm(prev_proj)
    curr_proj_len = np.linalg.norm(curr_proj)

    if prev_proj_len < 1e-10 or curr_proj_len < 1e-10:
        return 0.0

    prev_proj = prev_proj / prev_proj_len
    curr_proj = curr_proj / curr_proj_len

    # 计算转向角度
    cos_angle = np.clip(np.dot(prev_proj, curr_proj), -1, 1)

    # 使用叉积确定转向方向
    cross = np.cross(prev_proj, curr_proj)
    sin_angle = np.dot(cross, normal)

    turn_angle = np.arctan2(sin_angle, cos_angle)

    return turn_angle


def _generate_radial_uv_for_animation(
    mesh: trimesh.Trimesh,
    center_position: np.ndarray,
    center_normal: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray
) -> np.ndarray:
    """
    为动画生成径向UV坐标

    使用从中心点出发的测地线距离和角度来生成UV坐标。
    这种方法产生的UV坐标适合动画显示，但不保证精确的长度比例。

    Args:
        mesh: 3D网格
        center_position: 中心点位置
        center_normal: 中心点法向量
        x_axis, y_axis: 切平面坐标轴

    Returns:
        uv_coords: UV坐标数组 (N, 2)
    """
    vertices = mesh.vertices
    n_vertices = len(vertices)

    # 找到最近中心的顶点
    center_distances = np.linalg.norm(vertices - center_position, axis=1)
    center_vertex = np.argmin(center_distances)

    # 计算测地线距离
    geodesic_dists = compute_geodesic_distances(mesh, center_vertex)

    # 对于每个顶点，计算其相对于中心的角度
    uv_coords = np.zeros((n_vertices, 2))

    for i, v in enumerate(vertices):
        # 距离使用测地线距离
        dist = geodesic_dists[i]

        # 角度使用3D方向投影到切平面
        direction = v - center_position
        proj = direction - np.dot(direction, center_normal) * center_normal
        proj_len = np.linalg.norm(proj)

        if proj_len > 1e-10:
            proj = proj / proj_len
            x = np.dot(proj, x_axis)
            y = np.dot(proj, y_axis)
            angle = np.arctan2(y, x)
        else:
            angle = 0.0

        # UV坐标：极坐标转直角坐标
        uv_coords[i, 0] = dist * np.cos(angle)
        uv_coords[i, 1] = dist * np.sin(angle)

    return uv_coords


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

    改进：使用更精确的重心坐标计算，避免过度clipping导致的精度损失
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

        # 改进的重心坐标处理：
        # 只对明显超出范围的值进行微小调整，保持精度
        u, v, w = bary

        # 使用更宽松的容差，允许小的数值误差
        eps = 1e-6

        # 检查是否在三角形内或非常接近边界
        if u >= -eps and v >= -eps and w >= -eps:
            # 点在三角形内或边界附近，进行最小化校正
            # 只校正负值到0，不进行全局归一化（保持精度）
            u = max(0, u)
            v = max(0, v)
            w = max(0, w)

            # 只在总和明显偏离1时才归一化
            total = u + v + w
            if abs(total - 1.0) > eps:
                u, v, w = u/total, v/total, w/total
        else:
            # 点明显在三角形外，使用最近顶点
            # 找到最近的顶点
            dists = [np.linalg.norm(closest_point - vi) for vi in [v0, v1, v2]]
            min_idx = np.argmin(dists)
            if min_idx == 0:
                u, v, w = 1.0, 0.0, 0.0
            elif min_idx == 1:
                u, v, w = 0.0, 1.0, 0.0
            else:
                u, v, w = 0.0, 0.0, 1.0

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
    1. 对于有边界的曲面：使用LSCM展开
    2. 对于闭合曲面（无边界）：使用直接路径展开算法，精确保持路径长度

    Args:
        paths_3d: 路径字典 {point_id: 3D路径点数组}
        ir_points: IR点信息 {point_id: (position, name, is_center)}
        center_position: 中心点3D坐标
        mesh: 网格模型

    Returns:
        PathFlattenResult
    """
    print("\n" + "="*60)
    print("开始路径展开分析")
    print("="*60)

    print(f"\n网格信息:")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.faces)}")
    print(f"  表面积: {mesh.area:.2f}")

    # 检测是否为闭合曲面
    boundary_vertices = find_boundary_vertices(mesh)
    print(f"  边界顶点数: {len(boundary_vertices)}")

    # 对于闭合曲面（边界顶点 < 2），使用直接路径展开算法
    # 这种方法精确保持路径长度，避免LSCM在闭合曲面上的变形问题
    if len(boundary_vertices) < 2:
        print(f"\n  检测到闭合曲面，使用直接路径展开算法...")
        print(f"  （闭合曲面无法完美展开，但可以精确保持路径长度）")
        return flatten_paths_direct(paths_3d, ir_points, center_position, mesh)

    print(f"\n  检测到开放曲面，使用LSCM保形映射...")

    # 计算网格的3D尺寸作为参考
    bounds_3d = mesh.bounds
    size_3d = np.linalg.norm(bounds_3d[1] - bounds_3d[0])
    print(f"  3D模型对角线尺寸: {size_3d:.2f}")

    # 计算LSCM UV坐标
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
        std_ratio = np.std(length_ratios) if len(length_ratios) > 1 else 0
        print(f"\n  平均长度比例: {avg_ratio:.4f} (标准差: {std_ratio:.4f})")

        # 使用更严格的阈值：如果平均比例偏离1.0超过5%，就进行校正
        # 这比之前的50%-200%阈值严格得多，可以捕获更小的尺寸偏差
        if abs(avg_ratio - 1.0) > 0.05:  # 差异 > 5% 就校正
            correction = 1.0 / avg_ratio
            print(f"  警告: 2D/3D长度比例偏离1.0，应用校正因子 {correction:.4f}")

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

    # ============ 分两阶段展开 ============
    custom_paths = {}  # 保存自定义路径

    # 内部函数：展开单条路径
    def unfold_single_path(path_3d, start_angle):
        path_from_center = path_3d[::-1]
        path_2d_list = [center_2d.copy()]
        current_angle = start_angle

        for i in range(1, len(path_from_center)):
            segment_length = np.linalg.norm(path_from_center[i] - path_from_center[i-1])

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

                    prev_proj = prev_dir - np.dot(prev_dir, normal) * normal
                    curr_proj = curr_dir - np.dot(curr_dir, normal) * normal
                    cross = np.cross(prev_proj, curr_proj)
                    sign = np.sign(np.dot(cross, normal))
                    if sign == 0:
                        sign = 1

                    current_angle += sign * angle_change

            new_point = path_2d_list[-1] + segment_length * np.array([
                np.cos(current_angle),
                np.sin(current_angle)
            ])
            path_2d_list.append(new_point)

        return np.array(path_2d_list[::-1])

    # 对串联段路径进行排序，确保按顺序处理
    sorted_path_items = []
    serial_seg_items = []
    for point_id, path_3d in paths_3d.items():
        if "_to_" in point_id:
            custom_paths[point_id] = path_3d
        elif point_id.startswith("serial_seg_"):
            serial_seg_items.append((point_id, path_3d))
        else:
            sorted_path_items.append((point_id, path_3d))

    # 串联段按编号排序
    serial_seg_items.sort(key=lambda x: int(x[0].replace("serial_seg_", "")))
    sorted_path_items.extend(serial_seg_items)

    # 记录上一段的终点2D位置，用于连接后续段
    last_segment_end_2d = None
    last_segment_end_3d = None

    # 阶段1：展开主要路径
    for point_id, path_3d in sorted_path_items:
        if len(path_3d) < 2:
            continue

        start_angle = path_angles.get(point_id, 0.0)
        path_2d = unfold_single_path(path_3d, start_angle)

        # 对于串联段，检查是否需要调整位置以与前一段连接
        if point_id.startswith("serial_seg_") and last_segment_end_2d is not None:
            current_start_3d = path_3d[0]
            dist_to_last_end = np.linalg.norm(current_start_3d - last_segment_end_3d)
            if dist_to_last_end < 5.0:  # 5mm内认为是共享点
                offset = last_segment_end_2d - path_2d[0]
                path_2d = path_2d + offset

        paths_2d[point_id] = path_2d
        all_points.extend(path_2d.tolist())

        # 记录当前段的终点
        last_segment_end_2d = path_2d[-1].copy()
        last_segment_end_3d = path_3d[-1].copy()

        # IR点2D位置
        if point_id == "serial" or point_id.startswith("serial_seg_"):
            for ir_id, (ir_pos_3d, ir_name, ir_is_center) in ir_points.items():
                if ir_is_center:
                    continue
                if ir_id in ir_points_2d:
                    continue  # 已经处理过的点跳过
                distances = np.linalg.norm(path_3d - ir_pos_3d, axis=1)
                min_distance = np.min(distances)
                # 只有当点足够接近路径时才使用这个路径的2D坐标
                if min_distance < 5.0:  # 5mm阈值
                    nearest_idx = np.argmin(distances)
                    ir_points_2d[ir_id] = (path_2d[nearest_idx].copy(), ir_name, ir_is_center)
        else:
            ir_info = ir_points.get(point_id)
            if ir_info and len(path_2d) > 0:
                ir_points_2d[point_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])

    # 添加中心点
    for point_id, (pos, name, is_center) in ir_points.items():
        if is_center:
            ir_points_2d[point_id] = (center_2d.copy(), name, True)

    # 阶段2：展开自定义路径
    for point_id, path_3d in custom_paths.items():
        if len(path_3d) < 2:
            continue

        parts = point_id.split("_to_")
        if len(parts) != 2:
            continue
        from_id, to_id = parts

        start_angle = path_angles.get(point_id, 0.0)
        path_2d = unfold_single_path(path_3d, start_angle)

        # 检查端点是否已有2D坐标
        from_has_2d = from_id in ir_points_2d
        to_has_2d = to_id in ir_points_2d

        if from_has_2d and to_has_2d:
            from_2d_target = ir_points_2d[from_id][0]
            to_2d_target = ir_points_2d[to_id][0]
            path_2d = _adjust_path_endpoints(path_2d, from_2d_target, to_2d_target)
        elif from_has_2d:
            from_2d_target = ir_points_2d[from_id][0]
            offset = from_2d_target - path_2d[0]
            path_2d = path_2d + offset
            if to_id in ir_points and to_id not in ir_points_2d:
                ir_info = ir_points[to_id]
                ir_points_2d[to_id] = (path_2d[-1].copy(), ir_info[1], ir_info[2])
        elif to_has_2d:
            to_2d_target = ir_points_2d[to_id][0]
            offset = to_2d_target - path_2d[-1]
            path_2d = path_2d + offset
            if from_id in ir_points and from_id not in ir_points_2d:
                ir_info = ir_points[from_id]
                ir_points_2d[from_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])
        else:
            if from_id in ir_points:
                ir_info = ir_points[from_id]
                ir_points_2d[from_id] = (path_2d[0].copy(), ir_info[1], ir_info[2])
            if to_id in ir_points:
                ir_info = ir_points[to_id]
                ir_points_2d[to_id] = (path_2d[-1].copy(), ir_info[1], ir_info[2])

        paths_2d[point_id] = path_2d
        all_points.extend(path_2d.tolist())

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

    # 检查是否为串联模式（有 "serial" 路径或 "serial_seg_*" 路径）
    serial_path_keys = [k for k in flatten_result.paths_2d.keys()
                        if k == "serial" or k.startswith("serial_seg_")]
    is_serial_mode = len(serial_path_keys) > 0
    # 获取串联路径用于方向计算（优先使用完整的serial路径，否则合并所有段）
    if "serial" in flatten_result.paths_2d:
        serial_path_2d = flatten_result.paths_2d.get("serial")
    elif serial_path_keys:
        # 合并所有串联段路径用于方向计算
        all_serial_points = []
        for key in sorted(serial_path_keys):
            all_serial_points.extend(flatten_result.paths_2d[key].tolist())
        serial_path_2d = np.array(all_serial_points) if all_serial_points else None
    else:
        serial_path_2d = None

    # 辅助函数：查找与某个IR点相连的路径
    def find_path_for_point(point_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        查找与某个IR点相连的路径

        Returns:
            (path_2d, point_index) path_2d是2D路径，point_index是该点在路径中的索引（0=起点，-1=终点）
        """
        # 1. 星形模式：key就是点ID
        if point_id in flatten_result.paths_2d:
            return (flatten_result.paths_2d[point_id], 0)

        # 2. 检查自定义路径：key格式为 from_id_to_to_id
        for path_key, path_2d in flatten_result.paths_2d.items():
            if "_to_" in path_key:
                parts = path_key.split("_to_")
                if len(parts) == 2:
                    from_id, to_id = parts
                    if from_id == point_id:
                        return (path_2d, 0)  # 该点是路径起点
                    elif to_id == point_id:
                        return (path_2d, -1)  # 该点是路径终点

        # 3. 串联模式段路径：查找包含该点的段
        if is_serial_mode and point_id in flatten_result.ir_points_2d:
            point_pos = flatten_result.ir_points_2d[point_id][0]
            for path_key, path_2d in flatten_result.paths_2d.items():
                if path_key == "serial" or path_key.startswith("serial_seg_"):
                    if len(path_2d) >= 2:
                        # 查找路径上最近的点
                        distances = np.linalg.norm(path_2d - point_pos, axis=1)
                        min_dist = np.min(distances)
                        if min_dist < 1.0:  # 1mm阈值
                            nearest_idx = np.argmin(distances)
                            # 返回路径和一个特殊索引表示使用nearest计算
                            return (path_2d, nearest_idx)

        return None

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

            # 计算路径方向
            path_info = find_path_for_point(point_id)
            if path_info is not None:
                path_2d, point_index = path_info
                if len(path_2d) >= 2:
                    if point_index == 0:
                        # 该点是路径起点，方向从第一个点到第二个点
                        direction = path_2d[1] - path_2d[0]
                    elif point_index == -1 or point_index == len(path_2d) - 1:
                        # 该点是路径终点，方向从倒数第二个点到最后一个点
                        direction = path_2d[-1] - path_2d[-2]
                    else:
                        # 串联模式：点在路径中间，使用相邻点计算方向
                        if point_index > 0:
                            direction = path_2d[point_index] - path_2d[point_index - 1]
                        elif point_index < len(path_2d) - 1:
                            direction = path_2d[point_index + 1] - path_2d[point_index]
                        else:
                            direction = np.array([1.0, 0.0])
                else:
                    direction = np.array([1.0, 0.0])
            elif is_serial_mode and serial_path_2d is not None and len(serial_path_2d) >= 2:
                # 串联模式：找到路径上最近的点，计算方向
                distances = np.linalg.norm(serial_path_2d - pos, axis=1)
                nearest_idx = np.argmin(distances)
                if nearest_idx > 0:
                    direction = serial_path_2d[nearest_idx] - serial_path_2d[nearest_idx - 1]
                elif nearest_idx < len(serial_path_2d) - 1:
                    direction = serial_path_2d[nearest_idx + 1] - serial_path_2d[nearest_idx]
                else:
                    direction = np.array([1.0, 0.0])
            else:
                # 默认方向：指向中心点，或默认向右
                direction = flatten_result.center_2d - pos
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0])

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

    # 生成中心焊盘（仅在星形模式下，即有中心点标记时）
    center_pad = None
    has_center_point = any(is_center for _, (_, _, is_center) in flatten_result.ir_points_2d.items())

    if has_center_point:
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
    all_outlines = list(groove_outlines.values()) + list(ir_pads.values())
    if center_pad is not None:
        all_outlines.append(center_pad)
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


class UVTo3DMapper:
    """UV坐标到3D坐标的映射器"""

    def __init__(self, mesh: trimesh.Trimesh, uv_coords: np.ndarray):
        """
        初始化映射器

        Args:
            mesh: 3D网格模型
            uv_coords: 与网格顶点对应的UV坐标 (N, 2)
        """
        self.mesh = mesh
        self.uv_coords = uv_coords
        self.triangulation = None
        self.finder = None
        self.use_fallback = False

        # 尝试创建Delaunay三角剖分
        try:
            self.triangulation = Triangulation(
                uv_coords[:, 0], uv_coords[:, 1],
                mesh.faces.astype(np.int32)
            )
            self.finder = self.triangulation.get_trifinder()
        except Exception as e:
            print(f"  警告: 三角化失败 ({e})，使用备选映射方法")
            self.use_fallback = True
            # 预计算每个UV三角形的边界框，用于快速查找
            self._precompute_triangle_bounds()

    def _precompute_triangle_bounds(self):
        """预计算每个三角形在UV空间的边界框"""
        faces = self.mesh.faces
        n_faces = len(faces)
        self.uv_tri_min = np.zeros((n_faces, 2))
        self.uv_tri_max = np.zeros((n_faces, 2))
        self.uv_tri_centers = np.zeros((n_faces, 2))

        for i, face in enumerate(faces):
            uv_verts = self.uv_coords[face]
            self.uv_tri_min[i] = uv_verts.min(axis=0)
            self.uv_tri_max[i] = uv_verts.max(axis=0)
            self.uv_tri_centers[i] = uv_verts.mean(axis=0)

    def _find_containing_triangle_fallback(self, point_2d: np.ndarray) -> int:
        """使用备选方法查找包含点的三角形"""
        # 首先找到距离最近的几个三角形中心
        distances = np.linalg.norm(self.uv_tri_centers - point_2d, axis=1)
        candidates = np.argsort(distances)[:20]  # 取最近的20个

        # 检查点是否在这些三角形内
        for tri_idx in candidates:
            face = self.mesh.faces[tri_idx]
            uv_verts = self.uv_coords[face]

            if self._point_in_triangle_2d(point_2d, uv_verts[0], uv_verts[1], uv_verts[2]):
                return tri_idx

        # 如果没找到包含的三角形，返回最近的
        return candidates[0]

    def _point_in_triangle_2d(self, p, v0, v1, v2):
        """检查2D点是否在三角形内"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def map_2d_to_3d(self, points_2d: np.ndarray) -> np.ndarray:
        """
        将2D UV点映射回3D空间

        Args:
            points_2d: 2D点坐标数组 (N, 2)

        Returns:
            points_3d: 映射后的3D坐标数组 (N, 3)
        """
        if len(points_2d) == 0:
            return np.array([])

        points_3d = np.zeros((len(points_2d), 3))

        if self.use_fallback:
            # 使用备选方法
            for i, point_2d in enumerate(points_2d):
                tri_idx = self._find_containing_triangle_fallback(point_2d)
                face = self.mesh.faces[tri_idx]

                uv0, uv1, uv2 = self.uv_coords[face]
                v0, v1, v2 = self.mesh.vertices[face]

                # 计算重心坐标
                bary = self._compute_barycentric_2d(point_2d, uv0, uv1, uv2)

                # 映射到3D
                points_3d[i] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2

            return points_3d

        # 使用三角化方法
        tri_indices = self.finder(points_2d[:, 0], points_2d[:, 1])

        valid_mask = tri_indices != -1

        if not np.any(valid_mask):
            # 所有点都在三角化外部，使用备选方法
            for i, point_2d in enumerate(points_2d):
                tri_idx = self._find_containing_triangle_fallback(point_2d)
                face = self.mesh.faces[tri_idx]
                uv0, uv1, uv2 = self.uv_coords[face]
                v0, v1, v2 = self.mesh.vertices[face]
                bary = self._compute_barycentric_2d(point_2d, uv0, uv1, uv2)
                points_3d[i] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
            return points_3d

        valid_indices = np.where(valid_mask)[0]
        valid_tri_indices = tri_indices[valid_indices]

        face_vertices_indices = self.triangulation.triangles[valid_tri_indices]

        uv0 = self.uv_coords[face_vertices_indices[:, 0]]
        uv1 = self.uv_coords[face_vertices_indices[:, 1]]
        uv2 = self.uv_coords[face_vertices_indices[:, 2]]

        v0 = self.mesh.vertices[face_vertices_indices[:, 0]]
        v1 = self.mesh.vertices[face_vertices_indices[:, 1]]
        v2 = self.mesh.vertices[face_vertices_indices[:, 2]]

        v0v1 = uv1 - uv0
        v0v2 = uv2 - uv0
        v0p = points_2d[valid_indices] - uv0

        d00 = np.sum(v0v1 * v0v1, axis=1)
        d01 = np.sum(v0v1 * v0v2, axis=1)
        d11 = np.sum(v0v2 * v0v2, axis=1)
        d20 = np.sum(v0p * v0v1, axis=1)
        d21 = np.sum(v0p * v0v2, axis=1)

        denom = d00 * d11 - d01 * d01
        denom[np.abs(denom) < 1e-10] = 1.0

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        points_3d[valid_indices] = (
            v0 * u[:, np.newaxis] +
            v1 * v[:, np.newaxis] +
            v2 * w[:, np.newaxis]
        )

        # 处理无效点（三角化外的点）
        invalid_indices = np.where(~valid_mask)[0]
        for i in invalid_indices:
            tri_idx = self._find_containing_triangle_fallback(points_2d[i])
            face = self.mesh.faces[tri_idx]
            uv0, uv1, uv2 = self.uv_coords[face]
            v0, v1, v2 = self.mesh.vertices[face]
            bary = self._compute_barycentric_2d(points_2d[i], uv0, uv1, uv2)
            points_3d[i] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2

        return points_3d

    def _compute_barycentric_2d(self, p, a, b, c):
        """计算2D点在三角形内的重心坐标"""
        v0 = c - a
        v1 = b - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return (1.0, 0.0, 0.0)

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # 限制在合理范围内
        u = max(0, min(1, u))
        v = max(0, min(1, v))
        w = 1.0 - u - v
        w = max(0, w)

        # 归一化
        total = u + v + w
        if total > 0:
            return (w / total, v / total, u / total)
        return (1.0, 0.0, 0.0)


def triangulate_fpc_layout(
    fpc_result: FPCLayoutResult,
    mapper: Optional[UVTo3DMapper] = None
) -> Tuple[trimesh.Trimesh, Optional[trimesh.Trimesh]]:
    """
    将FPC布局轮廓转换为三角网格，并可选地映射回3D
    
    Args:
        fpc_result: FPC布局结果
        mapper: UV到3D映射器 (可选)
        
    Returns:
        (mesh_2d, mesh_3d): 2D网格和3D网格（如果mapper不为None）
    """
    vertices = []
    faces = []
    current_idx = 0
    
    # 收集所有轮廓（不合并，直接三角化每个轮廓）
    # 这样可以保持独立的颜色或结构，但为了简单，我们生成一个单一网格
    # 注意：trimesh.creation.triangulate_polygon 需要 shapely
    
    import shapely.geometry
    import trimesh.creation
    
    all_polygons = []
    
    # 添加路径凹槽
    for outline in fpc_result.groove_outlines.values():
        if len(outline) >= 3:
            all_polygons.append(shapely.geometry.Polygon(outline))
            
    # 添加焊盘
    for pad in fpc_result.ir_pads.values():
        if len(pad) >= 3:
            all_polygons.append(shapely.geometry.Polygon(pad))
            
    # 添加中心焊盘
    if fpc_result.center_pad is not None and len(fpc_result.center_pad) >= 3:
        all_polygons.append(shapely.geometry.Polygon(fpc_result.center_pad))
        
    if not all_polygons:
        return trimesh.Trimesh(), None
        
    # 合并多边形（处理重叠）
    from shapely.ops import unary_union
    merged = unary_union(all_polygons)
    
    # 处理MultiPolygon
    if isinstance(merged, shapely.geometry.Polygon):
        polys = [merged]
    else:
        polys = list(merged.geoms)
        
    # 三角化
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for poly in polys:
        # 使用trimesh的三角化功能
        try:
            # trimesh.creation.triangulate_polygon 返回 (vertices, faces)
            v, f = trimesh.creation.triangulate_polygon(poly)
            
            # 转换为3D点 (z=0)
            v_3d = np.column_stack([v, np.zeros(len(v))])
            
            all_vertices.append(v_3d)
            all_faces.append(f + vertex_offset)
            vertex_offset += len(v)
        except Exception as e:
            print(f"三角化失败: {e}")
            continue
            
    if not all_vertices:
        return trimesh.Trimesh(), None

    # 合并所有顶点和面
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    # 创建2D网格（禁用顶点合并以保持与3D网格一致）
    mesh_2d = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False  # 禁用自动处理（如顶点合并）
    )

    mesh_3d = None
    if mapper and len(mesh_2d.vertices) > 0:
        # 映射2D顶点到3D
        # 注意：mesh_2d.vertices 是 (x, y, 0)，我们需要 (x, y)
        points_2d = mesh_2d.vertices[:, :2]
        points_3d = mapper.map_2d_to_3d(points_2d)

        # 创建3D网格（同样禁用自动处理）
        mesh_3d = trimesh.Trimesh(
            vertices=points_3d,
            faces=mesh_2d.faces.copy(),
            process=False  # 禁用自动处理
        )

        # 手动计算顶点法向量（不调用fix_normals避免修改顶点）
        try:
            # 计算面法向量
            v0 = mesh_3d.vertices[mesh_3d.faces[:, 0]]
            v1 = mesh_3d.vertices[mesh_3d.faces[:, 1]]
            v2 = mesh_3d.vertices[mesh_3d.faces[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            face_normals_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = face_normals / (face_normals_len + 1e-10)

            # 计算顶点法向量（面法向量的平均）
            vertex_normals = np.zeros_like(mesh_3d.vertices)
            for i, face in enumerate(mesh_3d.faces):
                for v_idx in face:
                    vertex_normals[v_idx] += face_normals[i]
            vertex_normals_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            vertex_normals = vertex_normals / (vertex_normals_len + 1e-10)

            # 沿法向偏移 0.1mm
            mesh_3d.vertices = mesh_3d.vertices + vertex_normals * 0.1
        except Exception as e:
            print(f"  计算法向量时出错: {e}")

    return mesh_2d, mesh_3d


@dataclass
class FlattenValidationResult:
    """展平精度验证结果"""
    path_length_errors: Dict[str, float]  # point_id -> 长度误差百分比
    avg_length_error: float  # 平均长度误差
    max_length_error: float  # 最大长度误差
    point_distance_errors: Dict[str, float]  # point_id -> 点位置误差（mm）
    avg_distance_error: float  # 平均位置误差
    max_distance_error: float  # 最大位置误差
    roundtrip_errors: Dict[str, np.ndarray]  # point_id -> 每点的往返误差数组
    is_valid: bool  # 是否通过验证
    validation_message: str  # 验证结果描述


def validate_flatten_accuracy(
    mesh: trimesh.Trimesh,
    uv_coords: np.ndarray,
    paths_3d: Dict[str, np.ndarray],
    paths_2d: Dict[str, np.ndarray],
    ir_points: Dict[str, Tuple[np.ndarray, str, bool]],
    ir_points_2d: Dict[str, Tuple[np.ndarray, str, bool]],
    tolerance_percent: float = 5.0,
    tolerance_mm: float = 1.0
) -> FlattenValidationResult:
    """
    验证展平精度 - 检查3D到2D的映射是否保持几何关系

    验证内容：
    1. 路径长度保持：2D路径长度应该接近3D路径长度
    2. 点位置精度：IR点在2D中的相对位置应该正确
    3. 往返映射误差：3D -> 2D -> 3D 应该回到原位置

    Args:
        mesh: 3D网格模型
        uv_coords: UV坐标
        paths_3d: 3D路径字典
        paths_2d: 2D路径字典
        ir_points: 3D IR点字典
        ir_points_2d: 2D IR点字典
        tolerance_percent: 长度误差容差（百分比）
        tolerance_mm: 位置误差容差（毫米）

    Returns:
        FlattenValidationResult
    """
    print("\n" + "="*60)
    print("展平精度验证")
    print("="*60)

    path_length_errors = {}
    point_distance_errors = {}
    roundtrip_errors = {}

    # 创建2D到3D映射器
    mapper = UVTo3DMapper(mesh, uv_coords)

    # 1. 验证路径长度
    print("\n1. 路径长度验证:")
    for point_id in paths_3d.keys():
        if point_id not in paths_2d:
            continue

        path_3d = paths_3d[point_id]
        path_2d = paths_2d[point_id]

        if len(path_3d) < 2 or len(path_2d) < 2:
            continue

        # 计算3D路径长度
        length_3d = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))

        # 计算2D路径长度
        length_2d = np.sum(np.linalg.norm(np.diff(path_2d, axis=0), axis=1))

        if length_3d > 1e-6:
            error_percent = abs(length_2d - length_3d) / length_3d * 100
            path_length_errors[point_id] = error_percent
            print(f"  路径 {point_id[:8]}: 3D={length_3d:.2f}mm, 2D={length_2d:.2f}mm, 误差={error_percent:.2f}%")

    # 2. 验证IR点位置（往返映射）
    print("\n2. IR点往返映射验证:")
    for point_id, (pos_3d, name, is_center) in ir_points.items():
        if point_id not in ir_points_2d:
            continue

        pos_2d, _, _ = ir_points_2d[point_id]

        # 将2D点映射回3D
        pos_3d_roundtrip = mapper.map_2d_to_3d(pos_2d.reshape(1, 2))[0]

        # 计算往返误差
        distance_error = np.linalg.norm(pos_3d - pos_3d_roundtrip)
        point_distance_errors[point_id] = distance_error
        print(f"  点 {name}: 往返误差 = {distance_error:.4f}mm")

    # 3. 验证路径点的往返映射
    print("\n3. 路径点往返映射验证:")
    for point_id in paths_3d.keys():
        if point_id not in paths_2d:
            continue

        path_3d = paths_3d[point_id]
        path_2d = paths_2d[point_id]

        if len(path_3d) != len(path_2d):
            print(f"  警告: 路径 {point_id[:8]} 点数不匹配 (3D={len(path_3d)}, 2D={len(path_2d)})")
            continue

        # 将2D路径映射回3D
        path_3d_roundtrip = mapper.map_2d_to_3d(path_2d)

        # 计算每点的往返误差
        errors = np.linalg.norm(path_3d - path_3d_roundtrip, axis=1)
        roundtrip_errors[point_id] = errors

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"  路径 {point_id[:8]}: 平均误差={avg_error:.4f}mm, 最大误差={max_error:.4f}mm")

    # 计算总体统计
    if path_length_errors:
        avg_length_error = np.mean(list(path_length_errors.values()))
        max_length_error = np.max(list(path_length_errors.values()))
    else:
        avg_length_error = 0.0
        max_length_error = 0.0

    if point_distance_errors:
        avg_distance_error = np.mean(list(point_distance_errors.values()))
        max_distance_error = np.max(list(point_distance_errors.values()))
    else:
        avg_distance_error = 0.0
        max_distance_error = 0.0

    # 判断是否通过验证
    is_valid = (max_length_error <= tolerance_percent and
                max_distance_error <= tolerance_mm)

    if is_valid:
        validation_message = f"✓ 验证通过: 长度误差={max_length_error:.2f}%, 位置误差={max_distance_error:.4f}mm"
    else:
        issues = []
        if max_length_error > tolerance_percent:
            issues.append(f"长度误差过大: {max_length_error:.2f}% > {tolerance_percent}%")
        if max_distance_error > tolerance_mm:
            issues.append(f"位置误差过大: {max_distance_error:.4f}mm > {tolerance_mm}mm")
        validation_message = "✗ 验证失败: " + "; ".join(issues)

    print("\n" + "="*60)
    print("验证总结:")
    print(f"  平均长度误差: {avg_length_error:.2f}%")
    print(f"  最大长度误差: {max_length_error:.2f}%")
    print(f"  平均位置误差: {avg_distance_error:.4f}mm")
    print(f"  最大位置误差: {max_distance_error:.4f}mm")
    print(f"  {validation_message}")
    print("="*60 + "\n")

    return FlattenValidationResult(
        path_length_errors=path_length_errors,
        avg_length_error=avg_length_error,
        max_length_error=max_length_error,
        point_distance_errors=point_distance_errors,
        avg_distance_error=avg_distance_error,
        max_distance_error=max_distance_error,
        roundtrip_errors=roundtrip_errors,
        is_valid=is_valid,
        validation_message=validation_message
    )


def generate_fpc_fitting_animation_data(
    mesh: trimesh.Trimesh,
    uv_coords: np.ndarray,
    fpc_layout: FPCLayoutResult,
    n_frames: int = 60
) -> List[Tuple[trimesh.Trimesh, float]]:
    """
    生成FPC贴合3D模型的动画数据

    动画过程：
    1. FPC从平面状态开始（z=0的2D形态）
    2. 逐渐弯曲贴合到3D曲面上
    3. 完全贴合后展示最终状态

    Args:
        mesh: 3D网格模型
        uv_coords: UV坐标
        fpc_layout: FPC布局结果
        n_frames: 动画帧数

    Returns:
        List of (mesh, progress) tuples, where progress is 0.0 to 1.0
    """
    print("\n生成FPC贴合动画数据...")

    # 首先生成2D和3D的FPC网格
    mapper = UVTo3DMapper(mesh, uv_coords)
    mesh_2d, mesh_3d = triangulate_fpc_layout(fpc_layout, mapper)

    if mesh_2d is None or len(mesh_2d.vertices) == 0:
        print("  警告: 无法生成FPC网格")
        return []

    if mesh_3d is None or len(mesh_3d.vertices) == 0:
        print("  警告: 无法生成3D FPC网格")
        return []

    animation_frames = []

    # 获取2D和3D顶点
    vertices_2d = mesh_2d.vertices.copy()  # (x, y, 0)
    vertices_3d = mesh_3d.vertices.copy()  # 贴合曲面的3D坐标

    # 计算初始高度偏移（让2D FPC悬浮在模型上方）
    model_bounds = mesh.bounds
    model_center = (model_bounds[0] + model_bounds[1]) / 2
    model_size = np.linalg.norm(model_bounds[1] - model_bounds[0])

    # 将2D FPC居中并放置在模型上方
    fpc_center_2d = vertices_2d.mean(axis=0)
    vertices_2d_centered = vertices_2d - fpc_center_2d
    vertices_2d_centered[:, 0] += model_center[0]
    vertices_2d_centered[:, 1] += model_center[1]
    vertices_2d_centered[:, 2] = model_bounds[1][2] + model_size * 0.3  # 在模型上方

    print(f"  生成 {n_frames} 帧动画...")

    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0.0 到 1.0

        # 使用平滑的缓动函数（ease-in-out）
        t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)

        # 插值顶点位置
        vertices_interpolated = (1 - t_smooth) * vertices_2d_centered + t_smooth * vertices_3d

        # 创建当前帧的网格
        frame_mesh = trimesh.Trimesh(
            vertices=vertices_interpolated,
            faces=mesh_2d.faces.copy()
        )
        frame_mesh.fix_normals()

        animation_frames.append((frame_mesh, t))

    print(f"  动画数据生成完成: {len(animation_frames)} 帧")

    return animation_frames


class FPCFittingAnimator:
    """FPC贴合动画控制器"""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        fpc_layout: FPCLayoutResult
    ):
        """
        初始化动画控制器

        Args:
            mesh: 3D网格模型
            uv_coords: UV坐标
            fpc_layout: FPC布局结果
        """
        self.mesh = mesh
        self.uv_coords = uv_coords
        self.fpc_layout = fpc_layout

        self.mapper = UVTo3DMapper(mesh, uv_coords)
        self.mesh_2d = None
        self.mesh_3d = None
        self.vertices_2d_start = None
        self.vertices_3d_end = None

        self._prepare_meshes()

    def _prepare_meshes(self):
        """准备2D和3D网格"""
        self.mesh_2d, self.mesh_3d = triangulate_fpc_layout(
            self.fpc_layout, self.mapper
        )

        if self.mesh_2d is None or self.mesh_3d is None:
            print("  警告: 无法生成FPC网格")
            return

        # 验证顶点数一致
        if len(self.mesh_2d.vertices) != len(self.mesh_3d.vertices):
            print(f"  警告: 2D网格和3D网格顶点数不匹配 ({len(self.mesh_2d.vertices)} vs {len(self.mesh_3d.vertices)})")
            print(f"  尝试重新创建3D网格...")

            # 重新映射2D顶点到3D
            points_2d = self.mesh_2d.vertices[:, :2]
            points_3d = self.mapper.map_2d_to_3d(points_2d)

            self.mesh_3d = trimesh.Trimesh(
                vertices=points_3d,
                faces=self.mesh_2d.faces.copy(),
                process=False
            )

        # 获取顶点
        vertices_2d = self.mesh_2d.vertices.copy()
        vertices_3d = self.mesh_3d.vertices.copy()

        # 计算初始位置（2D FPC放在模型上方）
        model_bounds = self.mesh.bounds
        model_center = (model_bounds[0] + model_bounds[1]) / 2
        model_size = np.linalg.norm(model_bounds[1] - model_bounds[0])

        fpc_center = vertices_2d.mean(axis=0)
        self.vertices_2d_start = vertices_2d - fpc_center
        self.vertices_2d_start[:, 0] += model_center[0]
        self.vertices_2d_start[:, 1] += model_center[1]
        self.vertices_2d_start[:, 2] = model_bounds[1][2] + model_size * 0.3

        self.vertices_3d_end = vertices_3d

    def get_frame(self, progress: float) -> Optional[trimesh.Trimesh]:
        """
        获取指定进度的动画帧

        Args:
            progress: 动画进度 (0.0 到 1.0)

        Returns:
            当前帧的网格
        """
        if self.mesh_2d is None or self.vertices_2d_start is None:
            return None

        # 平滑缓动
        t = 0.5 - 0.5 * np.cos(progress * np.pi)

        # 插值
        vertices = (1 - t) * self.vertices_2d_start + t * self.vertices_3d_end

        frame_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.mesh_2d.faces.copy()
        )
        frame_mesh.fix_normals()

        return frame_mesh

    def is_ready(self) -> bool:
        """检查动画是否准备就绪"""
        return (self.mesh_2d is not None and
                self.vertices_2d_start is not None and
                self.vertices_3d_end is not None)
