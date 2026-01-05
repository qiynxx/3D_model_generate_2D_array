"""2D路径规划模块 - 在曲面上生成不交叉的弧线路径

核心思路：
1. 将所有IR点投影到中心点的切平面（建立局部2D坐标系）
2. 在2D切平面上规划径向弧线路径（从中心点到各IR点）
3. 将2D路径采样点映射回3D曲面

这种方法天然保证路径不交叉（因为是从一个中心点出发的射线/弧线）
"""
import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq
from scipy.interpolate import splprep, splev


def find_nearest_vertex(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    """找到离给定点最近的顶点索引"""
    distances = np.linalg.norm(mesh.vertices - point, axis=1)
    return int(np.argmin(distances))


def project_to_surface(mesh: trimesh.Trimesh, point: np.ndarray) -> np.ndarray:
    """
    将点投影到网格表面最近位置

    使用最近点查询，比简单的最近顶点更精确
    """
    try:
        # 尝试使用trimesh的closest_point方法（更精确）
        closest, distance, face_id = mesh.nearest.on_surface([point])
        return closest[0].copy()
    except:
        # 回退到最近顶点
        distances = np.linalg.norm(mesh.vertices - point, axis=1)
        nearest_idx = np.argmin(distances)
        return mesh.vertices[nearest_idx].copy()


def build_tangent_plane_basis(center_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建以center_normal为法向的切平面坐标系

    Returns:
        (x_axis, y_axis, normal) 三个正交单位向量
    """
    normal = center_normal / (np.linalg.norm(center_normal) + 1e-10)

    # 选择一个参考向量
    if abs(normal[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, normal)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

    return x_axis, y_axis, normal


def project_to_tangent_plane(
    point_3d: np.ndarray,
    center_3d: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray
) -> np.ndarray:
    """
    将3D点投影到切平面，返回2D坐标
    """
    offset = point_3d - center_3d
    x = np.dot(offset, x_axis)
    y = np.dot(offset, y_axis)
    return np.array([x, y])


def tangent_plane_to_3d(
    point_2d: np.ndarray,
    center_3d: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray
) -> np.ndarray:
    """
    将切平面上的2D点转换回3D空间
    """
    return center_3d + point_2d[0] * x_axis + point_2d[1] * y_axis


def create_arc_path_2d(
    start_2d: np.ndarray,
    end_2d: np.ndarray,
    num_points: int = 50,
    curvature: float = 0.0
) -> np.ndarray:
    """
    在2D平面上创建从start到end的弧线路径

    Args:
        start_2d: 起点（通常是中心点，即原点）
        end_2d: 终点
        num_points: 采样点数
        curvature: 曲率控制（0=直线，正值=左弯，负值=右弯）

    Returns:
        2D路径点数组 (num_points, 2)
    """
    if curvature == 0 or np.linalg.norm(end_2d - start_2d) < 1e-6:
        # 直线路径
        t = np.linspace(0, 1, num_points)
        path = np.outer(1 - t, start_2d) + np.outer(t, end_2d)
        return path

    # 弧线路径（使用二次贝塞尔曲线）
    # 控制点在中点垂线上偏移
    mid = (start_2d + end_2d) / 2
    direction = end_2d - start_2d
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)

    # 控制点
    control = mid + curvature * np.linalg.norm(direction) * perpendicular

    # 二次贝塞尔曲线
    t = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 2))
    for i, ti in enumerate(t):
        path[i] = (1-ti)**2 * start_2d + 2*(1-ti)*ti * control + ti**2 * end_2d

    return path


def map_2d_path_to_3d_surface_iterative(
    path_2d: np.ndarray,
    center_3d: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    mesh: trimesh.Trimesh,
    debug: bool = False
) -> np.ndarray:
    """
    将2D切平面上的路径映射回3D曲面 - 迭代步进方法

    使用迭代步进来避免高曲率区域的跳跃问题：
    1. 从起点开始
    2. 每一步沿着曲面朝目标方向移动一小步
    3. 投影到曲面上
    """
    if len(path_2d) < 2:
        return np.array([center_3d])

    path_3d = []
    current_3d = project_to_surface(mesh, center_3d)
    path_3d.append(current_3d.copy())

    max_jump = 0
    jump_count = 0

    for i in range(1, len(path_2d)):
        # 目标点在切平面上的位置
        target_2d = path_2d[i]

        # 将目标点转换到3D（仅用于方向参考）
        target_3d_ref = tangent_plane_to_3d(target_2d, center_3d, x_axis, y_axis)

        # 在曲面上找到最近的目标位置
        target_3d = project_to_surface(mesh, target_3d_ref)

        # 检查是否有大跳跃
        step_dist = np.linalg.norm(target_3d - current_3d)

        # 如果步长过大，使用细分步进
        expected_step = np.linalg.norm(path_2d[i] - path_2d[i-1])
        if step_dist > expected_step * 3 and expected_step > 0.01:
            # 跳跃过大，需要沿曲面步进
            jump_count += 1
            # 使用曲面上的方向逐步移动
            direction = target_3d - current_3d
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # 分成多个小步
            n_substeps = max(int(step_dist / expected_step), 2)
            for j in range(n_substeps):
                t = (j + 1) / n_substeps
                intermediate = current_3d + t * (target_3d - current_3d)
                intermediate = project_to_surface(mesh, intermediate)
                if j == n_substeps - 1:
                    path_3d.append(intermediate.copy())
                    current_3d = intermediate
        else:
            path_3d.append(target_3d.copy())
            current_3d = target_3d

        if i > 0:
            jump = np.linalg.norm(path_3d[-1] - path_3d[-2])
            if jump > max_jump:
                max_jump = jump

    if debug:
        print(f"    映射: {len(path_3d)} 点, 最大跳跃距离: {max_jump:.4f}, 跳跃修复次数: {jump_count}")

    return np.array(path_3d)


def compute_geodesic_dijkstra(
    mesh: trimesh.Trimesh,
    start_point: np.ndarray,
    end_point: np.ndarray,
    num_samples: int = 60,
    smooth_path: bool = False,
    turn_penalty: float = 2.0,
    normal_penalty: float = 5.0
) -> np.ndarray:
    """
    计算曲面上两点之间的最短路径 - 使用表面步进法

    Args:
        mesh: 网格模型
        start_point: 起点3D坐标
        end_point: 终点3D坐标
        num_samples: 输出采样点数

    Returns:
        曲面上的路径点数组
    """
    # 确保端点在曲面上
    start_on_surface = project_to_surface(mesh, start_point)
    end_on_surface = project_to_surface(mesh, end_point)

    # 获取起点的法向量
    _, _, start_face = mesh.nearest.on_surface([start_on_surface])
    start_normal = mesh.face_normals[int(start_face[0])]

    # 计算直线距离
    direct_dist = np.linalg.norm(end_on_surface - start_on_surface)
    if direct_dist < 1e-6:
        return np.array([start_on_surface, end_on_surface])

    # 使用小步增量沿曲面走
    # 步长设为直线距离的一小部分，确保不会跳到另一面
    step_size = direct_dist / (num_samples * 2)  # 使用更多步数来确保平滑

    points = [start_on_surface.copy()]
    current_pos = start_on_surface.copy()
    current_normal = start_normal.copy()

    max_iterations = num_samples * 4  # 防止无限循环
    iteration = 0

    while iteration < max_iterations:
        # 计算到终点的方向
        to_end = end_on_surface - current_pos
        dist_to_end = np.linalg.norm(to_end)

        if dist_to_end < step_size:
            # 已经很接近终点
            points.append(end_on_surface.copy())
            break

        direction = to_end / dist_to_end

        # 将方向投影到当前切平面（沿曲面走）
        direction_tangent = direction - np.dot(direction, current_normal) * current_normal
        tangent_norm = np.linalg.norm(direction_tangent)
        if tangent_norm > 1e-6:
            direction_tangent = direction_tangent / tangent_norm
        else:
            direction_tangent = direction

        # 沿切线方向移动一小步
        next_guess = current_pos + direction_tangent * step_size

        # 投影到曲面
        projected, dist, face_idx = mesh.nearest.on_surface([next_guess])
        proj_pos = projected[0]
        proj_normal = mesh.face_normals[int(face_idx[0])]

        # 检查是否跳到了另一面
        if np.dot(proj_normal, current_normal) < 0.0:
            # 法向量反转，说明可能跳到了另一面
            # 尝试使用更小的步长
            smaller_step = step_size * 0.5
            next_guess = current_pos + direction_tangent * smaller_step
            projected, dist, face_idx = mesh.nearest.on_surface([next_guess])
            proj_pos = projected[0]
            proj_normal = mesh.face_normals[int(face_idx[0])]

            if np.dot(proj_normal, current_normal) < 0.0:
                # 仍然跳跃，使用Dijkstra完成剩余路径
                remaining_path = _dijkstra_path_segment(mesh, current_pos, end_on_surface)
                if len(remaining_path) > 0:
                    points.extend(remaining_path[1:])  # 跳过第一个点（当前点）
                break

        points.append(proj_pos.copy())
        current_pos = proj_pos.copy()
        current_normal = proj_normal.copy()
        iteration += 1

    # 确保终点正确
    if np.linalg.norm(points[-1] - end_on_surface) > step_size:
        points.append(end_on_surface.copy())

    path = np.array(points)

    # 简化路径 - 移除共线的中间点
    simplified = _simplify_path(path)

    # 重采样到指定点数
    return resample_path(simplified, num_samples)


def _dijkstra_path_segment(
    mesh: trimesh.Trimesh,
    start_point: np.ndarray,
    end_point: np.ndarray
) -> np.ndarray:
    """使用Dijkstra获取路径段"""
    start_idx = find_nearest_vertex(mesh, start_point)
    end_idx = find_nearest_vertex(mesh, end_point)

    if start_idx == end_idx:
        return np.array([start_point, end_point])

    path_indices = dijkstra_path(mesh, start_idx, end_idx)
    if len(path_indices) == 0:
        return np.array([start_point, end_point])

    path_vertices = mesh.vertices[path_indices].copy()
    path_vertices[0] = project_to_surface(mesh, start_point)
    path_vertices[-1] = project_to_surface(mesh, end_point)

    return path_vertices


def _simplify_path(path: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """简化路径 - 使用Douglas-Peucker算法移除不必要的点"""
    if len(path) <= 2:
        return path

    # 计算每个点到首尾连线的距离
    start = path[0]
    end = path[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return np.array([start, end])

    line_unit = line_vec / line_len

    # 找到距离最远的点
    max_dist = 0
    max_idx = 0

    for i in range(1, len(path) - 1):
        # 点到直线的距离
        v = path[i] - start
        proj_len = np.dot(v, line_unit)
        proj_point = start + proj_len * line_unit
        dist = np.linalg.norm(path[i] - proj_point)

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # 如果最大距离小于容差，直接用首尾两点
    if max_dist < tolerance:
        return np.array([start, end])

    # 否则递归简化
    left = _simplify_path(path[:max_idx + 1], tolerance)
    right = _simplify_path(path[max_idx:], tolerance)

    return np.vstack([left[:-1], right])


def _walk_on_surface(
    mesh: trimesh.Trimesh,
    start_point: np.ndarray,
    end_point: np.ndarray,
    num_samples: int
) -> np.ndarray:
    """
    沿曲面行走 - 每步投影并检查法向量一致性
    """
    start_on_surface, _, start_face = mesh.nearest.on_surface([start_point])
    end_on_surface, _, _ = mesh.nearest.on_surface([end_point])
    start_pos = start_on_surface[0]
    end_pos = end_on_surface[0]
    start_normal = mesh.face_normals[int(start_face[0])]

    points = [start_pos.copy()]
    current_pos = start_pos.copy()
    current_normal = start_normal.copy()

    step_size = np.linalg.norm(end_pos - start_pos) / (num_samples - 1)

    for i in range(1, num_samples):
        direction = end_pos - current_pos
        dist = np.linalg.norm(direction)
        if dist < step_size * 0.5:
            points.append(end_pos.copy())
            break

        direction = direction / dist
        next_guess = current_pos + direction * min(step_size, dist)

        # 投影并检查法向量
        projected, _, face_idx = mesh.nearest.on_surface([next_guess])
        proj_normal = mesh.face_normals[int(face_idx[0])]

        # 如果法向量反转，说明跳到了另一面，需要沿边走
        if np.dot(proj_normal, current_normal) < 0.3:
            # 找到当前位置最近的顶点，沿边走
            nearest_v = find_nearest_vertex(mesh, current_pos)
            neighbors = _get_vertex_neighbors(mesh, nearest_v)

            # 选择朝向目标且法向量一致的邻居
            best_neighbor = None
            best_progress = -np.inf
            for nv in neighbors:
                nv_pos = mesh.vertices[nv]
                nv_normal = mesh.vertex_normals[nv]
                if np.dot(nv_normal, current_normal) > 0.3:
                    progress = np.dot(nv_pos - current_pos, direction)
                    if progress > best_progress:
                        best_progress = progress
                        best_neighbor = nv

            if best_neighbor is not None:
                projected = [mesh.vertices[best_neighbor]]
                proj_normal = mesh.vertex_normals[best_neighbor]

        points.append(projected[0].copy())
        current_pos = projected[0].copy()
        current_normal = proj_normal

    if len(points) < num_samples:
        points.append(end_pos.copy())

    return np.array(points)


def _get_vertex_neighbors(mesh: trimesh.Trimesh, vertex_idx: int) -> List[int]:
    """获取顶点的邻居顶点"""
    edges = mesh.edges_unique
    neighbors = []
    for e in edges:
        if e[0] == vertex_idx:
            neighbors.append(e[1])
        elif e[1] == vertex_idx:
            neighbors.append(e[0])
    return neighbors


def _straighten_path_aggressive(mesh: trimesh.Trimesh, path: np.ndarray) -> np.ndarray:
    """
    激进路径拉直 - 尽可能跳过中间点，生成最短路径

    使用贪心算法：从当前点出发，尝试跳到尽可能远的点
    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # 从当前点尝试跳到尽可能远的点
        best_jump = current_idx + 1

        for target_idx in range(len(path) - 1, current_idx + 1, -1):
            if _can_shortcut_strict(mesh, path[current_idx], path[target_idx]):
                best_jump = target_idx
                break

        result.append(path[best_jump])
        current_idx = best_jump

    return np.array(result)


def _can_shortcut_strict(
    mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    num_checks: int = 15
) -> bool:
    """
    检查两点之间是否可以走直线（不跳到另一面）

    通过检查中间点的法向量一致性和投影距离来判断
    """
    # 获取起点的法向量
    _, _, start_face = mesh.nearest.on_surface([start])
    start_normal = mesh.face_normals[int(start_face[0])]

    segment_length = np.linalg.norm(end - start)
    if segment_length < 1e-6:
        return True

    # 检查中间点
    for i in range(1, num_checks):
        t = i / num_checks
        mid_point = start * (1 - t) + end * t

        # 投影到曲面
        projected, dist, face_idx = mesh.nearest.on_surface([mid_point])
        mid_normal = mesh.face_normals[int(face_idx[0])]

        # 检查法向量一致性 - 允许更大的角度变化（曲面弯曲时法向量会变化）
        if np.dot(mid_normal, start_normal) < 0.0:  # 只有法向量完全反转才拒绝
            return False

        # 检查投影距离是否过大（相对于段长度）
        # 放宽条件：允许更大的投影距离
        if dist[0] > segment_length * 0.5:
            return False

    return True


def _straighten_path(mesh: trimesh.Trimesh, path: np.ndarray) -> np.ndarray:
    """
    路径拉直优化 - 尝试跳过中间点，同时保证路径不离开曲面同一侧

    使用贪心算法：从当前点出发，尝试跳到尽可能远的点
    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # 从当前点尝试跳到尽可能远的点
        best_jump = current_idx + 1  # 至少跳到下一个点

        for target_idx in range(len(path) - 1, current_idx + 1, -1):
            # 检查从当前点到目标点的直线是否可以安全走
            if _can_shortcut(mesh, path[current_idx], path[target_idx]):
                best_jump = target_idx
                break

        result.append(path[best_jump])
        current_idx = best_jump

    return np.array(result)


def _can_shortcut(mesh: trimesh.Trimesh, start: np.ndarray, end: np.ndarray, num_checks: int = 10) -> bool:
    """
    检查两点之间的直线投影是否可以安全走（不跳到另一面）

    通过检查中间点的法向量一致性来判断
    """
    # 获取起点的法向量
    _, _, start_face = mesh.nearest.on_surface([start])
    start_normal = mesh.face_normals[int(start_face[0])]

    # 检查中间点
    for i in range(1, num_checks):
        t = i / num_checks
        mid_point = start * (1 - t) + end * t

        # 投影到曲面
        projected, _, face_idx = mesh.nearest.on_surface([mid_point])
        mid_normal = mesh.face_normals[int(face_idx[0])]

        # 检查法向量一致性
        if np.dot(mid_normal, start_normal) < 0.3:
            # 法向量差异太大，说明跳到了另一面
            return False

        # 检查投影距离是否过大（说明跳跃了）
        dist = np.linalg.norm(projected[0] - mid_point)
        segment_length = np.linalg.norm(end - start)
        if dist > segment_length * 0.3:
            return False

    return True


def compute_geodesic_on_surface(
    mesh: trimesh.Trimesh,
    start_point: np.ndarray,
    end_point: np.ndarray,
    guide_path: Optional[np.ndarray] = None,
    num_samples: int = 60
) -> np.ndarray:
    """
    计算曲面上两点之间的测地线路径

    如果提供了guide_path（切平面上的引导路径），则沿引导路径映射
    否则使用Dijkstra+平滑

    Args:
        mesh: 网格模型
        start_point: 起点
        end_point: 终点
        guide_path: 引导路径（在切平面上规划好的路径点，已投影到3D）
        num_samples: 输出采样点数

    Returns:
        曲面上的路径点
    """
    if guide_path is not None and len(guide_path) > 2:
        # 使用引导路径，只需要平滑处理
        return smooth_path_on_surface(mesh, guide_path, num_samples)

    # 回退到Dijkstra最短路径
    start_idx = find_nearest_vertex(mesh, start_point)
    end_idx = find_nearest_vertex(mesh, end_point)

    if start_idx == end_idx:
        return np.array([start_point, end_point])

    path_indices = dijkstra_path(mesh, start_idx, end_idx)

    if len(path_indices) == 0:
        # 直接插值
        return interpolate_on_surface(mesh, start_point, end_point, num_samples)

    path_vertices = mesh.vertices[path_indices].copy()
    path_vertices[0] = project_to_surface(mesh, start_point)
    path_vertices[-1] = project_to_surface(mesh, end_point)

    return smooth_path_on_surface(mesh, path_vertices, num_samples)


def dijkstra_path(mesh: trimesh.Trimesh, start_idx: int, end_idx: int) -> List[int]:
    """Dijkstra最短路径"""
    n_vertices = len(mesh.vertices)
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length

    graph = {i: [] for i in range(n_vertices)}
    for (v1, v2), length in zip(edges, edge_lengths):
        graph[v1].append((v2, length))
        graph[v2].append((v1, length))

    distances = np.full(n_vertices, np.inf)
    distances[start_idx] = 0
    predecessors = np.full(n_vertices, -1, dtype=int)

    pq = [(0.0, start_idx)]
    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)

        if u == end_idx:
            break

        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph[u]:
            if v not in visited:
                new_dist = dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

    if predecessors[end_idx] == -1 and start_idx != end_idx:
        return []

    path = []
    current = end_idx
    while current != -1:
        path.append(current)
        current = predecessors[current]

    return path[::-1]


def dijkstra_path_smooth(
    mesh: trimesh.Trimesh,
    start_idx: int,
    end_idx: int,
    turn_penalty: float = 2.0,
    normal_penalty: float = 5.0
) -> List[int]:
    """
    方向连续性优先的Dijkstra路径算法

    类似高速公路设计：宁可绕远路也要避免急转弯，并且不绕到曲面另一侧

    Args:
        mesh: 网格模型
        start_idx: 起点顶点索引
        end_idx: 终点顶点索引
        turn_penalty: 转向惩罚权重（越大越平滑）
        normal_penalty: 法向量变化惩罚权重（防止绕到另一面）

    Returns:
        顶点索引列表
    """
    n_vertices = len(mesh.vertices)
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length

    # 计算起点到终点的参考方向（用于初始引导）
    start_pos = vertices[start_idx]
    end_pos = vertices[end_idx]
    goal_direction = end_pos - start_pos
    goal_dist = np.linalg.norm(goal_direction)
    if goal_dist > 1e-10:
        goal_direction = goal_direction / goal_dist

    # 获取起点的法向量（用于判断是否绕到另一面）
    start_normal = vertex_normals[start_idx]

    # 构建图 - 邻接表
    graph = {i: [] for i in range(n_vertices)}
    for (v1, v2), length in zip(edges, edge_lengths):
        graph[v1].append((v2, length))
        graph[v2].append((v1, length))

    # 状态：(顶点索引, 来源方向向量的哈希)
    # 简化：只跟踪顶点索引，但在代价计算中考虑来源方向

    # distances[v] = (cost, incoming_direction)
    distances = {}
    distances[start_idx] = (0.0, goal_direction)  # 初始方向指向目标
    predecessors = {start_idx: -1}

    # 优先队列: (cost, vertex_idx, incoming_direction)
    pq = [(0.0, start_idx, tuple(goal_direction))]
    visited = set()

    while pq:
        cost, u, incoming_dir_tuple = heapq.heappop(pq)
        incoming_dir = np.array(incoming_dir_tuple)

        if u == end_idx:
            break

        state_key = u
        if state_key in visited:
            continue
        visited.add(state_key)

        u_pos = vertices[u]
        u_normal = vertex_normals[u]

        for v, edge_length in graph[u]:
            if v in visited:
                continue

            v_pos = vertices[v]
            v_normal = vertex_normals[v]

            # 计算新的行进方向
            new_direction = v_pos - u_pos
            new_dir_norm = np.linalg.norm(new_direction)
            if new_dir_norm < 1e-10:
                continue
            new_direction = new_direction / new_dir_norm

            # 基础代价：边长
            base_cost = edge_length

            # 转向惩罚：与当前方向的夹角
            # cos(angle) = dot(incoming, new), 范围[-1, 1]
            # 惩罚 = turn_penalty * (1 - cos(angle)) / 2, 范围[0, turn_penalty]
            cos_angle = np.dot(incoming_dir, new_direction)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            turn_cost = turn_penalty * (1.0 - cos_angle) / 2.0 * edge_length

            # 法向量一致性惩罚：防止路径绕到曲面另一侧
            # 比较当前顶点法向量与起点法向量
            normal_dot = np.dot(v_normal, start_normal)
            if normal_dot < 0:
                # 法向量反向，说明绕到了另一面，施加大惩罚
                normal_cost = normal_penalty * edge_length * (1.0 - normal_dot)
            else:
                # 法向量大致同向，轻微惩罚偏离
                normal_cost = normal_penalty * edge_length * (1.0 - normal_dot) * 0.1

            # 额外：与目标方向的偏离惩罚（轻微引导向目标）
            to_goal = end_pos - v_pos
            to_goal_norm = np.linalg.norm(to_goal)
            if to_goal_norm > 1e-10:
                to_goal = to_goal / to_goal_norm
                goal_alignment = np.dot(new_direction, to_goal)
                # 如果朝向目标，给予轻微奖励
                goal_cost = -0.1 * edge_length * max(0, goal_alignment)
            else:
                goal_cost = 0

            # 总代价
            total_edge_cost = base_cost + turn_cost + normal_cost + goal_cost
            new_cost = cost + total_edge_cost

            # 更新最优路径
            if v not in distances or new_cost < distances[v][0]:
                distances[v] = (new_cost, new_direction)
                predecessors[v] = u
                heapq.heappush(pq, (new_cost, v, tuple(new_direction)))

    # 重建路径
    if end_idx not in predecessors:
        return []

    path = []
    current = end_idx
    while current != -1:
        path.append(current)
        current = predecessors.get(current, -1)

    return path[::-1]


def interpolate_on_surface(
    mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    num_points: int
) -> np.ndarray:
    """
    曲面上的路径插值 - 沿表面行走，不会跳到另一面

    方法：从起点开始，每步沿目标方向在曲面上移动一小步
    """
    # 确保起点和终点在曲面上
    start_on_surface, _, start_face = mesh.nearest.on_surface([start])
    end_on_surface, _, end_face = mesh.nearest.on_surface([end])
    start_pos = start_on_surface[0]
    end_pos = end_on_surface[0]
    start_face_idx = int(start_face[0])

    # 计算总距离和步长
    total_dist = np.linalg.norm(end_pos - start_pos)
    if total_dist < 1e-10:
        return np.array([start_pos, end_pos])

    # 获取起点的法向量
    start_normal = mesh.face_normals[start_face_idx]

    # 步进参数
    step_size = total_dist / (num_points - 1)

    points = [start_pos.copy()]
    current_pos = start_pos.copy()
    current_face = start_face_idx
    current_normal = start_normal.copy()

    for i in range(1, num_points):
        # 目标方向
        direction = end_pos - current_pos
        dist_to_end = np.linalg.norm(direction)

        if dist_to_end < step_size * 0.5:
            # 已经很接近终点
            points.append(end_pos.copy())
            break

        direction = direction / dist_to_end

        # 移动一步
        next_pos_guess = current_pos + direction * min(step_size, dist_to_end)

        # 投影到曲面，但要确保不跳到另一面
        candidates, distances, face_indices = mesh.nearest.on_surface([next_pos_guess])
        candidate_pos = candidates[0]
        candidate_face = int(face_indices[0])
        candidate_normal = mesh.face_normals[candidate_face]

        # 检查法向量一致性 - 如果法向量反转，说明跳到了另一面
        normal_dot = np.dot(candidate_normal, current_normal)

        if normal_dot < 0.0:
            # 法向量反转，需要沿表面走而不是直接跳过去
            # 使用局部搜索找到同侧的点
            candidate_pos, candidate_face = _find_surface_step(
                mesh, current_pos, current_face, direction, step_size
            )
            if candidate_pos is None:
                # 搜索失败，使用原始候选点
                candidate_pos = candidates[0]
                candidate_face = int(face_indices[0])

        points.append(candidate_pos.copy())
        current_pos = candidate_pos
        current_face = candidate_face
        current_normal = mesh.face_normals[current_face]

    # 确保终点正确
    if len(points) < num_points:
        points.append(end_pos.copy())

    return np.array(points)


def _find_surface_step(
    mesh: trimesh.Trimesh,
    current_pos: np.ndarray,
    current_face: int,
    direction: np.ndarray,
    step_size: float
) -> Tuple[Optional[np.ndarray], int]:
    """
    在曲面上找到下一步位置，确保不跳到另一面

    通过搜索相邻面来找到合适的下一步
    """
    # 获取当前面的相邻面
    try:
        # 获取当前面的顶点
        face_vertices = mesh.faces[current_face]

        # 找到包含这些顶点的所有面
        adjacent_faces = set()
        for vertex in face_vertices:
            # 找到包含这个顶点的所有面
            faces_with_vertex = np.where(np.any(mesh.faces == vertex, axis=1))[0]
            adjacent_faces.update(faces_with_vertex)

        # 当前面的法向量
        current_normal = mesh.face_normals[current_face]

        # 在相邻面中搜索最佳下一步位置
        best_pos = None
        best_face = current_face
        best_progress = -np.inf

        for adj_face in adjacent_faces:
            adj_normal = mesh.face_normals[adj_face]

            # 检查法向量一致性（同侧）
            if np.dot(adj_normal, current_normal) < 0.3:
                continue  # 跳过法向量差异太大的面

            # 计算这个面的中心
            face_center = mesh.vertices[mesh.faces[adj_face]].mean(axis=0)

            # 从当前位置到面中心的方向
            to_center = face_center - current_pos
            dist_to_center = np.linalg.norm(to_center)

            if dist_to_center < 1e-10:
                continue

            to_center_normalized = to_center / dist_to_center

            # 计算这个方向与目标方向的一致性
            progress = np.dot(to_center_normalized, direction)

            if progress > best_progress and dist_to_center < step_size * 3:
                # 在这个面上找一个更精确的点
                target_on_face = current_pos + direction * min(step_size, dist_to_center)

                # 投影到这个面
                face_verts = mesh.vertices[mesh.faces[adj_face]]
                projected = _project_to_triangle(target_on_face, face_verts)

                if projected is not None:
                    best_pos = projected
                    best_face = adj_face
                    best_progress = progress

        return best_pos, best_face

    except Exception:
        return None, current_face


def _project_to_triangle(point: np.ndarray, triangle: np.ndarray) -> Optional[np.ndarray]:
    """将点投影到三角形平面上，如果在三角形外则返回最近边上的点"""
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    # 计算三角形法向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    normal_len = np.linalg.norm(normal)

    if normal_len < 1e-10:
        return None

    normal = normal / normal_len

    # 将点投影到三角形平面
    d = np.dot(normal, v0)
    t = (d - np.dot(normal, point)) / (np.dot(normal, normal) + 1e-10)
    projected = point + t * normal

    # 检查投影点是否在三角形内（使用重心坐标）
    v0p = projected - v0
    dot00 = np.dot(edge1, edge1)
    dot01 = np.dot(edge1, edge2)
    dot02 = np.dot(edge1, v0p)
    dot11 = np.dot(edge2, edge2)
    dot12 = np.dot(edge2, v0p)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return None

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # 如果在三角形内
    if u >= 0 and v >= 0 and u + v <= 1:
        return projected

    # 否则找到最近的边上的点
    # 简化：返回投影点的最近顶点
    dists = [np.linalg.norm(projected - v0),
             np.linalg.norm(projected - v1),
             np.linalg.norm(projected - v2)]
    nearest_idx = np.argmin(dists)
    return triangle[nearest_idx].copy()


def smooth_path_on_surface(
    mesh: trimesh.Trimesh,
    path: np.ndarray,
    num_samples: int
) -> np.ndarray:
    """平滑路径并保持在曲面上"""
    if len(path) < 3:
        return path

    # 重采样
    resampled = resample_path(path, num_samples)

    # Laplacian平滑
    smoothed = resampled.copy()
    alpha = 0.5

    for _ in range(5):
        new_smoothed = smoothed.copy()
        for i in range(1, len(smoothed) - 1):
            avg = 0.5 * (smoothed[i-1] + smoothed[i+1])
            new_smoothed[i] = (1 - alpha) * smoothed[i] + alpha * avg
            new_smoothed[i] = project_to_surface(mesh, new_smoothed[i])
        smoothed = new_smoothed

    return smoothed


def resample_path(path: np.ndarray, num_samples: int) -> np.ndarray:
    """等距重采样路径"""
    if len(path) < 2:
        return path

    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-10:
        return path

    target_distances = np.linspace(0, total_length, num_samples)
    resampled = []

    for target_dist in target_distances:
        idx = np.searchsorted(cumulative, target_dist, side='right') - 1
        idx = np.clip(idx, 0, len(path) - 2)

        segment_length = cumulative[idx + 1] - cumulative[idx]
        if segment_length < 1e-10:
            resampled.append(path[idx].copy())
        else:
            t = (target_dist - cumulative[idx]) / segment_length
            point = path[idx] * (1 - t) + path[idx + 1] * t
            resampled.append(point)

    return np.array(resampled)


def compute_angle_from_center(
    ir_position: np.ndarray,
    center_position: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray
) -> float:
    """计算IR点相对于中心点的方位角"""
    point_2d = project_to_tangent_plane(ir_position, center_position, x_axis, y_axis)
    angle = np.arctan2(point_2d[1], point_2d[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle


def plan_paths_on_surface(
    ir_points: Dict[str, np.ndarray],
    center_position: np.ndarray,
    mesh: trimesh.Trimesh,
    center_normal: Optional[np.ndarray] = None,
    use_dijkstra: bool = True,
    smooth_path: bool = False,
    turn_penalty: float = 2.0,
    normal_penalty: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    在曲面上规划不交叉的弧线路径

    核心方法：
    1. 建立中心点的切平面坐标系（仅用于确定方向）
    2. 将所有IR点按角度排序（确保路径不交叉）
    3. 使用Dijkstra算法在网格上找路径（保证路径在曲面上）

    Args:
        ir_points: IR点3D坐标 {point_id: position}
        center_position: 中心点3D坐标
        mesh: 网格模型
        center_normal: 中心点表面法向量
        use_dijkstra: 是否使用Dijkstra路径（推荐True，对高曲率更稳定）
        smooth_path: 是否使用平滑路径算法（方向连续性优先，避免绕到另一面）
        turn_penalty: 转向惩罚权重（越大路径越平滑）
        normal_penalty: 法向量变化惩罚权重（防止路径绕到曲面另一侧）

    Returns:
        3D路径字典 {point_id: 路径点数组}
    """
    print("\n" + "="*60)
    print("路径规划开始")
    print("="*60)

    if center_normal is None:
        center_idx = find_nearest_vertex(mesh, center_position)
        center_normal = mesh.vertex_normals[center_idx]
        print(f"自动获取中心点法向量: {center_normal}")

    # 确保中心点在曲面上
    center_on_surface = project_to_surface(mesh, center_position)

    print(f"中心点位置: {center_position}")
    print(f"投影到曲面: {center_on_surface}")
    print(f"中心点法向量: {center_normal}")
    print(f"路径模式: {'Dijkstra边路径' if smooth_path else '直线投影（最短）'}")

    # 建立切平面坐标系（仅用于排序和方向参考）
    x_axis, y_axis, normal = build_tangent_plane_basis(center_normal)

    print(f"\n切平面坐标系:")
    print(f"  X轴: {x_axis}")
    print(f"  Y轴: {y_axis}")
    print(f"  法向: {normal}")

    # 验证正交性
    print(f"  正交性检查: X·Y={np.dot(x_axis, y_axis):.6f}, X·N={np.dot(x_axis, normal):.6f}, Y·N={np.dot(y_axis, normal):.6f}")

    # 将所有IR点投影到切平面并计算角度（仅用于排序）
    ir_points_2d = {}
    ir_angles = {}
    ir_distances = {}

    print(f"\nIR点投影信息 ({len(ir_points)} 个点):")
    for point_id, position in ir_points.items():
        point_2d = project_to_tangent_plane(position, center_on_surface, x_axis, y_axis)
        ir_points_2d[point_id] = point_2d
        angle = np.arctan2(point_2d[1], point_2d[0])
        ir_angles[point_id] = angle
        distance_3d = np.linalg.norm(position - center_on_surface)
        distance_2d = np.linalg.norm(point_2d)
        ir_distances[point_id] = (distance_3d, distance_2d)

        print(f"  {point_id[:8]}:")
        print(f"    3D位置: {position}")
        print(f"    2D投影: {point_2d}")
        print(f"    角度: {np.degrees(angle):.1f}°")
        print(f"    3D距离: {distance_3d:.2f}, 2D距离: {distance_2d:.2f}")

    # 按角度排序
    sorted_ids = sorted(ir_points.keys(), key=lambda pid: ir_angles[pid])

    print(f"\n按角度排序后的点:")
    for i, pid in enumerate(sorted_ids):
        print(f"  {i+1}. {pid[:8]} - 角度: {np.degrees(ir_angles[pid]):.1f}°")

    paths_3d = {}

    print(f"\n开始生成路径:")
    for i, point_id in enumerate(sorted_ids):
        ir_position = ir_points[point_id]
        ir_pos_2d = ir_points_2d[point_id]

        print(f"\n路径 {i+1}/{len(sorted_ids)} - {point_id[:8]}:")
        print(f"  IR点位置: {ir_position}")

        if use_dijkstra:
            # 使用直线插值或Dijkstra算法
            path_3d = compute_geodesic_dijkstra(
                mesh,
                center_on_surface,
                ir_position,
                num_samples=60,
                smooth_path=smooth_path,
                turn_penalty=turn_penalty,
                normal_penalty=normal_penalty
            )
        else:
            # 旧方法：切平面映射（可能有高曲率问题）
            center_2d = np.array([0.0, 0.0])
            path_2d = create_arc_path_2d(center_2d, ir_pos_2d, num_points=60, curvature=0.0)

            print(f"  2D路径: 从 {center_2d} 到 {ir_pos_2d}")
            print(f"  2D路径点数: {len(path_2d)}")

            # 使用迭代方法映射
            path_3d_guide = map_2d_path_to_3d_surface_iterative(
                path_2d, center_on_surface, x_axis, y_axis, mesh, debug=True
            )

            # 确保端点精确
            path_3d_guide[0] = project_to_surface(mesh, center_on_surface)
            path_3d_guide[-1] = project_to_surface(mesh, ir_position)

            # 平滑处理
            path_3d = smooth_path_on_surface(mesh, path_3d_guide, num_samples=60)

        # 确保端点精确
        path_3d[0] = project_to_surface(mesh, center_on_surface)
        path_3d[-1] = project_to_surface(mesh, ir_position)

        final_length = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))
        print(f"  路径点数: {len(path_3d)}")
        print(f"  路径长度: {final_length:.2f}")
        print(f"  起点: {path_3d[0]}")
        print(f"  终点: {path_3d[-1]}")

        # 检查路径是否有大的跳跃
        diffs = np.linalg.norm(np.diff(path_3d, axis=0), axis=1)
        max_step = np.max(diffs)
        avg_step = np.mean(diffs)
        if max_step > avg_step * 5:
            print(f"  警告: 检测到路径跳跃! 最大步长={max_step:.2f}, 平均步长={avg_step:.2f}")

        # 反转路径（从IR点到中心点）
        path_3d = path_3d[::-1]

        print(f"  反转后起点: {path_3d[0]} (应接近IR点)")
        print(f"  反转后终点: {path_3d[-1]} (应接近中心点)")

        if len(path_3d) > 0:
            paths_3d[point_id] = path_3d

    print(f"\n" + "="*60)
    print(f"路径规划完成: 共生成 {len(paths_3d)} 条路径")
    print("="*60 + "\n")

    return paths_3d


def verify_no_intersection(
    paths: Dict[str, np.ndarray],
    tolerance: float = 0.5
) -> bool:
    """验证路径之间没有交叉"""
    path_ids = list(paths.keys())

    for i in range(len(path_ids)):
        for j in range(i + 1, len(path_ids)):
            path1 = paths[path_ids[i]]
            path2 = paths[path_ids[j]]

            # 检查中间点是否过近
            inner1 = path1[1:-1] if len(path1) > 2 else np.array([])
            inner2 = path2[1:-1] if len(path2) > 2 else np.array([])

            if len(inner1) == 0 or len(inner2) == 0:
                continue

            for p1 in inner1:
                for p2 in inner2:
                    dist = np.linalg.norm(p1 - p2)
                    if dist < tolerance:
                        return False

    return True
