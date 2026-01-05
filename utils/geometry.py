"""几何工具函数"""
import numpy as np
from typing import Tuple, List, Optional


def point_to_barycentric(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> Tuple[float, float, float]:
    """将3D点转换为重心坐标"""
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


def barycentric_to_point(
    bary: Tuple[float, float, float],
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """将重心坐标转换为3D点"""
    return bary[0] * v0 + bary[1] * v1 + bary[2] * v2


def compute_face_normal(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """计算三角面法向量"""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        return normal / norm
    return np.array([0, 0, 1])


def project_point_to_plane(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """将点投影到平面上"""
    d = np.dot(point - plane_point, plane_normal)
    return point - d * plane_normal


def compute_path_length(points: np.ndarray) -> float:
    """计算路径长度"""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def resample_path(
    points: np.ndarray,
    num_samples: int
) -> np.ndarray:
    """等距重采样路径点"""
    if len(points) < 2:
        return points

    # 计算累积距离
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]

    if total_length < 1e-10:
        return points

    # 等距采样
    sample_distances = np.linspace(0, total_length, num_samples)
    resampled = np.zeros((num_samples, 3))

    for i, d in enumerate(sample_distances):
        idx = np.searchsorted(cumulative, d, side='right') - 1
        idx = np.clip(idx, 0, len(points) - 2)

        if cumulative[idx + 1] - cumulative[idx] < 1e-10:
            resampled[i] = points[idx]
        else:
            t = (d - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
            resampled[i] = points[idx] + t * (points[idx + 1] - points[idx])

    return resampled


def offset_path_on_surface(
    path_points: np.ndarray,
    normals: np.ndarray,
    offset: float
) -> Tuple[np.ndarray, np.ndarray]:
    """沿表面法向偏移路径，生成左右两侧边界"""
    n = len(path_points)
    if n < 2:
        return path_points, path_points

    # 计算路径切向量
    tangents = np.zeros_like(path_points)
    tangents[0] = path_points[1] - path_points[0]
    tangents[-1] = path_points[-1] - path_points[-2]
    for i in range(1, n - 1):
        tangents[i] = path_points[i + 1] - path_points[i - 1]

    # 归一化
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    tangents = tangents / norms

    # 计算侧向向量（切向 × 法向）
    side_vectors = np.cross(tangents, normals)
    norms = np.linalg.norm(side_vectors, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    side_vectors = side_vectors / norms

    # 偏移
    left = path_points + offset * side_vectors
    right = path_points - offset * side_vectors

    return left, right
