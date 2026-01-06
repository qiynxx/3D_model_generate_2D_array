"""测地线路径计算模块"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import uuid
import heapq
from scipy.interpolate import splprep, splev, CubicSpline
from .path_planner_2d import plan_paths_on_surface


def project_point_to_mesh(point: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """将点投影到网格表面最近位置"""
    distances = np.linalg.norm(mesh.vertices - point, axis=1)
    nearest_idx = np.argmin(distances)
    return mesh.vertices[nearest_idx].copy()


def smooth_path_bspline(
    path_points: np.ndarray,
    mesh: trimesh.Trimesh,
    num_samples: int = 100,
    smoothing: float = 0.0
) -> np.ndarray:
    """
    使用B样条平滑路径

    Args:
        path_points: 原始路径点
        mesh: 网格模型
        num_samples: 输出采样点数
        smoothing: 平滑因子 (0=插值通过所有点, >0=平滑)

    Returns:
        平滑后的路径点
    """
    if len(path_points) < 4:
        return path_points

    try:
        # 去除重复点
        unique_mask = np.ones(len(path_points), dtype=bool)
        for i in range(1, len(path_points)):
            if np.allclose(path_points[i], path_points[i-1], atol=1e-6):
                unique_mask[i] = False
        unique_points = path_points[unique_mask]

        if len(unique_points) < 4:
            return path_points

        # 使用B样条拟合
        k = min(3, len(unique_points) - 1)  # 样条阶数

        # 限制平滑因子，避免过大
        max_smooth = len(unique_points) * 0.1
        s = min(smoothing, max_smooth)

        tck, u = splprep(
            [unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]],
            s=s,
            k=k
        )

        # 重新采样
        u_new = np.linspace(0, 1, num_samples)
        smooth_points = np.array(splev(u_new, tck)).T

        # 投影回网格表面
        projected = []
        for point in smooth_points:
            proj = project_point_to_mesh(point, mesh)
            projected.append(proj)

        return np.array(projected)

    except Exception as e:
        # 回退到原始路径
        return path_points


def create_smooth_direct_path(
    start: np.ndarray,
    end: np.ndarray,
    mesh: trimesh.Trimesh,
    num_points: int = 50
) -> np.ndarray:
    """
    创建从起点到终点的平滑直接路径

    使用直线插值，然后投影到曲面

    Args:
        start: 起点
        end: 终点
        mesh: 网格模型
        num_points: 路径点数

    Returns:
        平滑路径点数组
    """
    # 线性插值
    t = np.linspace(0, 1, num_points)
    path_points = []

    for ti in t:
        point = start * (1 - ti) + end * ti
        # 投影到曲面
        projected = project_point_to_mesh(point, mesh)
        path_points.append(projected)

    return np.array(path_points)


def catmull_rom_spline(
    points: np.ndarray,
    num_samples: int = 50,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Catmull-Rom样条插值

    Args:
        points: 控制点
        num_samples: 每段采样数
        alpha: 张力参数 (0=uniform, 0.5=centripetal, 1=chordal)

    Returns:
        插值后的点
    """
    if len(points) < 2:
        return points
    if len(points) == 2:
        return np.linspace(points[0], points[1], num_samples)

    def get_t(t, alpha, p0, p1):
        d = p1 - p0
        a = np.dot(d, d)
        return t + a ** (alpha * 0.5)

    result = []

    # 添加虚拟端点
    pts = np.vstack([
        2 * points[0] - points[1],
        points,
        2 * points[-1] - points[-2]
    ])

    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]

        t0 = 0
        t1 = get_t(t0, alpha, p0, p1)
        t2 = get_t(t1, alpha, p1, p2)
        t3 = get_t(t2, alpha, p2, p3)

        t = np.linspace(t1, t2, num_samples // (len(points) - 1) + 1)

        for tj in t[:-1] if i < len(pts) - 3 else t:
            A1 = (t1 - tj) / (t1 - t0) * p0 + (tj - t0) / (t1 - t0) * p1
            A2 = (t2 - tj) / (t2 - t1) * p1 + (tj - t1) / (t2 - t1) * p2
            A3 = (t3 - tj) / (t3 - t2) * p2 + (tj - t2) / (t3 - t2) * p3

            B1 = (t2 - tj) / (t2 - t0) * A1 + (tj - t0) / (t2 - t0) * A2
            B2 = (t3 - tj) / (t3 - t1) * A2 + (tj - t1) / (t3 - t1) * A3

            C = (t2 - tj) / (t2 - t1) * B1 + (tj - t1) / (t2 - t1) * B2
            result.append(C)

    return np.array(result)


@dataclass
class IRPoint:
    """IR点数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_index: int = -1
    barycentric: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    is_center: bool = False
    connections: List[str] = field(default_factory=list)
    group: str = "default"
    name: str = ""
    # 焊盘尺寸（可单独调整）
    pad_length: float = 3.0  # 方形焊盘长度
    pad_width: float = 2.0   # 方形焊盘宽度

    def __post_init__(self):
        if not self.name:
            self.name = f"IR_{self.id}"


class GeodesicSolver:
    """测地线求解器"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self._build_graph()

    def _build_graph(self):
        """构建网格图用于Dijkstra算法"""
        n_vertices = len(self.vertices)
        self.graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_vertices)}

        # 从边构建图
        edges = self.mesh.edges_unique
        edge_lengths = self.mesh.edges_unique_length

        for (v1, v2), length in zip(edges, edge_lengths):
            self.graph[v1].append((v2, length))
            self.graph[v2].append((v1, length))

    def compute_distances(self, source_vertex: int) -> np.ndarray:
        """
        从源顶点计算到所有顶点的测地线距离（Dijkstra算法）

        Args:
            source_vertex: 源顶点索引

        Returns:
            距离数组
        """
        n = len(self.vertices)
        distances = np.full(n, np.inf)
        distances[source_vertex] = 0

        # 优先队列：(距离, 顶点索引)
        pq = [(0.0, source_vertex)]
        visited = set()

        while pq:
            dist, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            for v, weight in self.graph[u]:
                if v not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))

        return distances

    def find_path(
        self,
        start_vertex: int,
        end_vertex: int
    ) -> List[int]:
        """
        找到两个顶点之间的最短路径

        Args:
            start_vertex: 起始顶点
            end_vertex: 终止顶点

        Returns:
            顶点索引列表
        """
        n = len(self.vertices)
        distances = np.full(n, np.inf)
        distances[start_vertex] = 0
        predecessors = np.full(n, -1, dtype=int)

        pq = [(0.0, start_vertex)]
        visited = set()

        while pq:
            dist, u = heapq.heappop(pq)

            if u == end_vertex:
                break

            if u in visited:
                continue
            visited.add(u)

            for v, weight in self.graph[u]:
                if v not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        predecessors[v] = u
                        heapq.heappush(pq, (new_dist, v))

        # 回溯路径
        if predecessors[end_vertex] == -1 and start_vertex != end_vertex:
            return []

        path = []
        current = end_vertex
        while current != -1:
            path.append(current)
            current = predecessors[current]

        return path[::-1]

    def find_nearest_vertex(self, point: np.ndarray) -> int:
        """找到离给定点最近的顶点"""
        distances = np.linalg.norm(self.vertices - point, axis=1)
        return int(np.argmin(distances))

    def compute_path_between_points(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        计算两个3D点之间的测地线路径

        Args:
            start_point: 起点3D坐标
            end_point: 终点3D坐标

        Returns:
            (顶点索引列表, 路径点坐标数组)
        """
        start_v = self.find_nearest_vertex(start_point)
        end_v = self.find_nearest_vertex(end_point)

        path_indices = self.find_path(start_v, end_v)

        if not path_indices:
            return [], np.array([])

        path_points = self.vertices[path_indices]
        return path_indices, path_points


class PathManager:
    """路径管理器"""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.solver = GeodesicSolver(mesh)
        self.ir_points: Dict[str, IRPoint] = {}
        self.center_point_id: Optional[str] = None
        self.paths: Dict[str, Tuple[List[int], np.ndarray]] = {}
        self.smooth_paths: Dict[str, np.ndarray] = {}  # 平滑后的路径
        self.smoothness: float = 0.5  # 平滑度参数
        self._point_counter: int = 0  # 点序号计数器

    def set_smoothness(self, smoothness: float):
        """设置平滑度 (0-1)"""
        self.smoothness = max(0.0, min(1.0, smoothness))

    def add_point(
        self,
        position: np.ndarray,
        face_index: int,
        is_center: bool = False,
        name: str = ""
    ) -> IRPoint:
        """添加IR点"""
        # 如果未指定名称，使用序列号
        if not name:
            name = f"P{self._point_counter}"
            self._point_counter += 1

        point = IRPoint(
            position=position.copy(),
            face_index=face_index,
            is_center=is_center,
            name=name
        )

        self.ir_points[point.id] = point

        if is_center:
            self.set_center_point(point.id)

        return point

    def remove_point(self, point_id: str):
        """移除IR点"""
        if point_id in self.ir_points:
            if self.center_point_id == point_id:
                self.center_point_id = None
            del self.ir_points[point_id]
            # 移除相关路径
            self.paths.pop(point_id, None)

    def set_center_point(self, point_id: str):
        """设置中心连接点"""
        # 取消之前的中心点
        if self.center_point_id and self.center_point_id in self.ir_points:
            self.ir_points[self.center_point_id].is_center = False

        self.center_point_id = point_id
        if point_id in self.ir_points:
            self.ir_points[point_id].is_center = True

    def compute_all_paths(self, smooth: bool = True, exclude_point_ids: list = None) -> Dict[str, Tuple[List[int], np.ndarray]]:
        """
        计算所有IR点到中心点的路径

        使用2D路径规划法：
        1. 将所有点投影到2D平面
        2. 在2D平面上规划直线路径（遇障碍用弧线绕行）
        3. 将2D路径映射回3D曲面

        Args:
            smooth: 是否额外平滑（B样条）
            exclude_point_ids: 要排除的点ID列表（如坐标原点）
        """
        self.paths.clear()
        self.smooth_paths.clear()

        if exclude_point_ids is None:
            exclude_point_ids = []

        print("\n" + "#"*60)
        print("# 开始计算路径")
        print("#"*60)

        # 打印网格信息
        print(f"\n网格信息:")
        print(f"  顶点数: {len(self.mesh.vertices)}")
        print(f"  面数: {len(self.mesh.faces)}")
        bounds = self.mesh.bounds
        print(f"  边界框: [{bounds[0]}] - [{bounds[1]}]")
        print(f"  尺寸: {bounds[1] - bounds[0]}")

        if not self.center_point_id:
            print("警告: 未设置中心点")
            return self.paths

        center = self.ir_points.get(self.center_point_id)
        if not center:
            print("警告: 找不到中心点")
            return self.paths

        print(f"\n中心点: {center.name} ({self.center_point_id[:8]})")
        print(f"  位置: {center.position}")

        # 收集IR点位置
        ir_positions = {}
        print(f"\nIR点列表 (共 {len(self.ir_points)} 个):")
        for point_id, point in self.ir_points.items():
            # 跳过中心点和排除的点
            if point_id == self.center_point_id:
                continue
            if point_id in exclude_point_ids:
                print(f"  {point.name} ({point_id[:8]}): 已排除（坐标原点）")
                continue
            ir_positions[point_id] = point.position
            dist = np.linalg.norm(point.position - center.position)
            print(f"  {point.name} ({point_id[:8]}): {point.position}, 距中心: {dist:.2f}")

        if not ir_positions:
            print("警告: 没有非中心点的IR点")
            return self.paths

        # 获取中心点法向量
        distances = np.linalg.norm(self.mesh.vertices - center.position, axis=1)
        nearest_idx = np.argmin(distances)
        center_normal = self.mesh.vertex_normals[nearest_idx]

        print(f"开始2D路径规划: {len(ir_positions)} 个IR点")

        # 使用2D路径规划
        try:
            paths_3d = plan_paths_on_surface(
                ir_positions,
                center.position,
                self.mesh,
                center_normal
            )
            print(f"2D路径规划完成: 生成 {len(paths_3d)} 条路径")
        except Exception as e:
            print(f"2D路径规划失败: {e}")
            # 回退到直接投影法
            paths_3d = {}
            for point_id, point in self.ir_points.items():
                if point_id == self.center_point_id:
                    continue
                direct_path = create_smooth_direct_path(
                    point.position,
                    center.position,
                    self.mesh,
                    num_points=80
                )
                if len(direct_path) > 0:
                    paths_3d[point_id] = direct_path

        # 保存路径
        for point_id, path_3d in paths_3d.items():
            if len(path_3d) > 0:
                self.paths[point_id] = ([], path_3d)

                point = self.ir_points.get(point_id)
                if point:
                    point.connections = [self.center_point_id]

                # 计算原始路径长度和端点
                orig_length = np.sum(np.linalg.norm(np.diff(path_3d, axis=0), axis=1))
                orig_start = path_3d[0]
                orig_end = path_3d[-1]

                # 不再使用B样条平滑，直接使用原始路径（已经足够平滑）
                # B样条会导致路径脱离曲面
                self.smooth_paths[point_id] = path_3d
                print(f"\n路径 {point_id[:8]}: {len(path_3d)} 点, 长度 {orig_length:.2f}")

        print(f"\n路径计算完成: {len(self.smooth_paths)} 条路径")
        print("#"*60 + "\n")
        return self.paths

    def _smooth_path_iterative(self, path: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """迭代平滑路径（移动平均）"""
        if len(path) < 3:
            return path

        smoothed = path.copy()
        # 保持起点和终点不变
        for i in range(1, len(path) - 1):
            # 加权平均：当前点 + 相邻点
            smoothed[i] = (1 - alpha) * path[i] + alpha * 0.5 * (path[i-1] + path[i+1])

        return smoothed

    def _project_path_to_surface(self, path: np.ndarray) -> np.ndarray:
        """将路径点投影到网格表面"""
        projected = []
        for point in path:
            distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            projected.append(self.mesh.vertices[nearest_idx].copy())
        return np.array(projected)

    def get_smooth_path(self, point_id: str) -> Optional[np.ndarray]:
        """获取平滑后的路径"""
        return self.smooth_paths.get(point_id)

    def get_all_smooth_paths(self) -> Dict[str, np.ndarray]:
        """获取所有平滑路径"""
        return self.smooth_paths

    def get_all_path_points(self) -> List[np.ndarray]:
        """获取所有路径的点坐标"""
        return [path_points for _, path_points in self.paths.values()]

    def get_path_normals(self, path_indices: List[int]) -> np.ndarray:
        """获取路径上各点的表面法向量"""
        # 使用顶点法向量
        vertex_normals = self.mesh.vertex_normals
        return vertex_normals[path_indices]

    def to_dict(self) -> dict:
        """导出为字典格式"""
        return {
            'points': {
                pid: {
                    'id': p.id,
                    'position': p.position.tolist(),
                    'face_index': p.face_index,
                    'is_center': p.is_center,
                    'connections': p.connections,
                    'group': p.group,
                    'name': p.name,
                }
                for pid, p in self.ir_points.items()
            },
            'center_point_id': self.center_point_id,
        }

    def from_dict(self, data: dict):
        """从字典格式导入"""
        self.ir_points.clear()
        self.paths.clear()

        for pid, pdata in data.get('points', {}).items():
            point = IRPoint(
                id=pdata['id'],
                position=np.array(pdata['position']),
                face_index=pdata['face_index'],
                is_center=pdata['is_center'],
                connections=pdata['connections'],
                group=pdata['group'],
                name=pdata['name'],
            )
            self.ir_points[pid] = point

        self.center_point_id = data.get('center_point_id')
