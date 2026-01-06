"""3D视图组件"""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, QObject
from typing import Optional, List, Tuple, Callable
import trimesh
import logging
import os

# 抑制VTK的shader调试输出
os.environ.setdefault('VTK_SILENCE_GET_VOID_POINTER_WARNINGS', '1')

# 抑制VTK警告
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# 配置pyvista减少输出
pv.set_plot_theme('document')
pv.global_theme.multi_rendering_splitting_position = None

# 抑制相关日志
logging.getLogger('pyvista').setLevel(logging.ERROR)
logging.getLogger('vtk').setLevel(logging.ERROR)


class PickerCallback(QObject):
    """拾取回调信号"""
    point_picked = pyqtSignal(np.ndarray, int)  # (position, face_index)
    ir_point_selected = pyqtSignal(str)  # 选中的IR点ID


class Viewer3D(QWidget):
    """3D视图组件"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mesh: Optional[trimesh.Trimesh] = None
        self.pv_mesh: Optional[pv.PolyData] = None

        # 可视化元素
        self.ir_point_actors = {}  # id -> actor
        self.ir_point_positions = {}  # id -> position
        self.ir_point_is_center = {}  # id -> is_center
        self.path_actors = []
        self.groove_actors = []

        # 当前选中的IR点
        self.selected_ir_point_id: Optional[str] = None

        # 坐标轴相关
        self._coord_axis_actors = {}  # 坐标轴actor
        self._coord_axes_visible = False
        self._origin_highlight_actor = None

        # 拾取回调
        self.picker_callback = PickerCallback()
        self.picking_enabled = False

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # PyVista渲染器
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # 设置背景
        self.plotter.set_background('#1a1a2e')
        self.plotter.add_axes()

        # 注意: enable_shadows() 在某些系统上会导致OpenGL shader错误
        # 已禁用以避免调试输出

    def _enable_picking(self):
        """启用点击拾取 - 使用单击回调，不干扰视角控制"""
        # 使用 track_click_position 监听单击事件
        # 这种方式不会干扰正常的视角旋转/缩放
        try:
            self.plotter.track_click_position(
                callback=self._on_click,
                side='left',
                double=False
            )
        except Exception as e:
            # 备选方案：使用 iren 的观察者
            try:
                self._setup_click_observer()
            except Exception as e2:
                print(f"点击拾取设置失败: {e2}")

    def _setup_click_observer(self):
        """使用VTK的交互器观察者来监听点击"""
        iren = self.plotter.iren.interactor

        def on_left_button_release(obj, event):
            if not self.picking_enabled and not self.ir_point_positions:
                return
            # 获取点击位置
            click_pos = iren.GetEventPosition()
            self._pick_at_position(click_pos)

        iren.AddObserver('LeftButtonReleaseEvent', on_left_button_release)

    def _on_click(self, point):
        """单击回调 - point 是3D世界坐标"""
        if point is None:
            return

        # track_click_position 返回的是3D世界坐标，不是屏幕坐标
        if hasattr(point, '__len__') and len(point) >= 3:
            point_3d = np.array([point[0], point[1], point[2]])
            self._pick_from_3d_point(point_3d)

    def _pick_from_3d_point(self, clicked_point: np.ndarray):
        """
        从3D点击位置进行拾取 - 使用射线投射找到离相机最近的表面点
        """
        if self.mesh is None:
            return

        try:
            # 获取相机位置
            camera_pos = np.array(self.plotter.camera_position[0])

            # 从相机到点击位置的射线方向
            ray_direction = clicked_point - camera_pos
            ray_length = np.linalg.norm(ray_direction)
            if ray_length < 1e-10:
                return
            ray_direction = ray_direction / ray_length

            # 使用trimesh进行射线-网格相交测试
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=[camera_pos],
                ray_directions=[ray_direction]
            )

            if len(locations) == 0:
                return  # 没有拾取到任何点

            # 找到离相机最近的交点
            distances_to_camera = np.linalg.norm(locations - camera_pos, axis=1)
            nearest_idx = np.argmin(distances_to_camera)

            picked_point = locations[nearest_idx]
            face_idx = int(index_tri[nearest_idx])

            # 处理拾取的点
            self._process_picked_point_with_face(picked_point, face_idx)

        except Exception:
            pass

    def _pick_at_position(self, click_pos):
        """
        在屏幕位置进行拾取 - 使用射线投射找到离相机最近的表面点

        关键：从相机位置发射射线穿过鼠标点击位置，找到第一个交点
        """
        if self.mesh is None:
            return

        try:
            # 获取相机位置
            camera_pos = np.array(self.plotter.camera_position[0])

            # 将屏幕坐标转换为世界坐标射线
            renderer = self.plotter.renderer

            # 使用VTK的坐标转换
            # 创建一个世界坐标点拾取器
            coordinate = vtk.vtkCoordinate()
            coordinate.SetCoordinateSystemToDisplay()
            coordinate.SetValue(float(click_pos[0]), float(click_pos[1]), 0.0)

            # 获取近平面上的世界坐标点
            world_point_near = np.array(coordinate.GetComputedWorldValue(renderer))

            # 获取远平面上的点
            coordinate.SetValue(float(click_pos[0]), float(click_pos[1]), 1.0)
            world_point_far = np.array(coordinate.GetComputedWorldValue(renderer))

            # 计算射线方向（从近平面指向远平面）
            ray_direction = world_point_far - world_point_near
            ray_length = np.linalg.norm(ray_direction)
            if ray_length < 1e-10:
                return
            ray_direction = ray_direction / ray_length

            # 使用trimesh进行射线-网格相交测试
            # 从相机位置发射射线
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=[camera_pos],
                ray_directions=[ray_direction]
            )

            if len(locations) == 0:
                # 没有交点，尝试用近平面点作为射线起点
                locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                    ray_origins=[world_point_near],
                    ray_directions=[ray_direction]
                )

            if len(locations) == 0:
                return  # 没有拾取到任何点

            # 找到离相机最近的交点
            distances_to_camera = np.linalg.norm(locations - camera_pos, axis=1)
            nearest_idx = np.argmin(distances_to_camera)

            picked_point = locations[nearest_idx]
            face_idx = int(index_tri[nearest_idx])

            # 处理拾取的点
            self._process_picked_point_with_face(picked_point, face_idx)

        except Exception as e:
            print(f"拾取错误: {e}")
            import traceback
            traceback.print_exc()

    def _process_picked_point_with_face(self, picked_point: np.ndarray, face_idx: int):
        """处理已经确定面索引的拾取点"""
        if self.mesh is None:
            return

        try:
            # 首先检查是否点击了IR点
            if self._check_ir_point_click(picked_point):
                return

            # 如果没有启用添加模式，不添加新点
            if not self.picking_enabled:
                return

            # 验证面索引
            if face_idx < 0 or face_idx >= len(self.mesh.faces):
                return

            # 发射信号
            self.picker_callback.point_picked.emit(
                picked_point.astype(np.float64),
                face_idx
            )
        except Exception:
            pass

    def _on_cell_pick(self, cell):
        """网格面拾取回调 - 备用"""
        if cell is None:
            return
        try:
            if hasattr(cell, 'center'):
                point = np.array(cell.center)
            elif hasattr(cell, 'points') and len(cell.points) > 0:
                point = np.mean(np.array(cell.points), axis=0)
            elif isinstance(cell, np.ndarray):
                point = cell
            else:
                return
            self._process_picked_point(point)
        except Exception as e:
            print(f"cell pick 处理错误: {e}")

    def _on_surface_pick(self, point, picker=None):
        """表面拾取回调"""
        if point is None:
            return
        self._process_picked_point(np.array(point))

    def _on_point_pick(self, point):
        """点拾取回调"""
        if point is None:
            return
        self._process_picked_point(np.array(point))

    def _process_picked_point(self, picked_point: np.ndarray):
        """统一处理拾取的点"""
        if self.mesh is None:
            return

        try:
            # 首先检查是否点击了IR点
            if self._check_ir_point_click(picked_point):
                return

            # 如果没有启用添加模式，不添加新点
            if not self.picking_enabled:
                return

            # 使用trimesh找到曲面上最近的点
            closest, distance, face_idx = self.mesh.nearest.on_surface([picked_point])
            closest_point = closest[0]
            face_idx = int(face_idx[0])

            # 可选：使用射线投射优化（找离相机最近的交点）
            camera_pos = np.array(self.plotter.camera_position[0])
            ray_direction = picked_point - camera_pos
            ray_length = np.linalg.norm(ray_direction)

            if ray_length > 1e-10:
                ray_direction = ray_direction / ray_length
                try:
                    locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                        ray_origins=[camera_pos],
                        ray_directions=[ray_direction]
                    )
                    if len(locations) > 0:
                        # 找离相机最近的交点
                        distances_to_camera = np.linalg.norm(locations - camera_pos, axis=1)
                        nearest_idx = np.argmin(distances_to_camera)
                        closest_point = locations[nearest_idx]
                        face_idx = int(index_tri[nearest_idx])
                except:
                    pass  # 射线投射失败时使用最近点结果

            if face_idx < 0 or face_idx >= len(self.mesh.faces):
                return

            self.picker_callback.point_picked.emit(
                closest_point.astype(np.float64),
                face_idx
            )
        except Exception as e:
            # 静默处理拾取错误，避免频繁输出
            pass

    def _check_ir_point_click(self, picked_point: np.ndarray) -> bool:
        """检查是否点击了IR点，如果是则选中它"""
        if not self.ir_point_positions:
            return False

        # 计算点击位置到各IR点的距离
        min_dist = float('inf')
        closest_id = None

        # 根据模型大小确定选择阈值
        if self.mesh is not None:
            bounds = self.mesh.bounds
            size = np.linalg.norm(bounds[1] - bounds[0])
            threshold = size * 0.03  # 3% 的模型尺寸
        else:
            threshold = 1.0

        for point_id, pos in self.ir_point_positions.items():
            dist = np.linalg.norm(pos - picked_point)
            if dist < min_dist:
                min_dist = dist
                closest_id = point_id

        if closest_id and min_dist < threshold:
            self.select_ir_point(closest_id)
            return True

        return False

    def select_ir_point(self, point_id: str):
        """选中IR点并高亮显示"""
        # 取消之前的选中
        if self.selected_ir_point_id and self.selected_ir_point_id in self.ir_point_positions:
            old_id = self.selected_ir_point_id
            is_center = self.ir_point_is_center.get(old_id, False)
            self._update_ir_point_color(old_id, is_center, selected=False)

        # 选中新的点
        self.selected_ir_point_id = point_id
        if point_id in self.ir_point_positions:
            is_center = self.ir_point_is_center.get(point_id, False)
            self._update_ir_point_color(point_id, is_center, selected=True)

        # 发送选中信号
        self.picker_callback.ir_point_selected.emit(point_id)

    def _update_ir_point_color(self, point_id: str, is_center: bool, selected: bool):
        """更新IR点颜色"""
        if point_id not in self.ir_point_positions:
            return

        position = self.ir_point_positions[point_id]

        # 移除旧的
        if point_id in self.ir_point_actors:
            try:
                self.plotter.remove_actor(self.ir_point_actors[point_id])
            except:
                pass

        # 计算球体半径
        if self.mesh is not None:
            bounds = self.mesh.bounds
            size = np.linalg.norm(bounds[1] - bounds[0])
            radius = size * 0.012 if selected else size * 0.01
        else:
            radius = 0.6 if selected else 0.5

        # 创建球体
        sphere = pv.Sphere(radius=radius, center=position)

        # 选中时使用黄色高亮
        if selected:
            color = '#ffff00'  # 黄色高亮
        elif is_center:
            color = '#e74c3c'  # 红色 - 中心点
        else:
            color = '#2ecc71'  # 绿色 - 普通点

        actor = self.plotter.add_mesh(
            sphere,
            name=f'ir_point_{point_id}',
            color=color,
            opacity=1.0,
            pickable=False
        )

        self.ir_point_actors[point_id] = actor

    def deselect_all(self):
        """取消所有选中"""
        if self.selected_ir_point_id:
            old_id = self.selected_ir_point_id
            is_center = self.ir_point_is_center.get(old_id, False)
            self._update_ir_point_color(old_id, is_center, selected=False)
            self.selected_ir_point_id = None

    def load_mesh(self, mesh: trimesh.Trimesh):
        """加载网格模型"""
        self.mesh = mesh

        # 清除旧的
        self.clear_all()

        # 转换为PyVista格式
        faces = np.hstack([
            np.full((len(mesh.faces), 1), 3),
            mesh.faces
        ]).flatten()
        self.pv_mesh = pv.PolyData(mesh.vertices, faces)

        # 计算法向量用于光照
        self.pv_mesh.compute_normals(inplace=True)

        # 添加到场景 - 不透明实体，带光照
        self.plotter.add_mesh(
            self.pv_mesh,
            name='main_mesh',
            color='#4a69bd',
            show_edges=False,
            opacity=1.0,  # 完全不透明
            smooth_shading=True,
            specular=0.3,  # 降低镜面反射
            specular_power=10,
            ambient=0.4,  # 提高环境光，减少阴影对比
            diffuse=0.6,  # 漫反射
            pickable=True
        )

        # 计算模型边界用于设置光源位置
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        size = np.linalg.norm(bounds[1] - bounds[0])

        # 移除默认光源
        self.plotter.remove_all_lights()

        # 添加多个固定在世界空间的光源（不随相机移动）
        # 主光源 - 从右上前方
        light1 = pv.Light(
            position=(center[0] + size, center[1] + size, center[2] + size * 1.5),
            focal_point=center,
            color='white',
            intensity=0.7
        )
        light1.positional = False  # 方向光，不随相机移动

        # 补光 - 从左侧
        light2 = pv.Light(
            position=(center[0] - size * 1.5, center[1], center[2] + size * 0.5),
            focal_point=center,
            color='white',
            intensity=0.4
        )
        light2.positional = False

        # 底部补光 - 减少底部阴影
        light3 = pv.Light(
            position=(center[0], center[1] - size, center[2] - size * 0.5),
            focal_point=center,
            color='white',
            intensity=0.3
        )
        light3.positional = False

        self.plotter.add_light(light1)
        self.plotter.add_light(light2)
        self.plotter.add_light(light3)

        # 重置相机
        self.plotter.reset_camera()

        # 加载后启用拾取
        self._enable_picking()

    def set_picking_enabled(self, enabled: bool):
        """设置是否启用拾取"""
        self.picking_enabled = enabled

    def add_ir_point(
        self,
        point_id: str,
        position: np.ndarray,
        is_center: bool = False,
        name: str = ""
    ):
        """添加IR点可视化"""
        # 保存位置和状态
        self.ir_point_positions[point_id] = position.copy()
        self.ir_point_is_center[point_id] = is_center

        # 移除旧的
        if point_id in self.ir_point_actors:
            try:
                self.plotter.remove_actor(self.ir_point_actors[point_id])
            except:
                pass

        # 根据模型大小计算球体半径
        if self.mesh is not None:
            bounds = self.mesh.bounds
            size = np.linalg.norm(bounds[1] - bounds[0])
            radius = size * 0.01
        else:
            radius = 0.5

        # 创建球体
        sphere = pv.Sphere(radius=radius, center=position)

        color = '#e74c3c' if is_center else '#2ecc71'
        actor = self.plotter.add_mesh(
            sphere,
            name=f'ir_point_{point_id}',
            color=color,
            opacity=1.0,
            pickable=False
        )

        self.ir_point_actors[point_id] = actor

        # 添加标签
        if name:
            try:
                self.plotter.add_point_labels(
                    [position],
                    [name],
                    name=f'label_{point_id}',
                    font_size=12,
                    text_color='white',
                    point_size=0,
                    shape_opacity=0
                )
            except:
                pass

    def update_ir_point(self, point_id: str, position: np.ndarray = None,
                        is_center: bool = None, name: str = None):
        """更新IR点"""
        if point_id not in self.ir_point_positions:
            return

        if position is not None:
            self.ir_point_positions[point_id] = position.copy()
        if is_center is not None:
            self.ir_point_is_center[point_id] = is_center

        # 重新绘制
        pos = self.ir_point_positions[point_id]
        is_ctr = self.ir_point_is_center[point_id]
        selected = (self.selected_ir_point_id == point_id)

        self._update_ir_point_color(point_id, is_ctr, selected)

    def remove_ir_point(self, point_id: str):
        """移除IR点"""
        if point_id in self.ir_point_actors:
            try:
                self.plotter.remove_actor(self.ir_point_actors[point_id])
            except:
                pass
            del self.ir_point_actors[point_id]

        if point_id in self.ir_point_positions:
            del self.ir_point_positions[point_id]
        if point_id in self.ir_point_is_center:
            del self.ir_point_is_center[point_id]

        if self.selected_ir_point_id == point_id:
            self.selected_ir_point_id = None

        # 移除标签
        try:
            self.plotter.remove_actor(f'label_{point_id}')
        except:
            pass

    def add_path(
        self,
        path_points: np.ndarray,
        path_id: str = "",
        color: str = '#f39c12',
        width: float = 3.0
    ):
        """添加路径可视化"""
        if len(path_points) < 2:
            return

        # 创建线段
        lines = pv.lines_from_points(path_points)

        actor = self.plotter.add_mesh(
            lines,
            name=f'path_{path_id}' if path_id else None,
            color=color,
            line_width=width,
            pickable=False
        )

        self.path_actors.append(actor)

    def clear_paths(self):
        """清除所有路径"""
        for actor in self.path_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.path_actors.clear()

    def add_groove_preview(
        self,
        groove_mesh: trimesh.Trimesh,
        groove_id: str = ""
    ):
        """添加凹槽预览"""
        if groove_mesh is None or len(groove_mesh.vertices) == 0:
            print(f"凹槽 {groove_id} 无效，跳过")
            return

        try:
            # 确保网格有效
            groove_mesh.fix_normals()

            # 使用简单的方式创建 PolyData
            vertices = np.asarray(groove_mesh.vertices, dtype=np.float32)
            faces_array = groove_mesh.faces

            # 构建 PyVista 面数组
            n_faces = len(faces_array)
            pv_faces = np.zeros((n_faces, 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces_array
            pv_faces = pv_faces.flatten()

            pv_groove = pv.PolyData(vertices, pv_faces)

            # 使用简单渲染设置，避免shader问题
            actor = self.plotter.add_mesh(
                pv_groove,
                name=f'groove_{groove_id}' if groove_id else None,
                color='#e74c3c',
                opacity=0.9,
                show_edges=True,
                edge_color='#8b0000',
                line_width=1.0,
                smooth_shading=False,  # 使用flat shading避免shader问题
                lighting=True,
                pickable=False
            )

            self.groove_actors.append(actor)
            print(f"凹槽 {groove_id} 已添加: {len(groove_mesh.vertices)} 顶点, {len(groove_mesh.faces)} 面")

            # 强制刷新视图
            self.plotter.render()

        except Exception as e:
            print(f"凹槽 {groove_id} 添加失败: {e}")
            import traceback
            traceback.print_exc()

    def clear_grooves(self):
        """清除凹槽预览"""
        for actor in self.groove_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.groove_actors.clear()

    def clear_all(self):
        """清除所有可视化元素"""
        self.plotter.clear()
        self.ir_point_actors.clear()
        self.ir_point_positions.clear()
        self.ir_point_is_center.clear()
        self.path_actors.clear()
        self.groove_actors.clear()
        self.selected_ir_point_id = None

        # 重新添加坐标轴
        self.plotter.add_axes()

    def update_mesh_with_grooves(self, mesh_with_grooves: trimesh.Trimesh):
        """更新显示带凹槽的网格"""
        faces = np.hstack([
            np.full((len(mesh_with_grooves.faces), 1), 3),
            mesh_with_grooves.faces
        ]).flatten()
        pv_mesh = pv.PolyData(mesh_with_grooves.vertices, faces)
        pv_mesh.compute_normals(inplace=True)

        self.plotter.add_mesh(
            pv_mesh,
            name='mesh_with_grooves',
            color='#4a69bd',
            show_edges=False,
            opacity=1.0,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            ambient=0.3,
            diffuse=0.7
        )

    def screenshot(self, filename: str):
        """保存截图"""
        self.plotter.screenshot(filename)

    def add_coordinate_axes(
        self,
        origin: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        length: float = 3.0,
        shaft_scale: float = 1.0,
        tip_scale: float = 1.0
    ):
        """添加自定义坐标轴箭头

        Args:
            origin: 原点位置
            x_axis: X轴方向向量（单位向量）
            y_axis: Y轴方向向量（单位向量）
            z_axis: Z轴方向向量（单位向量）
            length: 箭头长度
            shaft_scale: 箭头杆粗细缩放 (1.0 = 默认)
            tip_scale: 箭头头部大小缩放 (1.0 = 默认)
        """
        # 先移除旧的坐标轴
        self.remove_coordinate_axes()

        # 计算箭头参数（使用固定基础值，不随长度变化）
        # 基础值保持轴的粗细恒定，只有shaft_scale和tip_scale影响粗细
        base_shaft_radius = 0.05 * shaft_scale  # 固定的轴半径
        base_tip_radius = 0.12 * tip_scale      # 固定的箭头头部半径
        base_tip_length = 0.25 * tip_scale      # 固定的箭头尖端长度

        shaft_radius = base_shaft_radius
        tip_radius = base_tip_radius
        tip_length = base_tip_length

        # X轴 - 红色
        x_end = origin + x_axis * length
        x_arrow = pv.Arrow(
            start=origin,
            direction=x_axis,
            scale=length,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            tip_length=tip_length
        )
        self._coord_axis_actors['x'] = self.plotter.add_mesh(
            x_arrow,
            name='coord_axis_x',
            color='#ff0000',
            opacity=0.9,
            pickable=False
        )

        # Y轴 - 绿色
        y_arrow = pv.Arrow(
            start=origin,
            direction=y_axis,
            scale=length,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            tip_length=tip_length
        )
        self._coord_axis_actors['y'] = self.plotter.add_mesh(
            y_arrow,
            name='coord_axis_y',
            color='#00ff00',
            opacity=0.9,
            pickable=False
        )

        # Z轴 - 蓝色
        z_arrow = pv.Arrow(
            start=origin,
            direction=z_axis,
            scale=length,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            tip_length=tip_length
        )
        self._coord_axis_actors['z'] = self.plotter.add_mesh(
            z_arrow,
            name='coord_axis_z',
            color='#0000ff',
            opacity=0.9,
            pickable=False
        )

        # 添加轴标签
        label_offset = length * 1.1
        try:
            self.plotter.add_point_labels(
                [origin + x_axis * label_offset],
                ['X'],
                name='coord_label_x',
                font_size=14,
                text_color='#ff0000',
                point_size=0,
                shape_opacity=0,
                bold=True
            )
            self.plotter.add_point_labels(
                [origin + y_axis * label_offset],
                ['Y'],
                name='coord_label_y',
                font_size=14,
                text_color='#00ff00',
                point_size=0,
                shape_opacity=0,
                bold=True
            )
            self.plotter.add_point_labels(
                [origin + z_axis * label_offset],
                ['Z'],
                name='coord_label_z',
                font_size=14,
                text_color='#0000ff',
                point_size=0,
                shape_opacity=0,
                bold=True
            )
        except:
            pass  # 标签可能因为字体问题失败

        # 添加原点标记
        origin_sphere = pv.Sphere(radius=length * 0.04, center=origin)
        self._coord_axis_actors['origin'] = self.plotter.add_mesh(
            origin_sphere,
            name='coord_origin',
            color='#ffffff',
            opacity=1.0,
            pickable=False
        )

        self._coord_axes_visible = True
        self.plotter.render()

    def update_coordinate_axes(
        self,
        origin: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        length: float = 3.0,
        shaft_scale: float = 1.0,
        tip_scale: float = 1.0
    ):
        """更新坐标轴（重新绘制）"""
        if self._coord_axes_visible:
            self.add_coordinate_axes(origin, x_axis, y_axis, z_axis, length, shaft_scale, tip_scale)

    def remove_coordinate_axes(self):
        """移除坐标轴"""
        for key, actor in list(self._coord_axis_actors.items()):
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self._coord_axis_actors.clear()

        # 移除标签
        for label_name in ['coord_label_x', 'coord_label_y', 'coord_label_z', 'coord_origin']:
            try:
                self.plotter.remove_actor(label_name)
            except:
                pass

        self._coord_axes_visible = False

    def set_coordinate_axes_visible(self, visible: bool):
        """设置坐标轴可见性"""
        if visible and not self._coord_axes_visible:
            # 需要重新添加
            pass
        elif not visible and self._coord_axes_visible:
            self.remove_coordinate_axes()

    def highlight_origin_point(self, point_id: str, position: np.ndarray):
        """高亮显示原点

        Args:
            point_id: 点ID
            position: 点位置
        """
        # 移除旧的原点高亮
        self.remove_origin_highlight()

        if self.mesh is not None:
            bounds = self.mesh.bounds
            size = np.linalg.norm(bounds[1] - bounds[0])
            radius = size * 0.015
        else:
            radius = 0.8

        # 创建高亮环
        ring = pv.Disc(center=position, inner=radius * 0.8, outer=radius * 1.2, normal=(0, 0, 1))
        self._origin_highlight_actor = self.plotter.add_mesh(
            ring,
            name='origin_highlight',
            color='#ffd700',  # 金色
            opacity=0.8,
            pickable=False
        )
        self.plotter.render()

    def remove_origin_highlight(self):
        """移除原点高亮"""
        if self._origin_highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._origin_highlight_actor)
            except:
                pass
            self._origin_highlight_actor = None
