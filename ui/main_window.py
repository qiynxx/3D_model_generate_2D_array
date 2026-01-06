"""主窗口"""
import sys
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QStatusBar,
    QTabWidget, QLabel, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from typing import Optional
import trimesh

from .viewer_3d import Viewer3D
from .view_2d import View2D
from .point_panel import PointPanel
from .param_panel import ParamPanel
from .coord_panel import CoordPanel
from core.mesh_loader import MeshLoader
from core.geodesic import PathManager, IRPoint
from core.groove_gen import GrooveGenerator
from core.conformal_map import ConformalMapper, FlattenResult
from core.path_flatten import (
    flatten_paths_preserve_geometry, PathFlattenResult,
    generate_fpc_layout, FPCLayoutResult,
    validate_flatten_accuracy, FlattenValidationResult,
    FPCFittingAnimator
)
from core.exporter import Exporter, BatchExporter, ProjectSnapshot, FolderExporter


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.mesh: Optional[trimesh.Trimesh] = None
        self.mesh_path: str = ""
        self.path_manager: Optional[PathManager] = None
        self.groove_generator: Optional[GrooveGenerator] = None
        self.conformal_mapper: Optional[ConformalMapper] = None
        self.flatten_result: Optional[FlattenResult] = None
        self.path_flatten_result: Optional[PathFlattenResult] = None
        self.fpc_layout_result: Optional[FPCLayoutResult] = None
        self.mesh_with_grooves: Optional[trimesh.Trimesh] = None

        # 动画相关
        self.fpc_animator: Optional[FPCFittingAnimator] = None
        self.animation_timer = None
        self.animation_progress = 0.0
        self.animation_actor = None

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        """设置UI"""
        self.setWindowTitle("IR阵列定位辅助生成工具")
        self.setMinimumSize(1200, 800)

        # 中心部件
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 3D视图区域
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)

        # 3D/2D选项卡
        self.view_tabs = QTabWidget()
        view_layout.addWidget(self.view_tabs)

        # 3D视图
        self.viewer_3d = Viewer3D()
        self.view_tabs.addTab(self.viewer_3d, "3D视图")

        # 2D展开视图
        self.view_2d = View2D()
        self.view_tabs.addTab(self.view_2d, "2D展开")

        splitter.addWidget(view_container)

        # 右侧面板（选项卡式）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 右侧选项卡
        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs)

        # 点位管理标签页
        point_tab = QWidget()
        point_tab_layout = QVBoxLayout(point_tab)
        self.point_panel = PointPanel()
        point_tab_layout.addWidget(self.point_panel)
        self.right_tabs.addTab(point_tab, "点位管理")

        # 坐标系统标签页
        coord_tab = QWidget()
        coord_tab_layout = QVBoxLayout(coord_tab)
        self.coord_panel = CoordPanel()
        coord_tab_layout.addWidget(self.coord_panel)
        self.right_tabs.addTab(coord_tab, "坐标系统")

        # 参数面板
        self.param_panel = ParamPanel()
        right_layout.addWidget(self.param_panel)

        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([800, 400])

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请加载3D模型")

    def _setup_menu(self):
        """设置菜单"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_action = QAction("打开模型(&O)...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_model)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # 项目快照
        save_snapshot = QAction("保存项目快照(&S)...", self)
        save_snapshot.setShortcut(QKeySequence.Save)
        save_snapshot.triggered.connect(self._on_save_snapshot)
        file_menu.addAction(save_snapshot)

        load_snapshot = QAction("加载项目快照(&L)...", self)
        load_snapshot.setShortcut("Ctrl+Shift+O")
        load_snapshot.triggered.connect(self._on_load_snapshot)
        file_menu.addAction(load_snapshot)

        file_menu.addSeparator()

        # 导出
        export_folder = QAction("导出到文件夹(&E)...", self)
        export_folder.setShortcut("Ctrl+E")
        export_folder.triggered.connect(self._on_export_to_folder)
        file_menu.addAction(export_folder)

        export_coords = QAction("导出坐标文件(&C)...", self)
        export_coords.triggered.connect(self._on_export_coordinates)
        file_menu.addAction(export_coords)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")

        clear_points = QAction("清除所有点", self)
        clear_points.triggered.connect(self._on_clear_points)
        edit_menu.addAction(clear_points)

        clear_paths = QAction("清除路径", self)
        clear_paths.triggered.connect(self._on_clear_paths)
        edit_menu.addAction(clear_paths)

        edit_menu.addSeparator()

        set_origin = QAction("设选中点为坐标原点", self)
        set_origin.triggered.connect(self._on_set_selected_as_origin)
        edit_menu.addAction(set_origin)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """连接信号"""
        # 3D视图拾取
        self.viewer_3d.picker_callback.point_picked.connect(self._on_point_picked)
        self.viewer_3d.picker_callback.ir_point_selected.connect(self._on_ir_point_selected_3d)

        # 点位面板
        self.point_panel.picking_toggled.connect(self._on_picking_toggled)
        self.point_panel.point_deleted.connect(self._on_point_deleted)
        self.point_panel.center_changed.connect(self._on_center_changed)
        self.point_panel.paths_requested.connect(self._on_generate_paths)
        self.point_panel.point_selected.connect(self._on_ir_point_selected_panel)
        self.point_panel.pad_size_changed.connect(self._on_pad_size_changed)

        # 坐标系统面板
        self.coord_panel.origin_changed.connect(self._on_origin_changed)
        self.coord_panel.axis_changed.connect(self._on_axis_changed)
        self.coord_panel.show_axes_changed.connect(self._on_show_axes_changed)

        # 参数面板
        self.param_panel.generate_grooves_clicked.connect(self._on_generate_grooves)
        self.param_panel.flatten_clicked.connect(self._on_flatten)
        self.param_panel.generate_fpc_layout_clicked.connect(self._on_generate_fpc_layout)
        self.param_panel.validate_flatten_clicked.connect(self._on_validate_flatten)
        self.param_panel.play_fitting_animation_clicked.connect(self._on_play_fitting_animation)
        self.param_panel.export_all_clicked.connect(self._on_export_to_folder)
        self.param_panel.export_single_clicked.connect(self._on_export_single)

    def _on_open_model(self):
        """打开模型"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "打开3D模型",
            "",
            "3D模型文件 (*.stl *.3mf *.obj *.ply);;所有文件 (*)"
        )

        if not filepath:
            return

        self.mesh = MeshLoader.load(filepath)
        if self.mesh is None:
            QMessageBox.critical(self, "错误", "无法加载模型文件")
            return

        self.mesh_path = filepath

        # 初始化管理器
        self.path_manager = PathManager(self.mesh)
        self.groove_generator = GrooveGenerator(self.mesh)
        self.conformal_mapper = ConformalMapper(self.mesh)

        # 显示模型
        self.viewer_3d.load_mesh(self.mesh)

        # 清除旧的点位
        self.point_panel.clear_all()

        # 更新状态 - 显示模型尺寸
        info = MeshLoader.get_mesh_info(self.mesh)
        bounds = info['bounds']
        size_x = bounds[1][0] - bounds[0][0]
        size_y = bounds[1][1] - bounds[0][1]
        size_z = bounds[1][2] - bounds[0][2]
        self.status_bar.showMessage(
            f"已加载: {Path(filepath).name} | "
            f"尺寸: {size_x:.1f}×{size_y:.1f}×{size_z:.1f} mm | "
            f"顶点: {info['vertices']} | 面: {info['faces']}"
        )

    def _on_picking_toggled(self, enabled: bool):
        """拾取模式切换"""
        self.viewer_3d.set_picking_enabled(enabled)

    def _on_ir_point_selected_3d(self, point_id: str):
        """3D视图中选中IR点"""
        # 同步到点位面板
        self.point_panel.select_point_by_id(point_id)
        self.status_bar.showMessage(f"选中点: {self.point_panel.points.get(point_id, {}).get('name', point_id)}")

    def _on_ir_point_selected_panel(self, point_id: str):
        """点位面板中选中IR点"""
        # 同步到3D视图
        self.viewer_3d.select_ir_point(point_id)

    def _on_point_picked(self, position: np.ndarray, face_index: int):
        """点被拾取"""
        if self.path_manager is None:
            return

        # 添加点
        point = self.path_manager.add_point(
            position=position,
            face_index=face_index
        )

        # 更新UI
        self.point_panel.add_point(
            point_id=point.id,
            position=position,
            name=point.name
        )

        # 更新3D显示
        self.viewer_3d.add_ir_point(
            point_id=point.id,
            position=position,
            is_center=point.is_center,
            name=point.name
        )

        # 同步到坐标面板
        self._sync_points_to_coord_panel()

        self.status_bar.showMessage(f"添加点: {point.name}")

    def _on_point_deleted(self, point_id: str):
        """删除点"""
        if self.path_manager:
            self.path_manager.remove_point(point_id)

        self.point_panel.remove_point(point_id)
        self.viewer_3d.remove_ir_point(point_id)

    def _on_center_changed(self, point_id: str):
        """中心点变更"""
        if self.path_manager:
            self.path_manager.set_center_point(point_id)

        # 更新所有点的显示
        for pid, info in self.point_panel.points.items():
            self.viewer_3d.add_ir_point(
                point_id=pid,
                position=info['position'],
                is_center=pid == point_id,
                name=info['name']
            )

    def _on_pad_size_changed(self, point_id: str, length: float, width: float):
        """焊盘尺寸变更"""
        if self.path_manager and point_id in self.path_manager.ir_points:
            self.path_manager.ir_points[point_id].pad_length = length
            self.path_manager.ir_points[point_id].pad_width = width
            self.status_bar.showMessage(f"点 {point_id[:8]} 焊盘尺寸: {length}x{width}mm")

    def _on_generate_paths(self):
        """生成路径"""
        if self.path_manager is None:
            return

        self.status_bar.showMessage("正在计算路径...")
        print(f"\n=== 开始生成路径 ===")
        print(f"IR点数量: {len(self.path_manager.ir_points)}")
        print(f"中心点ID: {self.path_manager.center_point_id}")

        # 清除旧路径
        self.viewer_3d.clear_paths()

        # 获取要排除的点（坐标原点）
        exclude_ids = []
        origin_point_id = self.coord_panel.origin_point_id
        if origin_point_id:
            exclude_ids.append(origin_point_id)
            print(f"排除坐标原点: {origin_point_id}")

        # 计算路径（启用平滑，排除原点）
        paths = self.path_manager.compute_all_paths(smooth=True, exclude_point_ids=exclude_ids)

        print(f"生成路径数量: {len(paths)}")
        print(f"平滑路径数量: {len(self.path_manager.smooth_paths)}")

        # 显示平滑后的路径
        path_count = 0
        for point_id in paths.keys():
            smooth_path = self.path_manager.get_smooth_path(point_id)
            if smooth_path is not None and len(smooth_path) > 0:
                self.viewer_3d.add_path(
                    path_points=smooth_path,
                    path_id=point_id
                )
                path_count += 1
                print(f"添加路径 {point_id}: {len(smooth_path)} 点")

        print(f"=== 路径生成完成: {path_count} 条 ===\n")
        self.status_bar.showMessage(f"已生成 {path_count} 条平滑路径")

    def _on_generate_grooves(self):
        """生成凹槽"""
        if self.path_manager is None or self.groove_generator is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        print(f"\n=== 开始生成凹槽 ===")
        print(f"smooth_paths 数量: {len(self.path_manager.smooth_paths)}")
        print(f"paths 数量: {len(self.path_manager.paths)}")

        if not self.path_manager.smooth_paths:
            QMessageBox.warning(self, "警告", "请先生成连接路径")
            return

        self.status_bar.showMessage("正在生成凹槽...")

        # 获取参数
        params = self.param_panel.get_groove_params()
        print(f"凹槽参数: 宽度={params.width}, 深度={params.depth}, 曲面贴合={params.conform_to_surface}")
        print(f"方形焊盘: 启用={params.pad_enabled}, 长={params.pad_length}, 宽={params.pad_width}")
        print(f"向外延伸: {params.extension_height}mm")

        self.groove_generator.set_params(
            width=params.width,
            depth=params.depth,
            auto_width=params.auto_width,
            min_width=params.min_width,
            max_width=params.max_width,
            conform_to_surface=params.conform_to_surface,
            extension_height=params.extension_height,
            pad_enabled=params.pad_enabled,
            pad_length=params.pad_length,
            pad_width=params.pad_width
        )

        # 清除旧凹槽
        self.viewer_3d.clear_grooves()

        # 收集所有凹槽网格
        all_grooves = []
        groove_count = 0
        pad_count = 0

        for point_id in self.path_manager.smooth_paths.keys():
            # 获取IR点信息
            ir_point_data = self.path_manager.ir_points.get(point_id)
            if ir_point_data is None:
                print(f"点 {point_id}: 无法获取IR点信息，跳过")
                continue

            # 使用平滑路径
            smooth_path = self.path_manager.get_smooth_path(point_id)
            if smooth_path is None:
                print(f"路径 {point_id}: smooth_path 为 None，跳过")
                continue

            if len(smooth_path) < 2:
                print(f"路径 {point_id}: 点数不足 ({len(smooth_path)})，跳过")
                continue

            # 计算平滑路径的法向量
            normals = self._compute_path_normals(smooth_path)
            print(f"路径 {point_id}: {len(smooth_path)} 点, 法向量 {len(normals)} 个")

            # IR点位置和法向量
            ir_position = ir_point_data.position
            ir_normal = normals[0] if len(normals) > 0 else np.array([0, 0, 1])

            # 生成完整凹槽（方形焊盘 + 路径）
            # 使用每个点的单独焊盘尺寸
            point_pad_length = ir_point_data.pad_length
            point_pad_width = ir_point_data.pad_width

            pad_mesh, path_mesh = self.groove_generator.generate_complete_groove(
                ir_position, ir_normal,
                smooth_path, normals,
                pad_length=point_pad_length,
                pad_width=point_pad_width
            )

            # 添加方形焊盘
            if pad_mesh is not None and len(pad_mesh.vertices) > 0:
                print(f"方形焊盘 {point_id}: {len(pad_mesh.vertices)} 顶点")
                self.viewer_3d.add_groove_preview(pad_mesh, f"{point_id}_pad")
                all_grooves.append(pad_mesh)
                pad_count += 1

            # 添加路径凹槽
            if path_mesh is not None and len(path_mesh.vertices) > 0:
                print(f"路径凹槽 {point_id}: {len(path_mesh.vertices)} 顶点")
                self.viewer_3d.add_groove_preview(path_mesh, f"{point_id}_path")
                all_grooves.append(path_mesh)
                groove_count += 1

        # 生成中心点焊盘凹槽
        if params.pad_enabled and self.path_manager.center_point_id:
            center_point = self.path_manager.ir_points.get(self.path_manager.center_point_id)
            if center_point:
                print(f"\n生成中心点焊盘凹槽...")
                center_position = center_point.position
                # 获取中心点法向量
                distances = np.linalg.norm(self.mesh.vertices - center_position, axis=1)
                nearest_idx = np.argmin(distances)
                center_normal = self.mesh.vertex_normals[nearest_idx]

                # 确定中心点方向（使用第一条路径的方向）
                center_direction = np.array([1.0, 0.0, 0.0])
                if self.path_manager.smooth_paths:
                    first_path = list(self.path_manager.smooth_paths.values())[0]
                    if len(first_path) >= 2:
                        # 从中心点向外的方向
                        center_direction = first_path[-2] - first_path[-1]

                # 获取中心点焊盘尺寸
                center_pad_length = center_point.pad_length
                center_pad_width = center_point.pad_width

                center_pad_mesh = self.groove_generator.generate_pad_groove(
                    center_position, center_normal, center_direction,
                    center_pad_length, center_pad_width, params.depth
                )

                if center_pad_mesh is not None and len(center_pad_mesh.vertices) > 0:
                    print(f"中心点焊盘: {len(center_pad_mesh.vertices)} 顶点")
                    self.viewer_3d.add_groove_preview(center_pad_mesh, "center_pad")
                    all_grooves.append(center_pad_mesh)
                    pad_count += 1

        print(f"=== 凹槽生成完成: {pad_count} 个方形焊盘, {groove_count} 个路径凹槽 ===")

        total_count = pad_count + groove_count
        if total_count > 0:
            self.status_bar.showMessage(f"已生成 {pad_count} 个方形焊盘, {groove_count} 个路径凹槽")

            # 尝试将凹槽应用到模型（布尔运算）
            try:
                combined_groove = all_grooves[0]
                for groove in all_grooves[1:]:
                    combined_groove = trimesh.util.concatenate([combined_groove, groove])

                # 尝试不同的布尔运算引擎
                boolean_success = False

                # 尝试 Blender 引擎
                try:
                    self.mesh_with_grooves = self.mesh.difference(combined_groove, engine='blender')
                    print("凹槽已应用到模型 (Blender引擎)")
                    boolean_success = True
                except Exception as e1:
                    print(f"Blender引擎不可用: {e1}")

                # 尝试 Manifold 引擎
                if not boolean_success:
                    try:
                        self.mesh_with_grooves = self.mesh.difference(combined_groove, engine='manifold')
                        print("凹槽已应用到模型 (Manifold引擎)")
                        boolean_success = True
                    except Exception as e2:
                        print(f"Manifold引擎不可用: {e2}")

                if not boolean_success:
                    print("\n提示: 布尔运算需要安装 blender 或 manifold3d")
                    print("  安装 manifold3d: pip install manifold3d")
                    print("  或安装 Blender 并确保可从命令行调用")
                    print("\n凹槽预览已显示，但无法应用到模型导出")
                    self.mesh_with_grooves = None

            except Exception as e:
                print(f"凹槽处理错误: {e}")
                self.mesh_with_grooves = None
        else:
            self.status_bar.showMessage("没有生成任何凹槽")

    def _compute_path_normals(self, path_points: np.ndarray) -> np.ndarray:
        """
        计算路径点的表面法向量

        使用改进的方法：
        1. 找到路径点所在的最近三角面
        2. 使用该面的法向量
        3. 对法向量进行平滑处理以避免突变
        """
        normals = []

        for point in path_points:
            # 方法1：找最近的面，使用面法向量（更准确）
            closest_point, distance, face_id = self.mesh.nearest.on_surface([point])
            if face_id is not None and len(face_id) > 0:
                face_normal = self.mesh.face_normals[face_id[0]]
                normals.append(face_normal)
            else:
                # 回退：使用最近顶点的法向量
                distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
                nearest_idx = np.argmin(distances)
                normals.append(self.mesh.vertex_normals[nearest_idx])

        normals = np.array(normals)

        # 对法向量进行平滑，避免相邻点法向量差异太大导致凹槽扭曲
        if len(normals) > 3:
            smoothed = normals.copy()
            for _ in range(2):  # 迭代平滑2次
                for i in range(1, len(normals) - 1):
                    # 使用相邻3点的加权平均
                    avg = 0.5 * normals[i] + 0.25 * normals[i-1] + 0.25 * normals[i+1]
                    norm = np.linalg.norm(avg)
                    if norm > 1e-6:
                        smoothed[i] = avg / norm
                normals = smoothed.copy()

        return normals

    def _on_flatten(self):
        """路径展开"""
        if self.path_manager is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if not self.path_manager.smooth_paths:
            QMessageBox.warning(self, "警告", "请先生成连接路径")
            return

        self.status_bar.showMessage("正在展开路径（使用保形映射）...")

        try:
            # 准备路径数据
            paths_3d = {}
            for point_id, smooth_path in self.path_manager.smooth_paths.items():
                if len(smooth_path) > 0:
                    paths_3d[point_id] = smooth_path

            # 准备IR点数据
            ir_points = {}
            center_position = None
            center_normal = None

            for point_id, point in self.path_manager.ir_points.items():
                ir_points[point_id] = (point.position, point.name, point.is_center)
                if point.is_center:
                    center_position = point.position
                    # 获取中心点的表面法向量
                    distances = np.linalg.norm(self.mesh.vertices - center_position, axis=1)
                    nearest_idx = np.argmin(distances)
                    center_normal = self.mesh.vertex_normals[nearest_idx]

            if center_position is None:
                QMessageBox.warning(self, "警告", "未找到中心点")
                return

            # 使用保形映射展开（传入mesh以启用LSCM算法）
            self.path_flatten_result = flatten_paths_preserve_geometry(
                paths_3d, ir_points, center_position, center_normal,
                mesh=self.mesh  # 传入mesh启用保形映射
            )

            # 设置2D视图
            self.view_2d.set_flatten_result(
                self.path_flatten_result.paths_2d,
                self.path_flatten_result.ir_points_2d,
                self.path_flatten_result.center_2d,
                self.path_flatten_result.total_bounds
            )

            # 切换到2D视图
            self.view_tabs.setCurrentIndex(1)

            self.status_bar.showMessage(
                f"路径展开完成（LSCM保形映射） - {len(paths_3d)} 条路径"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"展开失败: {e}")

    def _on_generate_fpc_layout(self):
        """生成FPC图纸布局"""
        if self.path_flatten_result is None:
            QMessageBox.warning(self, "警告", "请先展开路径")
            return

        self.status_bar.showMessage("正在生成FPC图纸...")

        try:
            # 获取FPC参数
            fpc_params = self.param_panel.get_fpc_params()
            # 获取凹槽参数（用于方形焊盘）
            groove_params = self.param_panel.get_groove_params()

            # 获取每个点的焊盘尺寸
            point_pad_sizes = None
            if self.path_manager:
                point_pad_sizes = {}
                for point_id, point in self.path_manager.ir_points.items():
                    point_pad_sizes[point_id] = (point.pad_length, point.pad_width)

            # 生成FPC布局
            self.fpc_layout_result = generate_fpc_layout(
                flatten_result=self.path_flatten_result,
                groove_width=fpc_params.groove_width,
                pad_radius=fpc_params.pad_radius,
                center_pad_radius=fpc_params.center_pad_radius,
                rectangular_pad_enabled=groove_params.pad_enabled,
                rectangular_pad_length=groove_params.pad_length,
                rectangular_pad_width=groove_params.pad_width,
                point_pad_sizes=point_pad_sizes
            )

            # 设置2D视图显示FPC布局
            self.view_2d.set_fpc_layout(
                groove_outlines=self.fpc_layout_result.groove_outlines,
                ir_pads=self.fpc_layout_result.ir_pads,
                center_pad=self.fpc_layout_result.center_pad,
                merged_outline=self.fpc_layout_result.merged_outline,
                bounds=self.fpc_layout_result.total_bounds
            )

            # 切换到2D视图
            self.view_tabs.setCurrentIndex(1)

            pad_type = "方形" if groove_params.pad_enabled else "圆形"
            self.status_bar.showMessage(
                f"FPC图纸生成完成 - 走线宽度: {fpc_params.groove_width}mm, "
                f"{len(self.fpc_layout_result.groove_outlines)} 条凹槽轮廓, "
                f"{pad_type}焊盘"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"生成FPC图纸失败: {e}")

    def _transform_path_to_2d(self, path_3d: np.ndarray) -> np.ndarray:
        """将3D路径转换到2D"""
        if self.flatten_result is None or len(path_3d) == 0:
            return np.array([])

        path_2d = []
        for point in path_3d:
            point_2d = self._transform_point_to_2d(point)
            if point_2d is not None:
                path_2d.append(point_2d)

        return np.array(path_2d) if path_2d else np.array([])

    def _transform_point_to_2d(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """将3D点转换到2D"""
        if self.flatten_result is None:
            return None

        # 找到最近的顶点
        distances = np.linalg.norm(self.mesh.vertices - point_3d, axis=1)
        nearest_idx = np.argmin(distances)

        # 返回对应的UV坐标
        return self.flatten_result.uv_coords[nearest_idx].copy()

    def _on_export_all(self):
        """导出所有文件"""
        export_params = self.param_panel.get_export_params()

        if not export_params.output_dir:
            dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if not dir_path:
                return
            export_params.output_dir = dir_path

        self.status_bar.showMessage("正在导出文件...")

        try:
            exporter = BatchExporter(export_params.output_dir)

            # 从View2D获取已正确转换的2D数据
            paths_2d = self.view_2d.get_paths_for_export()
            ir_points_2d = self.view_2d.get_points_for_export()

            # 如果2D视图没有数据，尝试从3D数据转换
            if not paths_2d and self.flatten_result and self.path_manager:
                for point_id in self.path_manager.smooth_paths.keys():
                    smooth_path = self.path_manager.get_smooth_path(point_id)
                    if smooth_path is not None and len(smooth_path) > 0:
                        path_2d = self._transform_path_to_2d(smooth_path)
                        if len(path_2d) > 0:
                            # 应用缩放因子
                            paths_2d.append(path_2d * self.flatten_result.scale)

            if not ir_points_2d and self.flatten_result and self.path_manager:
                for point_id, point in self.path_manager.ir_points.items():
                    point_2d = self._transform_point_to_2d(point.position)
                    if point_2d is not None:
                        # 应用缩放因子
                        ir_points_2d.append((point_2d * self.flatten_result.scale, point.name))

            # 导出
            ir_points_data = self.path_manager.to_dict() if self.path_manager else {}
            groove_params = {
                'width': self.param_panel.groove_params.width,
                'depth': self.param_panel.groove_params.depth,
            }

            # 获取FPC布局数据（如果有）
            fpc_layout = None
            if self.view_2d.has_fpc_layout():
                fpc_layout = self.view_2d.get_fpc_layout_for_export()

            exported = exporter.export_all(
                mesh=self.mesh,
                mesh_with_grooves=self.mesh_with_grooves,
                paths_2d=paths_2d,
                ir_points_2d=ir_points_2d,
                ir_points_data=ir_points_data,
                groove_params=groove_params,
                project_name="ir_array",
                fpc_layout=fpc_layout
            )

            # 显示结果
            msg = "导出完成:\n\n" + "\n".join(
                f"- {k}: {v}" for k, v in exported.items()
            )
            QMessageBox.information(self, "导出成功", msg)

            self.status_bar.showMessage(f"已导出 {len(exported)} 个文件")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))

    def _on_save_project(self):
        """保存项目"""
        if self.path_manager is None:
            QMessageBox.warning(self, "警告", "没有可保存的项目")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存项目",
            "ir_array_project.json",
            "JSON文件 (*.json)"
        )

        if not filepath:
            return

        try:
            Exporter.export_project(
                filepath=filepath,
                mesh_path=self.mesh_path,
                ir_points_data=self.path_manager.to_dict(),
                groove_params={
                    'width': self.param_panel.groove_params.width,
                    'depth': self.param_panel.groove_params.depth,
                },
                flatten_params={}
            )
            self.status_bar.showMessage(f"项目已保存: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_load_project(self):
        """加载项目"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载项目",
            "",
            "JSON文件 (*.json)"
        )

        if not filepath:
            return

        try:
            data = Exporter.load_project(filepath)
            if data is None:
                raise Exception("无法解析项目文件")

            # 加载模型
            mesh_path = data.get('mesh_path', '')
            if mesh_path and Path(mesh_path).exists():
                self.mesh = MeshLoader.load(mesh_path)
                if self.mesh:
                    self.mesh_path = mesh_path
                    self.viewer_3d.load_mesh(self.mesh)
                    self.path_manager = PathManager(self.mesh)
                    self.groove_generator = GrooveGenerator(self.mesh)
                    self.conformal_mapper = ConformalMapper(self.mesh)

            # 恢复点位
            if self.path_manager and 'ir_points' in data:
                self.path_manager.from_dict(data['ir_points'])

                # 更新UI
                self.point_panel.clear_all()
                for pid, point in self.path_manager.ir_points.items():
                    self.point_panel.add_point(
                        point_id=pid,
                        position=point.position,
                        name=point.name,
                        is_center=point.is_center,
                        group=point.group
                    )
                    self.viewer_3d.add_ir_point(
                        point_id=pid,
                        position=point.position,
                        is_center=point.is_center,
                        name=point.name
                    )

            self.status_bar.showMessage(f"项目已加载: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_clear_points(self):
        """清除所有点"""
        reply = QMessageBox.question(
            self, "确认",
            "确定要清除所有点吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.path_manager:
                self.path_manager.ir_points.clear()
                self.path_manager.paths.clear()
                self.path_manager.center_point_id = None

            self.point_panel.clear_all()
            self.viewer_3d.clear_all()

            if self.mesh:
                self.viewer_3d.load_mesh(self.mesh)

    def _on_clear_paths(self):
        """清除路径"""
        self.viewer_3d.clear_paths()
        self.viewer_3d.clear_grooves()

        if self.path_manager:
            self.path_manager.paths.clear()

    def _on_about(self):
        """关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "IR阵列定位辅助生成工具\n\n"
            "用于在3D曲面模型上布置IR阵列点位，\n"
            "计算测地线连接路径，生成FPC走线凹槽，\n"
            "并将曲面展开为2D平面用于FPC制造。\n\n"
            "版本: 2.0"
        )

    # ========== 坐标系统相关方法 ==========

    def _on_origin_changed(self, point_id: str):
        """坐标原点变更"""
        # 更新点位面板的原点标记
        self.point_panel.set_origin_point(point_id if point_id else None)

        if point_id and point_id in self.point_panel.points:
            position = self.point_panel.points[point_id]['position']
            self._update_coordinate_axes()
            self.status_bar.showMessage(f"坐标原点已设置为: {point_id[:8]}")
        else:
            self.viewer_3d.remove_coordinate_axes()
            self.status_bar.showMessage("坐标原点已清除")

    def _on_axis_changed(self):
        """坐标轴方向变更"""
        self._update_coordinate_axes()

    def _on_show_axes_changed(self, visible: bool):
        """显示/隐藏坐标轴"""
        if visible:
            self._update_coordinate_axes()
        else:
            self.viewer_3d.remove_coordinate_axes()

    def _update_coordinate_axes(self):
        """更新坐标轴显示"""
        if not self.coord_panel.is_axes_visible():
            return

        origin = self.coord_panel.get_origin_position()
        if origin is None:
            return

        x_axis, y_axis, z_axis = self.coord_panel.get_axis_vectors()
        length = self.coord_panel.get_axis_length()
        shaft_scale = self.coord_panel.get_shaft_scale()
        tip_scale = self.coord_panel.get_tip_scale()

        self.viewer_3d.add_coordinate_axes(
            origin, x_axis, y_axis, z_axis, length, shaft_scale, tip_scale
        )

    def _on_set_selected_as_origin(self):
        """设选中点为坐标原点"""
        point_id = self.point_panel.get_selected_point_id()
        if point_id:
            self.coord_panel.set_origin_by_id(point_id)
            self.right_tabs.setCurrentIndex(1)  # 切换到坐标系统标签页
        else:
            QMessageBox.warning(self, "警告", "请先选择一个点")

    def _sync_points_to_coord_panel(self):
        """同步点位数据到坐标面板"""
        self.coord_panel.update_points(self.point_panel.points)

    # ========== 项目快照相关方法 ==========

    def _on_save_snapshot(self):
        """保存项目快照"""
        if self.path_manager is None or not self.point_panel.points:
            QMessageBox.warning(self, "警告", "没有可保存的项目数据")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存项目快照",
            "ir_array_project.irproj",
            "IR项目文件 (*.irproj)"
        )

        if not filepath:
            return

        self.status_bar.showMessage("正在保存项目快照...")

        try:
            # 收集点位数据
            ir_points_data = {}
            for point_id, info in self.point_panel.points.items():
                ir_points_data[point_id] = {
                    'id': point_id,
                    'name': info.get('name', ''),
                    'position': info['position'],
                    'is_center': info.get('is_center', False),
                    'group': info.get('group', 'default'),
                    'pad_length': info.get('pad_length', 3.0),
                    'pad_width': info.get('pad_width', 2.0)
                }

            # 收集参数
            groove_params = self.param_panel.get_groove_params()
            fpc_params = self.param_panel.get_fpc_params()

            groove_params_dict = {
                'width': groove_params.width,
                'depth': groove_params.depth,
                'auto_width': groove_params.auto_width,
                'min_width': groove_params.min_width,
                'max_width': groove_params.max_width,
                'conform_to_surface': groove_params.conform_to_surface,
                'extension_height': groove_params.extension_height,
                'pad_enabled': groove_params.pad_enabled,
                'pad_length': groove_params.pad_length,
                'pad_width': groove_params.pad_width
            }

            fpc_params_dict = {
                'groove_width': fpc_params.groove_width,
                'pad_radius': fpc_params.pad_radius,
                'center_pad_radius': fpc_params.center_pad_radius
            }

            # 坐标系统设置
            coord_system = self.coord_panel.to_dict()

            # 路径数据
            paths_data = None
            if self.path_manager and self.path_manager.smooth_paths:
                paths_data = {}
                for path_id, path in self.path_manager.smooth_paths.items():
                    paths_data[path_id] = path

            # 保存快照
            success = ProjectSnapshot.save_snapshot(
                filepath=filepath,
                mesh_path=self.mesh_path,
                ir_points_data=ir_points_data,
                groove_params=groove_params_dict,
                fpc_params=fpc_params_dict,
                coord_system=coord_system,
                paths_data=paths_data
            )

            if success:
                self.status_bar.showMessage(f"项目快照已保存: {filepath}")
                QMessageBox.information(self, "保存成功", f"项目快照已保存到:\n{filepath}")
            else:
                QMessageBox.critical(self, "保存失败", "无法保存项目快照")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_load_snapshot(self):
        """加载项目快照"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载项目快照",
            "",
            "IR项目文件 (*.irproj)"
        )

        if not filepath:
            return

        self.status_bar.showMessage("正在加载项目快照...")

        try:
            data = ProjectSnapshot.load_snapshot(filepath)
            if data is None:
                raise Exception("无法解析项目文件")

            # 加载模型
            mesh_path = data.get('mesh_path')
            if mesh_path and Path(mesh_path).exists():
                self.mesh = MeshLoader.load(mesh_path)
                if self.mesh:
                    self.mesh_path = mesh_path
                    self.viewer_3d.load_mesh(self.mesh)
                    self.path_manager = PathManager(self.mesh)
                    self.groove_generator = GrooveGenerator(self.mesh)
                    self.conformal_mapper = ConformalMapper(self.mesh)
            else:
                QMessageBox.warning(
                    self, "警告",
                    f"找不到原始网格文件: {mesh_path}\n请手动加载模型。"
                )

            # 恢复点位
            if 'ir_points' in data:
                self.point_panel.clear_all()
                if self.path_manager:
                    self.path_manager.ir_points.clear()

                for point_id, info in data['ir_points'].items():
                    position = info['position']
                    if isinstance(position, list):
                        position = np.array(position)

                    # 添加到点面板
                    self.point_panel.add_point(
                        point_id=point_id,
                        position=position,
                        name=info.get('name', ''),
                        is_center=info.get('is_center', False),
                        group=info.get('group', 'default'),
                        pad_length=info.get('pad_length', 3.0),
                        pad_width=info.get('pad_width', 2.0)
                    )

                    # 添加到3D视图
                    if self.mesh:
                        self.viewer_3d.add_ir_point(
                            point_id=point_id,
                            position=position,
                            is_center=info.get('is_center', False),
                            name=info.get('name', '')
                        )

                    # 添加到PathManager
                    if self.path_manager:
                        ir_point = IRPoint(
                            id=point_id,
                            position=position,
                            name=info.get('name', ''),
                            is_center=info.get('is_center', False),
                            group=info.get('group', 'default'),
                            pad_length=info.get('pad_length', 3.0),
                            pad_width=info.get('pad_width', 2.0)
                        )
                        self.path_manager.ir_points[point_id] = ir_point

                        if info.get('is_center', False):
                            self.path_manager.center_point_id = point_id

            # 恢复凹槽参数
            if 'groove_params' in data:
                gp = data['groove_params']
                self.param_panel.groove_width_spin.setValue(gp.get('width', 1.0))
                self.param_panel.groove_depth_spin.setValue(gp.get('depth', 0.5))

            # 恢复坐标系统设置
            if 'coord_system' in data:
                self._sync_points_to_coord_panel()
                self.coord_panel.from_dict(data['coord_system'])
                self._update_coordinate_axes()

            self.status_bar.showMessage(f"项目快照已加载: {filepath}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "加载失败", str(e))

    # ========== 导出相关方法 ==========

    def _on_export_to_folder(self):
        """导出到文件夹"""
        if self.mesh is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 从参数面板获取项目名称
        project_name = self.param_panel.get_project_name()

        # 获取输出目录
        export_params = self.param_panel.get_export_params()
        output_dir = export_params.output_dir

        if not output_dir:
            # 选择输出目录
            output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if not output_dir:
                return

        self.status_bar.showMessage("正在导出文件...")

        try:
            exporter = FolderExporter(output_dir, project_name)

            # 获取2D数据
            paths_2d = self.view_2d.get_paths_for_export()
            ir_points_2d = self.view_2d.get_points_for_export()

            # 收集IR点数据
            ir_points_data = {}
            for point_id, info in self.point_panel.points.items():
                ir_points_data[point_id] = {
                    'name': info.get('name', ''),
                    'position': info['position'],
                    'is_center': info.get('is_center', False),
                    'group': info.get('group', 'default'),
                    'pad_length': info.get('pad_length', 3.0),
                    'pad_width': info.get('pad_width', 2.0)
                }

            # 获取参数
            groove_params = self.param_panel.get_groove_params()
            fpc_params = self.param_panel.get_fpc_params()

            groove_params_dict = {
                'width': groove_params.width,
                'depth': groove_params.depth,
                'pad_enabled': groove_params.pad_enabled,
                'pad_length': groove_params.pad_length,
                'pad_width': groove_params.pad_width
            }

            fpc_params_dict = {
                'groove_width': fpc_params.groove_width,
                'pad_radius': fpc_params.pad_radius,
                'center_pad_radius': fpc_params.center_pad_radius
            }

            coord_system = self.coord_panel.to_dict()

            # 获取FPC布局数据
            fpc_layout = None
            if self.view_2d.has_fpc_layout():
                fpc_layout = self.view_2d.get_fpc_layout_for_export()

            exported = exporter.export_all(
                mesh=self.mesh,
                mesh_with_grooves=self.mesh_with_grooves,
                paths_2d=paths_2d,
                ir_points_2d=ir_points_2d,
                ir_points_data=ir_points_data,
                groove_params=groove_params_dict,
                fpc_params=fpc_params_dict,
                coord_system=coord_system,
                fpc_layout=fpc_layout
            )

            # 显示结果
            output_path = exported.get('output_dir', output_dir)
            msg = f"导出完成!\n\n输出目录: {output_path}\n\n导出文件: {len(exported)} 个"
            QMessageBox.information(self, "导出成功", msg)

            self.status_bar.showMessage(f"已导出到: {output_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))

    def _on_export_coordinates(self):
        """导出坐标文件"""
        if not self.point_panel.points:
            QMessageBox.warning(self, "警告", "没有可导出的点位")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出坐标文件",
            "coordinates.csv",
            "CSV文件 (*.csv);;JSON文件 (*.json)"
        )

        if not filepath:
            return

        try:
            # 收集点位数据
            ir_points_data = {}
            for point_id, info in self.point_panel.points.items():
                ir_points_data[point_id] = {
                    'name': info.get('name', ''),
                    'position': info['position'],
                    'is_center': info.get('is_center', False),
                    'pad_length': info.get('pad_length', 3.0),
                    'pad_width': info.get('pad_width', 2.0)
                }

            coord_system = self.coord_panel.to_dict()
            origin_id = coord_system.get('origin_point_id')
            axis_rotation = coord_system.get('axis_rotation')

            # 确定坐标系类型
            coordinate_system = "local" if origin_id else "world"

            if filepath.endswith('.csv'):
                success = BatchExporter.export_coordinates_csv(
                    filepath, ir_points_data,
                    origin_point_id=origin_id,
                    axis_rotation=axis_rotation,
                    coordinate_system=coordinate_system
                )
            else:
                success = BatchExporter.export_coordinates(
                    filepath, ir_points_data,
                    origin_point_id=origin_id,
                    axis_rotation=axis_rotation,
                    coordinate_system=coordinate_system
                )

            if success:
                self.status_bar.showMessage(f"坐标文件已导出: {filepath}")
                QMessageBox.information(self, "导出成功", f"坐标文件已保存到:\n{filepath}")
            else:
                QMessageBox.critical(self, "导出失败", "无法保存坐标文件")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))

    def _on_export_single(self, export_type: str):
        """单独导出某类型文件"""
        try:
            # 定义文件过滤器和默认文件名
            filters = {
                'stl': ("STL文件 (*.stl)", "model.stl"),
                'groove_stl': ("STL文件 (*.stl)", "model_with_grooves.stl"),
                'dxf': ("DXF文件 (*.dxf)", "2d_paths.dxf"),
                'svg': ("SVG文件 (*.svg)", "2d_paths.svg"),
                'fpc_dxf': ("DXF文件 (*.dxf)", "fpc_layout.dxf"),
                'fpc_svg': ("SVG文件 (*.svg)", "fpc_layout.svg"),
                'world_json': ("JSON文件 (*.json)", "world_coordinates.json"),
                'world_csv': ("CSV文件 (*.csv)", "world_coordinates.csv"),
                'local_json': ("JSON文件 (*.json)", "local_coordinates.json"),
                'local_csv': ("CSV文件 (*.csv)", "local_coordinates.csv"),
                'project': ("JSON文件 (*.json)", "project_config.json")
            }

            if export_type not in filters:
                QMessageBox.warning(self, "警告", f"未知的导出类型: {export_type}")
                return

            file_filter, default_name = filters[export_type]

            filepath, _ = QFileDialog.getSaveFileName(
                self, "保存文件", default_name, file_filter
            )

            if not filepath:
                return

            success = False
            exporter = Exporter

            if export_type == 'stl':
                if self.mesh is None:
                    QMessageBox.warning(self, "警告", "没有可导出的模型")
                    return
                success = exporter.export_mesh(self.mesh, filepath)

            elif export_type == 'groove_stl':
                if self.mesh_with_grooves is None:
                    QMessageBox.warning(self, "警告", "没有可导出的带凹槽模型，请先生成凹槽")
                    return
                success = exporter.export_mesh(self.mesh_with_grooves, filepath)

            elif export_type == 'dxf':
                paths_2d = self.view_2d.get_paths_for_export()
                ir_points_2d = self.view_2d.get_points_for_export()
                if not paths_2d:
                    QMessageBox.warning(self, "警告", "没有可导出的2D路径，请先展开曲面")
                    return
                success = exporter.export_dxf(paths_2d, ir_points_2d, filepath)

            elif export_type == 'svg':
                paths_2d = self.view_2d.get_paths_for_export()
                ir_points_2d = self.view_2d.get_points_for_export()
                if not paths_2d:
                    QMessageBox.warning(self, "警告", "没有可导出的2D路径，请先展开曲面")
                    return
                success = exporter.export_svg(paths_2d, ir_points_2d, filepath)

            elif export_type in ('fpc_dxf', 'fpc_svg'):
                if not self.view_2d.has_fpc_layout():
                    QMessageBox.warning(self, "警告", "没有可导出的FPC布局，请先生成FPC图纸")
                    return
                fpc_layout = self.view_2d.get_fpc_layout_for_export()
                groove_outlines = fpc_layout.get('groove_outlines', [])
                ir_pads = fpc_layout.get('ir_pads', [])
                center_pad = fpc_layout.get('center_pad')
                merged_outline = fpc_layout.get('merged_outline')
                ir_points_2d = self.view_2d.get_points_for_export()

                if export_type == 'fpc_dxf':
                    success = exporter.export_fpc_dxf(
                        groove_outlines, ir_pads, center_pad, merged_outline,
                        ir_points_2d, filepath
                    )
                else:
                    success = exporter.export_fpc_svg(
                        groove_outlines, ir_pads, center_pad, merged_outline,
                        ir_points_2d, filepath
                    )

            elif export_type in ('world_json', 'world_csv', 'local_json', 'local_csv'):
                if not self.point_panel.points:
                    QMessageBox.warning(self, "警告", "没有可导出的点位")
                    return

                # 收集点位数据
                ir_points_data = {}
                for point_id, info in self.point_panel.points.items():
                    ir_points_data[point_id] = {
                        'name': info.get('name', ''),
                        'position': info['position'],
                        'is_center': info.get('is_center', False),
                        'pad_length': info.get('pad_length', 3.0),
                        'pad_width': info.get('pad_width', 2.0)
                    }

                coord_system = self.coord_panel.to_dict()
                origin_id = coord_system.get('origin_point_id')
                axis_rotation = coord_system.get('axis_rotation')

                is_local = export_type.startswith('local')
                is_csv = export_type.endswith('csv')

                if is_local and not origin_id:
                    QMessageBox.warning(self, "警告", "导出局部坐标需要先设置坐标原点")
                    return

                coordinate_system = "local" if is_local else "world"

                if is_csv:
                    success = BatchExporter.export_coordinates_csv(
                        filepath, ir_points_data,
                        origin_point_id=origin_id if is_local else None,
                        axis_rotation=axis_rotation if is_local else None,
                        coordinate_system=coordinate_system
                    )
                else:
                    success = BatchExporter.export_coordinates(
                        filepath, ir_points_data,
                        origin_point_id=origin_id if is_local else None,
                        axis_rotation=axis_rotation if is_local else None,
                        coordinate_system=coordinate_system
                    )

            elif export_type == 'project':
                # 导出项目配置
                groove_params = self.param_panel.get_groove_params()
                fpc_params = self.param_panel.get_fpc_params()
                coord_system = self.coord_panel.to_dict()

                config = {
                    'groove_params': {
                        'width': groove_params.width,
                        'depth': groove_params.depth,
                        'pad_enabled': groove_params.pad_enabled,
                        'pad_length': groove_params.pad_length,
                        'pad_width': groove_params.pad_width
                    },
                    'fpc_params': {
                        'groove_width': fpc_params.groove_width,
                        'pad_radius': fpc_params.pad_radius,
                        'center_pad_radius': fpc_params.center_pad_radius
                    },
                    'coord_system': coord_system
                }

                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                success = True

            if success:
                self.status_bar.showMessage(f"文件已保存: {filepath}")
                QMessageBox.information(self, "保存成功", f"文件已保存到:\n{filepath}")
            else:
                QMessageBox.critical(self, "保存失败", "无法保存文件")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", str(e))

    # ========== 精度验证相关方法 ==========

    def _on_validate_flatten(self):
        """验证展平精度"""
        if self.path_flatten_result is None:
            QMessageBox.warning(self, "警告", "请先展开路径")
            return

        if self.mesh is None or self.path_manager is None:
            QMessageBox.warning(self, "警告", "请先加载模型和生成路径")
            return

        self.status_bar.showMessage("正在验证展平精度...")

        try:
            # 准备3D路径数据
            paths_3d = {}
            for point_id, smooth_path in self.path_manager.smooth_paths.items():
                if len(smooth_path) > 0:
                    paths_3d[point_id] = smooth_path

            # 准备3D IR点数据
            ir_points_3d = {}
            for point_id, point in self.path_manager.ir_points.items():
                ir_points_3d[point_id] = (point.position, point.name, point.is_center)

            # 验证
            validation_result = validate_flatten_accuracy(
                mesh=self.mesh,
                uv_coords=self.path_flatten_result.uv_coords,
                paths_3d=paths_3d,
                paths_2d=self.path_flatten_result.paths_2d,
                ir_points=ir_points_3d,
                ir_points_2d=self.path_flatten_result.ir_points_2d
            )

            # 显示结果
            if validation_result.is_valid:
                QMessageBox.information(
                    self, "验证通过",
                    f"{validation_result.validation_message}\n\n"
                    f"平均长度误差: {validation_result.avg_length_error:.2f}%\n"
                    f"最大长度误差: {validation_result.max_length_error:.2f}%\n"
                    f"平均位置误差: {validation_result.avg_distance_error:.4f}mm\n"
                    f"最大位置误差: {validation_result.max_distance_error:.4f}mm"
                )
            else:
                QMessageBox.warning(
                    self, "验证失败",
                    f"{validation_result.validation_message}\n\n"
                    f"平均长度误差: {validation_result.avg_length_error:.2f}%\n"
                    f"最大长度误差: {validation_result.max_length_error:.2f}%\n"
                    f"平均位置误差: {validation_result.avg_distance_error:.4f}mm\n"
                    f"最大位置误差: {validation_result.max_distance_error:.4f}mm\n\n"
                    "建议检查模型和路径，可能需要调整展平参数"
                )

            self.status_bar.showMessage(validation_result.validation_message)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"验证失败: {e}")

    # ========== FPC贴合动画相关方法 ==========

    def _on_play_fitting_animation(self):
        """播放FPC贴合动画"""
        if self.fpc_layout_result is None:
            QMessageBox.warning(self, "警告", "请先生成FPC图纸")
            return

        if self.path_flatten_result is None:
            QMessageBox.warning(self, "警告", "请先展开路径")
            return

        if self.mesh is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        self.status_bar.showMessage("正在准备FPC贴合动画...")

        try:
            # 创建动画控制器
            self.fpc_animator = FPCFittingAnimator(
                mesh=self.mesh,
                uv_coords=self.path_flatten_result.uv_coords,
                fpc_layout=self.fpc_layout_result
            )

            if not self.fpc_animator.is_ready():
                QMessageBox.warning(self, "警告", "无法创建动画，请检查FPC布局数据")
                return

            # 切换到3D视图
            self.view_tabs.setCurrentIndex(0)

            # 初始化动画参数
            self.animation_progress = 0.0

            # 使用QTimer创建动画
            from PyQt5.QtCore import QTimer
            if self.animation_timer is not None:
                self.animation_timer.stop()

            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._on_animation_frame)
            self.animation_timer.start(33)  # 约30fps

            self.status_bar.showMessage("播放FPC贴合动画... (0%)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"动画播放失败: {e}")

    def _on_animation_frame(self):
        """动画帧更新"""
        if self.fpc_animator is None or not self.fpc_animator.is_ready():
            self._stop_animation()
            return

        # 更新进度
        self.animation_progress += 0.015  # 约2秒完成一次动画

        if self.animation_progress >= 1.0:
            # 动画完成后保持最终状态一段时间，然后停止
            self.animation_progress = 1.0
            self._update_animation_display()
            self._stop_animation()
            self.status_bar.showMessage("FPC贴合动画完成")
            return

        self._update_animation_display()
        progress_percent = int(self.animation_progress * 100)
        self.status_bar.showMessage(f"播放FPC贴合动画... ({progress_percent}%)")

    def _update_animation_display(self):
        """更新动画显示"""
        if self.fpc_animator is None:
            return

        frame_mesh = self.fpc_animator.get_frame(self.animation_progress)
        if frame_mesh is None:
            return

        # 更新3D视图中的FPC网格
        self.viewer_3d.update_fpc_animation_mesh(frame_mesh)

    def _stop_animation(self):
        """停止动画"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
