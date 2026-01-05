"""主窗口"""
import sys
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QStatusBar,
    QTabWidget, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from typing import Optional
import trimesh

from .viewer_3d import Viewer3D
from .view_2d import View2D
from .point_panel import PointPanel
from .param_panel import ParamPanel
from core.mesh_loader import MeshLoader
from core.geodesic import PathManager, IRPoint
from core.groove_gen import GrooveGenerator
from core.conformal_map import ConformalMapper, FlattenResult
from core.path_flatten import flatten_paths_preserve_geometry, PathFlattenResult, generate_fpc_layout, FPCLayoutResult
from core.exporter import Exporter, BatchExporter


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

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 点位管理
        self.point_panel = PointPanel()
        right_layout.addWidget(self.point_panel)

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

        save_action = QAction("保存项目(&S)...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        load_project_action = QAction("加载项目(&L)...", self)
        load_project_action.triggered.connect(self._on_load_project)
        file_menu.addAction(load_project_action)

        file_menu.addSeparator()

        export_action = QAction("导出所有(&E)...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_all)
        file_menu.addAction(export_action)

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

        # 参数面板
        self.param_panel.generate_grooves_clicked.connect(self._on_generate_grooves)
        self.param_panel.flatten_clicked.connect(self._on_flatten)
        self.param_panel.generate_fpc_layout_clicked.connect(self._on_generate_fpc_layout)
        self.param_panel.export_all_clicked.connect(self._on_export_all)

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

        # 更新状态
        info = MeshLoader.get_mesh_info(self.mesh)
        self.status_bar.showMessage(
            f"已加载: {Path(filepath).name} | "
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

        # 计算路径（启用平滑）
        paths = self.path_manager.compute_all_paths(smooth=True)

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
        print(f"凹槽参数: 宽度={params.width}, 深度={params.depth}")

        self.groove_generator.set_params(
            width=params.width,
            depth=params.depth,
            auto_width=params.auto_width,
            min_width=params.min_width,
            max_width=params.max_width
        )

        # 清除旧凹槽
        self.viewer_3d.clear_grooves()

        # 收集所有凹槽网格
        all_grooves = []
        groove_count = 0

        for point_id in self.path_manager.smooth_paths.keys():
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

            groove = self.groove_generator.generate_groove_profile(
                smooth_path, normals
            )

            if groove is not None:
                print(f"凹槽 {point_id}: {len(groove.vertices)} 顶点, {len(groove.faces)} 面")
                if len(groove.vertices) > 0 and len(groove.faces) > 0:
                    self.viewer_3d.add_groove_preview(groove, point_id)
                    all_grooves.append(groove)
                    groove_count += 1
                else:
                    print(f"凹槽 {point_id}: 顶点或面为空")
            else:
                print(f"凹槽 {point_id}: 生成返回 None")

        print(f"=== 凹槽生成完成: {groove_count} 个 ===\n")

        if groove_count > 0:
            self.status_bar.showMessage(f"已生成 {groove_count} 个凹槽预览")

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
        """计算路径点的表面法向量"""
        normals = []
        for point in path_points:
            # 找最近的顶点
            distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            normals.append(self.mesh.vertex_normals[nearest_idx])
        return np.array(normals)

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

            # 生成FPC布局
            self.fpc_layout_result = generate_fpc_layout(
                flatten_result=self.path_flatten_result,
                groove_width=fpc_params.groove_width,
                pad_radius=fpc_params.pad_radius,
                center_pad_radius=fpc_params.center_pad_radius
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

            self.status_bar.showMessage(
                f"FPC图纸生成完成 - 走线宽度: {fpc_params.groove_width}mm, "
                f"{len(self.fpc_layout_result.groove_outlines)} 条凹槽轮廓"
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
            "版本: 1.0"
        )
