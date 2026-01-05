"""文件导出模块"""
import numpy as np
import trimesh
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


class Exporter:
    """文件导出器"""

    @staticmethod
    def export_mesh(mesh: trimesh.Trimesh, filepath: str) -> bool:
        """
        导出3D网格

        Args:
            mesh: 网格对象
            filepath: 保存路径

        Returns:
            是否成功
        """
        try:
            mesh.export(filepath)
            return True
        except Exception as e:
            print(f"导出网格失败: {e}")
            return False

    @staticmethod
    def export_dxf(
        paths_2d: List[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        filepath: str,
        scale: float = 1.0,
        line_width: float = 1.0
    ) -> bool:
        """
        导出2D路径为DXF格式

        Args:
            paths_2d: 2D路径列表
            ir_points_2d: IR点2D坐标和名称
            filepath: 保存路径
            scale: 缩放因子（mm）
            line_width: 线宽

        Returns:
            是否成功
        """
        if not HAS_EZDXF:
            print("ezdxf库未安装，无法导出DXF")
            return False

        try:
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # 创建图层
            doc.layers.add('PATHS', color=1)  # 红色
            doc.layers.add('IR_POINTS', color=3)  # 绿色
            doc.layers.add('LABELS', color=7)  # 白色

            # 绘制路径
            for path in paths_2d:
                if len(path) < 2:
                    continue
                points = [(p[0] * scale, p[1] * scale) for p in path]
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'PATHS', 'lineweight': int(line_width * 100)}
                )

            # 绘制IR点
            point_radius = 0.5 * scale
            for point_2d, name in ir_points_2d:
                x, y = point_2d[0] * scale, point_2d[1] * scale
                msp.add_circle(
                    (x, y),
                    radius=point_radius,
                    dxfattribs={'layer': 'IR_POINTS'}
                )
                msp.add_text(
                    name,
                    dxfattribs={
                        'layer': 'LABELS',
                        'height': point_radius * 0.8
                    }
                ).set_placement((x + point_radius * 1.2, y))

            doc.saveas(filepath)
            return True

        except Exception as e:
            print(f"导出DXF失败: {e}")
            return False

    @staticmethod
    def export_fpc_dxf(
        groove_outlines: List[np.ndarray],
        ir_pads: List[np.ndarray],
        center_pad: np.ndarray,
        merged_outline: Optional[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        filepath: str,
        scale: float = 1.0
    ) -> bool:
        """
        导出FPC布局图为DXF格式

        Args:
            groove_outlines: 凹槽轮廓列表
            ir_pads: IR点焊盘轮廓列表
            center_pad: 中心焊盘轮廓
            merged_outline: 合并后的总轮廓
            ir_points_2d: IR点2D坐标和名称
            filepath: 保存路径
            scale: 缩放因子

        Returns:
            是否成功
        """
        if not HAS_EZDXF:
            print("ezdxf库未安装，无法导出DXF")
            return False

        try:
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # 创建图层
            doc.layers.add('GROOVES', color=1)  # 红色 - 凹槽轮廓
            doc.layers.add('IR_PADS', color=3)  # 绿色 - IR焊盘
            doc.layers.add('CENTER_PAD', color=5)  # 蓝色 - 中心焊盘
            doc.layers.add('OUTLINE', color=7)  # 白色 - 总轮廓
            doc.layers.add('LABELS', color=7)  # 白色 - 标签

            # 绘制凹槽轮廓
            for outline in groove_outlines:
                if len(outline) < 3:
                    continue
                points = [(p[0] * scale, p[1] * scale) for p in outline]
                # 闭合多边形
                points.append(points[0])
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'GROOVES'}
                )

            # 绘制IR点焊盘
            for pad in ir_pads:
                if len(pad) < 3:
                    continue
                points = [(p[0] * scale, p[1] * scale) for p in pad]
                points.append(points[0])
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'IR_PADS'}
                )

            # 绘制中心焊盘
            if center_pad is not None and len(center_pad) >= 3:
                points = [(p[0] * scale, p[1] * scale) for p in center_pad]
                points.append(points[0])
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'CENTER_PAD'}
                )

            # 绘制总轮廓
            if merged_outline is not None and len(merged_outline) >= 3:
                points = [(p[0] * scale, p[1] * scale) for p in merged_outline]
                points.append(points[0])
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'OUTLINE', 'lineweight': 50}
                )

            # 添加标签
            for point_2d, name in ir_points_2d:
                x, y = point_2d[0] * scale, point_2d[1] * scale
                msp.add_text(
                    name,
                    dxfattribs={
                        'layer': 'LABELS',
                        'height': 1.0
                    }
                ).set_placement((x + 2, y))

            doc.saveas(filepath)
            return True

        except Exception as e:
            print(f"导出FPC DXF失败: {e}")
            return False

    @staticmethod
    def export_fpc_svg(
        groove_outlines: List[np.ndarray],
        ir_pads: List[np.ndarray],
        center_pad: np.ndarray,
        merged_outline: Optional[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        filepath: str,
        scale: float = 1.0,
        stroke_width: float = 0.5,
        canvas_margin: float = 10.0
    ) -> bool:
        """
        导出FPC布局图为SVG格式

        Args:
            groove_outlines: 凹槽轮廓列表
            ir_pads: IR点焊盘轮廓列表
            center_pad: 中心焊盘轮廓
            merged_outline: 合并后的总轮廓
            ir_points_2d: IR点2D坐标和名称
            filepath: 保存路径
            scale: 缩放因子
            stroke_width: 线宽
            canvas_margin: 画布边距

        Returns:
            是否成功
        """
        try:
            # 收集所有点计算边界
            all_points = []
            for outline in groove_outlines:
                all_points.extend(outline)
            for pad in ir_pads:
                all_points.extend(pad)
            if center_pad is not None:
                all_points.extend(center_pad)
            if merged_outline is not None:
                all_points.extend(merged_outline)
            for point, _ in ir_points_2d:
                all_points.append(point)

            if not all_points:
                return False

            all_points = np.array(all_points) * scale
            min_xy = all_points.min(axis=0) - canvas_margin
            max_xy = all_points.max(axis=0) + canvas_margin
            width = max_xy[0] - min_xy[0]
            height = max_xy[1] - min_xy[1]

            # 生成SVG
            svg_lines = [
                f'<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="{min_xy[0]} {-max_xy[1]} {width} {height}" '
                f'width="{width}mm" height="{height}mm">',
                f'<g transform="scale(1,-1)">',
            ]

            # 凹槽轮廓（橙色填充）
            svg_lines.append(f'<g stroke="#f39c12" fill="#f39c1233" stroke-width="{stroke_width}">')
            for outline in groove_outlines:
                if len(outline) < 3:
                    continue
                points_str = ' '.join([f'{p[0]*scale},{p[1]*scale}' for p in outline])
                svg_lines.append(f'  <polygon points="{points_str}"/>')
            svg_lines.append('</g>')

            # IR焊盘（绿色填充）
            svg_lines.append(f'<g stroke="#2ecc71" fill="#2ecc7166" stroke-width="{stroke_width}">')
            for pad in ir_pads:
                if len(pad) < 3:
                    continue
                points_str = ' '.join([f'{p[0]*scale},{p[1]*scale}' for p in pad])
                svg_lines.append(f'  <polygon points="{points_str}"/>')
            svg_lines.append('</g>')

            # 中心焊盘（红色填充）
            if center_pad is not None and len(center_pad) >= 3:
                svg_lines.append(f'<g stroke="#e74c3c" fill="#e74c3c66" stroke-width="{stroke_width}">')
                points_str = ' '.join([f'{p[0]*scale},{p[1]*scale}' for p in center_pad])
                svg_lines.append(f'  <polygon points="{points_str}"/>')
                svg_lines.append('</g>')

            # 总轮廓（虚线）
            if merged_outline is not None and len(merged_outline) >= 3:
                svg_lines.append(f'<g stroke="#ffffff" fill="none" stroke-width="{stroke_width * 2}" stroke-dasharray="5,3">')
                points_str = ' '.join([f'{p[0]*scale},{p[1]*scale}' for p in merged_outline])
                svg_lines.append(f'  <polygon points="{points_str}"/>')
                svg_lines.append('</g>')

            # 标签
            svg_lines.append(f'<g fill="black" font-size="2">')
            for point, name in ir_points_2d:
                x, y = point[0] * scale, point[1] * scale
                svg_lines.append(
                    f'  <text x="{x + 2}" y="{y}" transform="scale(1,-1) translate(0,{-2*y})">{name}</text>'
                )
            svg_lines.append('</g>')

            svg_lines.append('</g>')
            svg_lines.append('</svg>')

            with open(filepath, 'w') as f:
                f.write('\n'.join(svg_lines))

            return True

        except Exception as e:
            print(f"导出FPC SVG失败: {e}")
            return False

    @staticmethod
    def export_svg(
        paths_2d: List[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        filepath: str,
        scale: float = 1.0,
        stroke_width: float = 0.5,
        canvas_margin: float = 10.0
    ) -> bool:
        """
        导出2D路径为SVG格式

        Args:
            paths_2d: 2D路径列表
            ir_points_2d: IR点2D坐标和名称
            filepath: 保存路径
            scale: 缩放因子
            stroke_width: 线宽
            canvas_margin: 画布边距

        Returns:
            是否成功
        """
        try:
            # 计算边界
            all_points = []
            for path in paths_2d:
                all_points.extend(path)
            for point, _ in ir_points_2d:
                all_points.append(point)

            if not all_points:
                return False

            all_points = np.array(all_points) * scale
            min_xy = all_points.min(axis=0) - canvas_margin
            max_xy = all_points.max(axis=0) + canvas_margin
            width = max_xy[0] - min_xy[0]
            height = max_xy[1] - min_xy[1]

            # 生成SVG
            svg_lines = [
                f'<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="{min_xy[0]} {-max_xy[1]} {width} {height}" '
                f'width="{width}mm" height="{height}mm">',
                f'<g transform="scale(1,-1)">',
            ]

            # 路径
            svg_lines.append(f'<g stroke="red" fill="none" stroke-width="{stroke_width}">')
            for path in paths_2d:
                if len(path) < 2:
                    continue
                points_str = ' '.join([f'{p[0]*scale},{p[1]*scale}' for p in path])
                svg_lines.append(f'  <polyline points="{points_str}"/>')
            svg_lines.append('</g>')

            # IR点
            svg_lines.append(f'<g fill="green">')
            point_radius = 0.5 * scale
            for point, name in ir_points_2d:
                x, y = point[0] * scale, point[1] * scale
                svg_lines.append(f'  <circle cx="{x}" cy="{y}" r="{point_radius}"/>')
                svg_lines.append(
                    f'  <text x="{x + point_radius * 1.5}" y="{y}" '
                    f'font-size="{point_radius}" fill="black">{name}</text>'
                )
            svg_lines.append('</g>')

            svg_lines.append('</g>')
            svg_lines.append('</svg>')

            with open(filepath, 'w') as f:
                f.write('\n'.join(svg_lines))

            return True

        except Exception as e:
            print(f"导出SVG失败: {e}")
            return False

    @staticmethod
    def export_project(
        filepath: str,
        mesh_path: str,
        ir_points_data: dict,
        groove_params: dict,
        flatten_params: dict
    ) -> bool:
        """
        导出项目配置为JSON

        Args:
            filepath: 保存路径
            mesh_path: 网格文件路径
            ir_points_data: IR点数据
            groove_params: 凹槽参数
            flatten_params: 展开参数

        Returns:
            是否成功
        """
        try:
            project_data = {
                'version': '1.0',
                'mesh_path': mesh_path,
                'ir_points': ir_points_data,
                'groove_params': groove_params,
                'flatten_params': flatten_params,
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"导出项目失败: {e}")
            return False

    @staticmethod
    def load_project(filepath: str) -> Optional[dict]:
        """
        加载项目配置

        Args:
            filepath: 文件路径

        Returns:
            项目数据字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载项目失败: {e}")
            return None


class BatchExporter:
    """批量导出器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        mesh: trimesh.Trimesh,
        mesh_with_grooves: Optional[trimesh.Trimesh],
        paths_2d: List[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        ir_points_data: dict,
        groove_params: dict,
        project_name: str = "ir_array",
        fpc_layout: Optional[dict] = None
    ) -> Dict[str, str]:
        """
        一键导出所有文件

        Args:
            mesh: 原始网格
            mesh_with_grooves: 带凹槽的网格
            paths_2d: 2D路径列表
            ir_points_2d: IR点2D坐标和名称
            ir_points_data: IR点数据字典
            groove_params: 凹槽参数
            project_name: 项目名称
            fpc_layout: FPC布局数据（可选）

        Returns:
            文件路径字典
        """
        exported = {}

        # 原始网格
        mesh_path = self.output_dir / f"{project_name}_original.stl"
        if Exporter.export_mesh(mesh, str(mesh_path)):
            exported['original_mesh'] = str(mesh_path)

        # 带凹槽的网格
        if mesh_with_grooves is not None:
            groove_mesh_path = self.output_dir / f"{project_name}_with_grooves.stl"
            if Exporter.export_mesh(mesh_with_grooves, str(groove_mesh_path)):
                exported['groove_mesh'] = str(groove_mesh_path)

        # 2D路径 DXF
        dxf_path = self.output_dir / f"{project_name}_2d_paths.dxf"
        if Exporter.export_dxf(paths_2d, ir_points_2d, str(dxf_path)):
            exported['dxf'] = str(dxf_path)

        # 2D路径 SVG
        svg_path = self.output_dir / f"{project_name}_2d_paths.svg"
        if Exporter.export_svg(paths_2d, ir_points_2d, str(svg_path)):
            exported['svg'] = str(svg_path)

        # FPC布局图（如果有）
        if fpc_layout is not None:
            groove_outlines = fpc_layout.get('groove_outlines', [])
            ir_pads = fpc_layout.get('ir_pads', [])
            center_pad = fpc_layout.get('center_pad')
            merged_outline = fpc_layout.get('merged_outline')

            if groove_outlines:
                # FPC DXF
                fpc_dxf_path = self.output_dir / f"{project_name}_fpc_layout.dxf"
                if Exporter.export_fpc_dxf(
                    groove_outlines, ir_pads, center_pad, merged_outline,
                    ir_points_2d, str(fpc_dxf_path)
                ):
                    exported['fpc_dxf'] = str(fpc_dxf_path)

                # FPC SVG
                fpc_svg_path = self.output_dir / f"{project_name}_fpc_layout.svg"
                if Exporter.export_fpc_svg(
                    groove_outlines, ir_pads, center_pad, merged_outline,
                    ir_points_2d, str(fpc_svg_path)
                ):
                    exported['fpc_svg'] = str(fpc_svg_path)

        # 项目配置
        project_path = self.output_dir / f"{project_name}_project.json"
        if Exporter.export_project(
            str(project_path),
            str(mesh_path),
            ir_points_data,
            groove_params,
            {}
        ):
            exported['project'] = str(project_path)

        return exported
