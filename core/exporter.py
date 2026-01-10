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

            # 设置单位为毫米 (INSUNITS = 4 表示毫米)
            doc.header['$INSUNITS'] = 4
            # 设置测量单位为公制
            doc.header['$MEASUREMENT'] = 1

            # 创建图层
            doc.layers.add('PATHS', color=1)  # 红色
            doc.layers.add('IR_POINTS', color=3)  # 绿色
            doc.layers.add('LABELS', color=7)  # 白色

            # 绘制路径
            all_x = []
            all_y = []
            for path in paths_2d:
                if len(path) < 2:
                    continue
                points = [(p[0] * scale, p[1] * scale) for p in path]
                all_x.extend([p[0] for p in points])
                all_y.extend([p[1] for p in points])
                msp.add_lwpolyline(
                    points,
                    dxfattribs={'layer': 'PATHS', 'lineweight': int(line_width * 100)}
                )

            # 输出尺寸调试信息
            if all_x and all_y:
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                print(f"DXF导出尺寸 (mm): X范围={x_range:.2f}, Y范围={y_range:.2f}")
                print(f"  X: [{min(all_x):.2f}, {max(all_x):.2f}]")
                print(f"  Y: [{min(all_y):.2f}, {max(all_y):.2f}]")

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

            # 设置单位为毫米 (INSUNITS = 4 表示毫米)
            doc.header['$INSUNITS'] = 4
            # 设置测量单位为公制
            doc.header['$MEASUREMENT'] = 1

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
    def export_fpc_outline_only_dxf(
        groove_outlines: List[np.ndarray],
        ir_pads: List[np.ndarray],
        center_pad: np.ndarray,
        filepath: str,
        scale: float = 1.0
    ) -> bool:
        """
        导出FPC布局为单一封闭轮廓的DXF格式（无交叉线）

        使用布尔并集将所有凹槽和焊盘合并为一个封闭轮廓

        Args:
            groove_outlines: 凹槽轮廓列表
            ir_pads: IR点焊盘轮廓列表
            center_pad: 中心焊盘轮廓
            filepath: 保存路径
            scale: 缩放因子

        Returns:
            是否成功
        """
        if not HAS_EZDXF:
            print("ezdxf库未安装，无法导出DXF")
            return False

        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

            # 收集所有多边形
            all_polygons = []

            # 添加凹槽轮廓
            for outline in groove_outlines:
                if len(outline) >= 3:
                    try:
                        poly = Polygon(outline)
                        if poly.is_valid:
                            all_polygons.append(poly)
                        else:
                            # 尝试修复无效多边形
                            poly = poly.buffer(0)
                            if poly.is_valid and not poly.is_empty:
                                all_polygons.append(poly)
                    except:
                        pass

            # 添加IR焊盘
            for pad in ir_pads:
                if len(pad) >= 3:
                    try:
                        poly = Polygon(pad)
                        if poly.is_valid:
                            all_polygons.append(poly)
                        else:
                            poly = poly.buffer(0)
                            if poly.is_valid and not poly.is_empty:
                                all_polygons.append(poly)
                    except:
                        pass

            # 添加中心焊盘
            if center_pad is not None and len(center_pad) >= 3:
                try:
                    poly = Polygon(center_pad)
                    if poly.is_valid:
                        all_polygons.append(poly)
                    else:
                        poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty:
                            all_polygons.append(poly)
                except:
                    pass

            if not all_polygons:
                print("没有有效的轮廓可导出")
                return False

            # 使用布尔并集合并所有多边形
            merged = unary_union(all_polygons)

            # 创建DXF文档
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # 设置单位为毫米
            doc.header['$INSUNITS'] = 4
            doc.header['$MEASUREMENT'] = 1

            # 创建图层
            doc.layers.add('OUTLINE', color=7)  # 白色 - 轮廓

            def add_polygon_to_dxf(poly, layer='OUTLINE'):
                """将shapely多边形添加到DXF"""
                if poly.is_empty:
                    return

                # 外轮廓
                exterior_coords = list(poly.exterior.coords)
                if len(exterior_coords) >= 3:
                    points = [(p[0] * scale, p[1] * scale) for p in exterior_coords]
                    msp.add_lwpolyline(
                        points,
                        close=True,
                        dxfattribs={'layer': layer, 'lineweight': 50}
                    )

                # 内部孔洞（如果有）
                for interior in poly.interiors:
                    interior_coords = list(interior.coords)
                    if len(interior_coords) >= 3:
                        points = [(p[0] * scale, p[1] * scale) for p in interior_coords]
                        msp.add_lwpolyline(
                            points,
                            close=True,
                            dxfattribs={'layer': layer}
                        )

            # 处理合并结果
            if merged.geom_type == 'Polygon':
                add_polygon_to_dxf(merged)
            elif merged.geom_type == 'MultiPolygon':
                for poly in merged.geoms:
                    add_polygon_to_dxf(poly)
            else:
                print(f"不支持的几何类型: {merged.geom_type}")
                return False

            doc.saveas(filepath)
            print(f"已导出单一轮廓DXF: {filepath}")
            return True

        except ImportError:
            print("需要shapely库来合并轮廓，请安装: pip install shapely")
            return False
        except Exception as e:
            print(f"导出单一轮廓DXF失败: {e}")
            import traceback
            traceback.print_exc()
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

    @staticmethod
    def export_coordinates(
        filepath: str,
        points_data: Dict[str, Dict],
        origin_point_id: Optional[str] = None,
        axis_rotation: Optional[List[float]] = None,
        coordinate_system: str = "world"
    ) -> bool:
        """
        导出点位坐标文件

        Args:
            filepath: 保存路径
            points_data: 点位数据 {point_id: {name, position, is_center, pad_length, pad_width}}
            origin_point_id: 原点的点ID（如果设置了局部坐标系）
            axis_rotation: 轴旋转角度 [roll, pitch, yaw]（度）
            coordinate_system: 坐标系类型 "world" 或 "local"

        Returns:
            是否成功
        """
        try:
            # 计算旋转矩阵
            R = np.eye(3)
            if axis_rotation is not None and coordinate_system == "local":
                roll, pitch, yaw = np.radians(axis_rotation)
                cos_r, sin_r = np.cos(roll), np.sin(roll)
                cos_p, sin_p = np.cos(pitch), np.sin(pitch)
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)

                Rx = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
                Ry = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
                Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
                R = (Rz @ Ry @ Rx).T  # 逆旋转

            # 获取原点
            origin = np.zeros(3)
            if origin_point_id and origin_point_id in points_data:
                origin = np.array(points_data[origin_point_id]['position'])

            # 构建坐标数据
            coord_data = {
                'coordinate_system': coordinate_system,
                'origin_point_id': origin_point_id,
                'origin_position': origin.tolist() if origin_point_id else None,
                'axis_rotation': axis_rotation,
                'points': []
            }

            for point_id, info in points_data.items():
                pos = np.array(info['position'])

                # 转换为局部坐标
                if coordinate_system == "local" and origin_point_id:
                    local_pos = R @ (pos - origin)
                else:
                    local_pos = pos

                coord_data['points'].append({
                    'id': point_id,
                    'name': info.get('name', point_id),
                    'is_center': info.get('is_center', False),
                    'is_origin': point_id == origin_point_id,
                    'world_position': pos.tolist(),
                    'local_position': local_pos.tolist() if coordinate_system == "local" else None,
                    'pad_length': info.get('pad_length', 3.0),
                    'pad_width': info.get('pad_width', 2.0)
                })

            # 保存为JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(coord_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"导出坐标文件失败: {e}")
            return False

    @staticmethod
    def export_coordinates_csv(
        filepath: str,
        points_data: Dict[str, Dict],
        origin_point_id: Optional[str] = None,
        axis_rotation: Optional[List[float]] = None,
        coordinate_system: str = "world"
    ) -> bool:
        """
        导出点位坐标为CSV格式

        Args:
            filepath: 保存路径
            points_data: 点位数据
            origin_point_id: 原点的点ID
            axis_rotation: 轴旋转角度
            coordinate_system: 坐标系类型

        Returns:
            是否成功
        """
        try:
            # 计算旋转矩阵
            R = np.eye(3)
            if axis_rotation is not None and coordinate_system == "local":
                roll, pitch, yaw = np.radians(axis_rotation)
                cos_r, sin_r = np.cos(roll), np.sin(roll)
                cos_p, sin_p = np.cos(pitch), np.sin(pitch)
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)

                Rx = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
                Ry = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
                Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
                R = (Rz @ Ry @ Rx).T

            # 获取原点
            origin = np.zeros(3)
            if origin_point_id and origin_point_id in points_data:
                origin = np.array(points_data[origin_point_id]['position'])

            lines = []
            if coordinate_system == "local":
                lines.append("名称,ID,X,Y,Z,世界X,世界Y,世界Z,是中心点,是原点,焊盘长度,焊盘宽度")
            else:
                lines.append("名称,ID,X,Y,Z,是中心点,是原点,焊盘长度,焊盘宽度")

            for point_id, info in points_data.items():
                pos = np.array(info['position'])
                name = info.get('name', point_id)
                is_center = "是" if info.get('is_center', False) else "否"
                is_origin = "是" if point_id == origin_point_id else "否"
                pad_l = info.get('pad_length', 3.0)
                pad_w = info.get('pad_width', 2.0)

                if coordinate_system == "local" and origin_point_id:
                    local_pos = R @ (pos - origin)
                    lines.append(
                        f"{name},{point_id},{local_pos[0]:.4f},{local_pos[1]:.4f},{local_pos[2]:.4f},"
                        f"{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},{is_center},{is_origin},{pad_l},{pad_w}"
                    )
                else:
                    lines.append(
                        f"{name},{point_id},{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},"
                        f"{is_center},{is_origin},{pad_l},{pad_w}"
                    )

            with open(filepath, 'w', encoding='utf-8-sig') as f:
                f.write('\n'.join(lines))

            return True

        except Exception as e:
            print(f"导出坐标CSV失败: {e}")
            return False


class ProjectSnapshot:
    """项目快照 - 完整保存和恢复项目状态"""

    VERSION = "2.0"

    @staticmethod
    def save_snapshot(
        filepath: str,
        mesh_path: str,
        ir_points_data: Dict,
        groove_params: Dict,
        fpc_params: Dict,
        coord_system: Dict,
        paths_data: Optional[Dict] = None,
        additional_data: Optional[Dict] = None
    ) -> bool:
        """
        保存项目快照

        Args:
            filepath: 保存路径（.irproj文件）
            mesh_path: 原始网格文件路径
            ir_points_data: IR点数据
            groove_params: 凹槽参数
            fpc_params: FPC参数
            coord_system: 坐标系统设置
            paths_data: 路径数据（可选）
            additional_data: 其他数据（可选）

        Returns:
            是否成功
        """
        try:
            import shutil
            import zipfile
            from datetime import datetime

            # 创建临时目录
            base_path = Path(filepath)
            temp_dir = base_path.parent / f".{base_path.stem}_temp"
            temp_dir.mkdir(exist_ok=True)

            # 复制网格文件
            mesh_file = Path(mesh_path)
            if mesh_file.exists():
                mesh_dest = temp_dir / f"mesh{mesh_file.suffix}"
                shutil.copy2(mesh_path, mesh_dest)
                relative_mesh_path = mesh_dest.name
            else:
                relative_mesh_path = None

            # 序列化IR点位置（numpy数组转列表）
            serialized_points = {}
            for point_id, info in ir_points_data.items():
                serialized_info = info.copy()
                if 'position' in serialized_info and hasattr(serialized_info['position'], 'tolist'):
                    serialized_info['position'] = serialized_info['position'].tolist()
                serialized_points[point_id] = serialized_info

            # 创建项目数据
            project_data = {
                'version': ProjectSnapshot.VERSION,
                'created_at': datetime.now().isoformat(),
                'mesh_path': relative_mesh_path,
                'original_mesh_path': mesh_path,
                'ir_points': serialized_points,
                'groove_params': groove_params,
                'fpc_params': fpc_params,
                'coord_system': coord_system,
            }

            if paths_data:
                # 序列化路径数据
                serialized_paths = {}
                for path_id, path in paths_data.items():
                    if hasattr(path, 'tolist'):
                        serialized_paths[path_id] = path.tolist()
                    else:
                        serialized_paths[path_id] = path
                project_data['paths'] = serialized_paths

            if additional_data:
                project_data['additional'] = additional_data

            # 保存项目JSON
            project_json_path = temp_dir / "project.json"
            with open(project_json_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            # 打包为zip文件
            if filepath.endswith('.irproj'):
                zip_path = filepath
            else:
                zip_path = filepath + '.irproj'

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in temp_dir.iterdir():
                    zf.write(file, file.name)

            # 清理临时目录
            shutil.rmtree(temp_dir)

            return True

        except Exception as e:
            print(f"保存项目快照失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def load_snapshot(filepath: str, extract_to: Optional[str] = None) -> Optional[Dict]:
        """
        加载项目快照

        Args:
            filepath: 快照文件路径
            extract_to: 解压目录（可选，默认为临时目录）

        Returns:
            项目数据字典，包含：
            - version: 版本号
            - mesh_path: 网格文件路径（解压后的绝对路径）
            - ir_points: IR点数据
            - groove_params: 凹槽参数
            - fpc_params: FPC参数
            - coord_system: 坐标系统设置
            - paths: 路径数据（如果有）
        """
        try:
            import zipfile
            import tempfile

            if not Path(filepath).exists():
                print(f"文件不存在: {filepath}")
                return None

            # 确定解压目录
            if extract_to:
                extract_dir = Path(extract_to)
            else:
                extract_dir = Path(tempfile.mkdtemp(prefix="irproj_"))

            extract_dir.mkdir(parents=True, exist_ok=True)

            # 解压
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(extract_dir)

            # 读取项目JSON
            project_json_path = extract_dir / "project.json"
            if not project_json_path.exists():
                print("无效的项目文件：缺少project.json")
                return None

            with open(project_json_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            # 更新网格路径为绝对路径
            if project_data.get('mesh_path'):
                mesh_path = extract_dir / project_data['mesh_path']
                if mesh_path.exists():
                    project_data['mesh_path'] = str(mesh_path)
                elif project_data.get('original_mesh_path'):
                    # 尝试使用原始路径
                    if Path(project_data['original_mesh_path']).exists():
                        project_data['mesh_path'] = project_data['original_mesh_path']

            # 转换IR点位置回numpy数组
            for point_id, info in project_data.get('ir_points', {}).items():
                if 'position' in info and isinstance(info['position'], list):
                    info['position'] = np.array(info['position'])

            # 转换路径数据回numpy数组
            if 'paths' in project_data:
                for path_id, path in project_data['paths'].items():
                    if isinstance(path, list):
                        project_data['paths'][path_id] = np.array(path)

            project_data['_extract_dir'] = str(extract_dir)

            return project_data

        except Exception as e:
            print(f"加载项目快照失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class FolderExporter:
    """打包导出到文件夹"""

    def __init__(self, output_dir: str, project_name: str = "ir_array_export"):
        """
        初始化文件夹导出器

        Args:
            output_dir: 输出目录
            project_name: 项目名称（将创建同名子文件夹）
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir) / project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.mesh_dir = self.output_dir / "meshes"
        self.drawing_dir = self.output_dir / "drawings"
        self.coord_dir = self.output_dir / "coordinates"
        self.config_dir = self.output_dir / "config"

        for d in [self.mesh_dir, self.drawing_dir, self.coord_dir, self.config_dir]:
            d.mkdir(exist_ok=True)

    def export_all(
        self,
        mesh: trimesh.Trimesh,
        mesh_with_grooves: Optional[trimesh.Trimesh],
        paths_2d: List[np.ndarray],
        ir_points_2d: List[Tuple[np.ndarray, str]],
        ir_points_data: Dict,
        groove_params: Dict,
        fpc_params: Dict,
        coord_system: Dict,
        fpc_layout: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        导出所有文件到文件夹

        Returns:
            导出文件路径字典
        """
        exported = {}
        name = self.project_name

        # === 网格文件 ===
        # 原始网格
        mesh_path = self.mesh_dir / f"{name}_original.stl"
        if Exporter.export_mesh(mesh, str(mesh_path)):
            exported['original_mesh'] = str(mesh_path)

        # 带凹槽的网格
        if mesh_with_grooves is not None:
            groove_mesh_path = self.mesh_dir / f"{name}_with_grooves.stl"
            if Exporter.export_mesh(mesh_with_grooves, str(groove_mesh_path)):
                exported['groove_mesh'] = str(groove_mesh_path)

        # === 图纸文件 ===
        # 2D路径 DXF
        dxf_path = self.drawing_dir / f"{name}_2d_paths.dxf"
        if Exporter.export_dxf(paths_2d, ir_points_2d, str(dxf_path)):
            exported['dxf'] = str(dxf_path)

        # 2D路径 SVG
        svg_path = self.drawing_dir / f"{name}_2d_paths.svg"
        if Exporter.export_svg(paths_2d, ir_points_2d, str(svg_path)):
            exported['svg'] = str(svg_path)

        # FPC布局图
        if fpc_layout is not None:
            groove_outlines = fpc_layout.get('groove_outlines', [])
            ir_pads = fpc_layout.get('ir_pads', [])
            center_pad = fpc_layout.get('center_pad')
            merged_outline = fpc_layout.get('merged_outline')

            if groove_outlines:
                fpc_dxf_path = self.drawing_dir / f"{name}_fpc_layout.dxf"
                if Exporter.export_fpc_dxf(
                    groove_outlines, ir_pads, center_pad, merged_outline,
                    ir_points_2d, str(fpc_dxf_path)
                ):
                    exported['fpc_dxf'] = str(fpc_dxf_path)

                fpc_svg_path = self.drawing_dir / f"{name}_fpc_layout.svg"
                if Exporter.export_fpc_svg(
                    groove_outlines, ir_pads, center_pad, merged_outline,
                    ir_points_2d, str(fpc_svg_path)
                ):
                    exported['fpc_svg'] = str(fpc_svg_path)

        # === 坐标文件 ===
        origin_id = coord_system.get('origin_point_id')
        axis_rot = coord_system.get('axis_rotation')

        # 世界坐标JSON
        world_coord_path = self.coord_dir / f"{name}_world_coordinates.json"
        if BatchExporter.export_coordinates(
            str(world_coord_path), ir_points_data,
            coordinate_system="world"
        ):
            exported['world_coord_json'] = str(world_coord_path)

        # 世界坐标CSV
        world_csv_path = self.coord_dir / f"{name}_world_coordinates.csv"
        if BatchExporter.export_coordinates_csv(
            str(world_csv_path), ir_points_data,
            coordinate_system="world"
        ):
            exported['world_coord_csv'] = str(world_csv_path)

        # 局部坐标（如果设置了原点）
        if origin_id:
            local_coord_path = self.coord_dir / f"{name}_local_coordinates.json"
            if BatchExporter.export_coordinates(
                str(local_coord_path), ir_points_data,
                origin_point_id=origin_id,
                axis_rotation=axis_rot,
                coordinate_system="local"
            ):
                exported['local_coord_json'] = str(local_coord_path)

            local_csv_path = self.coord_dir / f"{name}_local_coordinates.csv"
            if BatchExporter.export_coordinates_csv(
                str(local_csv_path), ir_points_data,
                origin_point_id=origin_id,
                axis_rotation=axis_rot,
                coordinate_system="local"
            ):
                exported['local_coord_csv'] = str(local_csv_path)

        # === 配置文件 ===
        config_path = self.config_dir / f"{name}_config.json"
        config_data = {
            'groove_params': groove_params,
            'fpc_params': fpc_params,
            'coord_system': coord_system
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        exported['config'] = str(config_path)

        # 生成README
        readme_path = self.output_dir / "README.txt"
        readme_content = f"""IR阵列定位辅助生成工具 - 导出文件说明
========================================

项目名称: {name}
导出时间: {Path(readme_path).stat().st_mtime if readme_path.exists() else 'N/A'}

目录结构:
---------
meshes/          - 3D网格文件
  ├── *_original.stl      - 原始网格
  └── *_with_grooves.stl  - 带凹槽的网格

drawings/        - 2D图纸文件
  ├── *_2d_paths.dxf      - 2D路径DXF
  ├── *_2d_paths.svg      - 2D路径SVG
  ├── *_fpc_layout.dxf    - FPC布局DXF
  └── *_fpc_layout.svg    - FPC布局SVG

coordinates/     - 坐标文件
  ├── *_world_coordinates.json  - 世界坐标(JSON)
  ├── *_world_coordinates.csv   - 世界坐标(CSV)
  ├── *_local_coordinates.json  - 局部坐标(JSON)
  └── *_local_coordinates.csv   - 局部坐标(CSV)

config/          - 配置文件
  └── *_config.json       - 参数配置

导出文件列表:
{chr(10).join(f'  - {k}: {v}' for k, v in exported.items())}
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        exported['readme'] = str(readme_path)
        exported['output_dir'] = str(self.output_dir)

        return exported
