"""3D模型加载模块"""
import trimesh
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class MeshLoader:
    """网格加载器，支持STL、3MF、OBJ等格式"""

    SUPPORTED_FORMATS = {'.stl', '.3mf', '.obj', '.ply', '.off'}

    # 单位转换因子到毫米
    UNIT_FACTORS = {
        'm': 1000.0,      # 米 -> 毫米
        'cm': 10.0,       # 厘米 -> 毫米
        'mm': 1.0,        # 毫米 -> 毫米
        'inch': 25.4,     # 英寸 -> 毫米
        'ft': 304.8,      # 英尺 -> 毫米
    }

    @classmethod
    def load(cls, filepath: str, auto_convert_units: bool = True) -> Optional[trimesh.Trimesh]:
        """
        加载3D模型文件

        Args:
            filepath: 文件路径
            auto_convert_units: 是否自动检测并转换单位到毫米

        Returns:
            trimesh.Trimesh对象，加载失败返回None
        """
        path = Path(filepath)

        if not path.exists():
            print(f"文件不存在: {filepath}")
            return None

        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_FORMATS:
            print(f"不支持的格式: {suffix}")
            return None

        try:
            mesh = trimesh.load(filepath, force='mesh')

            # 如果加载的是场景，提取第一个网格
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    mesh = list(mesh.geometry.values())[0]
                else:
                    print("场景中没有几何体")
                    return None

            # 验证网格
            if not isinstance(mesh, trimesh.Trimesh):
                print("无法解析为三角网格")
                return None

            # 修复常见问题
            mesh = cls._repair_mesh(mesh)

            # 自动单位转换
            if auto_convert_units:
                detected_unit, factor = cls.detect_unit(mesh)
                if factor != 1.0:
                    print(f"检测到模型可能使用 {detected_unit} 单位，自动转换到毫米 (×{factor})")
                    mesh.vertices *= factor

            return mesh

        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

    @classmethod
    def detect_unit(cls, mesh: trimesh.Trimesh) -> Tuple[str, float]:
        """
        检测模型可能使用的单位

        基于模型尺寸的启发式检测：
        - 典型的IR阵列产品尺寸约为 20-200mm
        - 如果模型尺寸 < 1，可能是米
        - 如果模型尺寸 < 10，可能是厘米或英寸
        - 如果模型尺寸 10-500，可能是毫米
        - 如果模型尺寸 > 500，可能已经是毫米

        Returns:
            (检测到的单位, 转换因子到毫米)
        """
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        max_dim = np.max(size)

        print(f"模型原始尺寸: {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}")
        print(f"最大维度: {max_dim:.4f}")

        # 启发式判断
        if max_dim < 0.5:
            # 尺寸很小，很可能是米
            return ('m', 1000.0)
        elif max_dim < 5:
            # 可能是厘米或英寸，假设厘米更常见
            return ('cm', 10.0)
        elif max_dim < 25.4:
            # 可能是英寸 (1 inch = 25.4mm)
            # 但也可能是小型毫米模型，需要更多判断
            # 如果尺寸在1-25之间，可能是毫米的小模型
            if max_dim >= 10:
                return ('mm', 1.0)
            else:
                # 5-10之间比较模糊，假设是厘米
                return ('cm', 10.0)
        else:
            # 尺寸较大，很可能已经是毫米
            return ('mm', 1.0)

    @classmethod
    def _repair_mesh(cls, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """修复网格常见问题"""
        # 移除重复顶点
        mesh.merge_vertices()

        # 移除退化三角形（兼容新旧版本）
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            # 新版本使用 update_faces 过滤退化面
            # 退化面是面积为0的三角形
            face_mask = mesh.area_faces > 1e-10
            if not face_mask.all():
                mesh.update_faces(face_mask)

        # 修复法向量
        mesh.fix_normals()

        return mesh

    @classmethod
    def get_mesh_info(cls, mesh: trimesh.Trimesh) -> dict:
        """获取网格信息"""
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'bounds': mesh.bounds.tolist(),
            'centroid': mesh.centroid.tolist(),
            'is_watertight': mesh.is_watertight,
            'volume': mesh.volume if mesh.is_watertight else None,
            'area': mesh.area,
        }

    @classmethod
    def save_mesh(
        cls,
        mesh: trimesh.Trimesh,
        filepath: str,
        file_type: Optional[str] = None
    ) -> bool:
        """
        保存网格到文件

        Args:
            mesh: 要保存的网格
            filepath: 保存路径
            file_type: 文件类型，默认从扩展名推断

        Returns:
            是否保存成功
        """
        try:
            mesh.export(filepath, file_type=file_type)
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
