"""核心模块"""
from .mesh_loader import MeshLoader
from .geodesic import GeodesicSolver, PathManager, IRPoint
from .groove_gen import GrooveGenerator, GrooveParams
from .conformal_map import ConformalMapper, FlattenResult
from .exporter import Exporter, BatchExporter
