# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IR Array Layout Assistance Generation Tool (IR阵列定位辅助生成工具) - A PyQt5 desktop application for placing infrared sensor points on 3D curved surface models, computing geodesic paths, generating FPC routing grooves, and flattening surfaces to 2D for flexible circuit board manufacturing.

## Common Commands

```bash
# Run application (auto-creates venv and installs dependencies)
./run.sh

# Direct run (requires venv activated)
python main.py

# Install dependencies manually
pip install -r requirements.txt
```

## Architecture

**Pattern:** MVC-inspired with PyQt5 signal/slot pattern

**Core Processing Pipeline:**
1. Load 3D mesh (trimesh) → `core/mesh_loader.py`
2. Pick IR points on surface → `ui/viewer_3d.py`
3. Compute geodesic paths (Dijkstra) → `core/geodesic.py`
4. Plan 2D paths on surface → `core/path_planner_2d.py`
5. Generate FPC grooves → `core/groove_gen.py`
6. Flatten surface (LSCM conformal mapping) → `core/conformal_map.py`
7. Export (DXF/SVG/STL/JSON) → `core/exporter.py`

**Key Classes:**
- `MainWindow` (ui/main_window.py) - Orchestrator connecting all UI and core components
- `PathManager` / `GeodesicSolver` (core/geodesic.py) - IR point management and Dijkstra path computation
- `ConformalMapper` (core/conformal_map.py) - LSCM surface unwrapping with sparse matrix solving
- `GrooveGenerator` (core/groove_gen.py) - FPC groove geometry with boolean operations

**UI Components:**
- `Viewer3D` (ui/viewer_3d.py) - PyVista/VTK 3D visualization with interactive picking
- `View2D` (ui/view_2d.py) - 2D unwrapped surface display
- `PointPanel` (ui/point_panel.py) - IR point list and management
- `ParamPanel` (ui/param_panel.py) - Groove/flatten/export parameters

## Key Technical Details

- **Mesh formats:** STL, 3MF, OBJ, PLY, OFF (via trimesh with auto-repair)
- **Geodesic algorithm:** Dijkstra on triangular mesh edges
- **Surface flattening:** LSCM (Least Squares Conformal Mapping) preserves local angles
- **Boolean operations:** Uses manifold3d (or Blender as fallback) for groove cutting
- **Path smoothing:** B-spline interpolation with surface projection

## Dependencies

Core: PyQt5, PyVista, trimesh, numpy, scipy, networkx, manifold3d, ezdxf
