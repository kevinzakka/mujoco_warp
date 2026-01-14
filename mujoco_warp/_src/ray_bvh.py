# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BVH-accelerated ray casting for mujoco_warp.

This module provides BVH (Bounding Volume Hierarchy) acceleration for ray casting
queries, replacing the brute-force O(rays * geoms) approach with O(rays * log(geoms)).

Usage:
    import mujoco_warp as mjw

    # 1. Build BVH once at initialization
    ctx = mjw.build_ray_bvh(m, d)

    # 2. After each physics step, refit bounds (cheaper than rebuilding)
    mjw.refit_ray_bvh(m, d, ctx)

    # 3. Cast rays using BVH
    mjw.rays_bvh(m, d, ctx, pnt, vec, geomgroup, flg_static,
                 bodyexclude, dist, geomid, normal)

The BVH uses a two-level hierarchy:
  - World-level BVH: Groups geometries by world for efficient multi-world queries
  - Per-mesh/heightfield BVHs: Accelerates narrow-phase triangle intersection

Note:
    If geometries are added or removed, you must rebuild the BVH with
    build_ray_bvh(). The refit operation only updates bounds for existing
    geometries.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import warp as wp

from .ray import _ray_eliminate
from .ray import _ray_hfield
from .ray import _ray_map
from .ray import ray_geom
from .ray import ray_mesh
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec6

wp.set_module_options({"enable_backward": False})

# =============================================================================
# Constants
# =============================================================================

# Fallback size for planes with zero/negative size (effectively infinite)
_PLANE_BOUNDS_FALLBACK: float = 1000.0

# Padding for plane AABBs to avoid numerical precision issues
_PLANE_BOUNDS_PADDING: float = 0.01


@dataclass
class RayBvhContext:
  """Context for BVH-accelerated ray casting.

  This context stores the BVH data structures needed for efficient ray queries.
  It should be created once via `build_ray_bvh()` and updated each physics step
  via `refit_ray_bvh()`.
  """

  bvh: wp.Bvh
  bvh_id: wp.uint64
  lower: wp.array  # dtype=wp.vec3, shape=(nworld * ngeom,)
  upper: wp.array  # dtype=wp.vec3, shape=(nworld * ngeom,)
  group: wp.array  # dtype=int, shape=(nworld * ngeom,)
  group_root: wp.array  # dtype=int, shape=(nworld,)
  ngeom: int
  enabled_geom_ids: wp.array  # dtype=int, shape=(ngeom,) - maps BVH index to geom ID
  mesh_bvh_id: wp.array  # dtype=wp.uint64, shape=(nmesh,) - per-mesh BVH IDs
  mesh_bounds_size: wp.array  # dtype=wp.vec3, shape=(nmesh,) - half-extents
  hfield_bvh_id: wp.array  # dtype=wp.uint64, shape=(nhfield,) - per-hfield BVH IDs
  hfield_bounds_size: wp.array  # dtype=wp.vec3, shape=(nhfield,) - half-extents
  mesh_registry: dict  # Keep references to prevent GC
  hfield_registry: dict


# =============================================================================
# AABB Computation Functions
# =============================================================================


@wp.func
def _compute_box_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for a box geometry."""
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        local_corner = wp.vec3(
          size[0] * (2.0 * float(i) - 1.0),
          size[1] * (2.0 * float(j) - 1.0),
          size[2] * (2.0 * float(k) - 1.0),
        )
        world_corner = pos + rot @ local_corner
        min_bound = wp.min(min_bound, world_corner)
        max_bound = wp.max(max_bound, world_corner)

  return min_bound, max_bound


@wp.func
def _compute_sphere_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for a sphere geometry."""
  radius = size[0]
  return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)


@wp.func
def _compute_capsule_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for a capsule geometry."""
  radius = size[0]
  half_length = size[1]
  local_end1 = wp.vec3(0.0, 0.0, -half_length)
  local_end2 = wp.vec3(0.0, 0.0, half_length)
  world_end1 = pos + rot @ local_end1
  world_end2 = pos + rot @ local_end2

  seg_min = wp.min(world_end1, world_end2)
  seg_max = wp.max(world_end1, world_end2)

  inflate = wp.vec3(radius, radius, radius)
  return seg_min - inflate, seg_max + inflate


@wp.func
def _compute_plane_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for a plane geometry."""
  size_scale = wp.max(size[0], size[1]) * 2.0
  if size[0] <= 0.0 or size[1] <= 0.0:
    size_scale = _PLANE_BOUNDS_FALLBACK
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      local_corner = wp.vec3(
        size_scale * (2.0 * float(i) - 1.0),
        size_scale * (2.0 * float(j) - 1.0),
        0.0,
      )
      world_corner = pos + rot @ local_corner
      min_bound = wp.min(min_bound, world_corner)
      max_bound = wp.max(max_bound, world_corner)

  min_bound = min_bound - wp.vec3(_PLANE_BOUNDS_PADDING, _PLANE_BOUNDS_PADDING, _PLANE_BOUNDS_PADDING)
  max_bound = max_bound + wp.vec3(_PLANE_BOUNDS_PADDING, _PLANE_BOUNDS_PADDING, _PLANE_BOUNDS_PADDING)

  return min_bound, max_bound


@wp.func
def _compute_ellipsoid_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for an ellipsoid geometry."""
  row0 = wp.vec3(rot[0, 0] * size[0], rot[0, 1] * size[1], rot[0, 2] * size[2])
  row1 = wp.vec3(rot[1, 0] * size[0], rot[1, 1] * size[1], rot[1, 2] * size[2])
  row2 = wp.vec3(rot[2, 0] * size[0], rot[2, 1] * size[1], rot[2, 2] * size[2])
  extent = wp.vec3(wp.length(row0), wp.length(row1), wp.length(row2))
  return pos - extent, pos + extent


@wp.func
def _compute_cylinder_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  """Compute world-space AABB for a cylinder geometry."""
  radius = size[0]
  half_height = size[1]

  axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
  axis_abs = wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2]))

  basis_x = wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0])
  basis_y = wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1])

  radial_x = radius * wp.sqrt(basis_x[0] * basis_x[0] + basis_y[0] * basis_y[0])
  radial_y = radius * wp.sqrt(basis_x[1] * basis_x[1] + basis_y[1] * basis_y[1])
  radial_z = radius * wp.sqrt(basis_x[2] * basis_x[2] + basis_y[2] * basis_y[2])

  extent = wp.vec3(
    radial_x + half_height * axis_abs[0],
    radial_y + half_height * axis_abs[1],
    radial_z + half_height * axis_abs[2],
  )

  return pos - extent, pos + extent


@wp.kernel
def _compute_bvh_bounds(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  nworld_in: int,
  # In:
  ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bounds_size: wp.array(dtype=wp.vec3),
  hfield_bounds_size: wp.array(dtype=wp.vec3),
  # Out:
  lower_out: wp.array(dtype=wp.vec3),
  upper_out: wp.array(dtype=wp.vec3),
  group_out: wp.array(dtype=int),
):
  """Compute AABBs for all geometries in all worlds."""
  tid = wp.tid()
  world_id = tid // ngeom
  bvh_geom_local = tid % ngeom

  if bvh_geom_local >= ngeom or world_id >= nworld_in:
    return

  geom_id = enabled_geom_ids[bvh_geom_local]

  pos = geom_xpos_in[world_id, geom_id]
  rot = geom_xmat_in[world_id, geom_id]
  size = geom_size[world_id, geom_id]
  gtype = geom_type[geom_id]

  # Initialize to degenerate bounds (will never be hit if geom type unrecognized)
  lower_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  upper_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  if gtype == GeomType.SPHERE:
    lower_bound, upper_bound = _compute_sphere_bounds(pos, rot, size)
  elif gtype == GeomType.CAPSULE:
    lower_bound, upper_bound = _compute_capsule_bounds(pos, rot, size)
  elif gtype == GeomType.PLANE:
    lower_bound, upper_bound = _compute_plane_bounds(pos, rot, size)
  elif gtype == GeomType.MESH:
    size = mesh_bounds_size[geom_dataid[geom_id]]
    lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)
  elif gtype == GeomType.ELLIPSOID:
    lower_bound, upper_bound = _compute_ellipsoid_bounds(pos, rot, size)
  elif gtype == GeomType.CYLINDER:
    lower_bound, upper_bound = _compute_cylinder_bounds(pos, rot, size)
  elif gtype == GeomType.BOX:
    lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)
  elif gtype == GeomType.HFIELD:
    size = hfield_bounds_size[geom_dataid[geom_id]]
    lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)

  lower_out[world_id * ngeom + bvh_geom_local] = lower_bound
  upper_out[world_id * ngeom + bvh_geom_local] = upper_bound
  group_out[world_id * ngeom + bvh_geom_local] = world_id


@wp.kernel
def _compute_bvh_group_roots(
  bvh_id: wp.uint64,
  group_root_out: wp.array(dtype=int),
):
  """Compute the root node for each world group in the BVH."""
  tid = wp.tid()
  root = wp.bvh_get_group_root(bvh_id, tid)
  group_root_out[tid] = root


# =============================================================================
# Mesh and HField BVH Construction
# =============================================================================


def _build_mesh_bvh(
  m: Model,
  mesh_vert_np: np.ndarray,
  mesh_face_np: np.ndarray,
  mesh_vertadr_np: np.ndarray,
  mesh_vertnum_np: np.ndarray,
  mesh_faceadr_np: np.ndarray,
  nmeshface: int,
  meshid: int,
) -> Tuple[wp.Mesh, np.ndarray]:
  """Create a Warp mesh BVH from mesh data."""
  v_start = mesh_vertadr_np[meshid]
  v_end = v_start + mesh_vertnum_np[meshid]
  points = mesh_vert_np[v_start:v_end]

  f_start = mesh_faceadr_np[meshid]
  f_end = nmeshface if (meshid + 1) >= len(mesh_faceadr_np) else mesh_faceadr_np[meshid + 1]
  indices = mesh_face_np[f_start:f_end].flatten()

  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = 0.5 * (pmax - pmin)

  points_wp = wp.array(points, dtype=wp.vec3)
  indices_wp = wp.array(indices, dtype=wp.int32)
  mesh = wp.Mesh(points=points_wp, indices=indices_wp, bvh_constructor="sah")

  return mesh, half


def _build_hfield_bvh(
  hfield_data_np: np.ndarray,
  hfield_size_np: np.ndarray,
  hfield_nrow_np: np.ndarray,
  hfield_ncol_np: np.ndarray,
  hfield_adr_np: np.ndarray,
  hfieldid: int,
) -> Tuple[wp.Mesh, np.ndarray]:
  """Create a Warp mesh BVH from heightfield data."""
  nr = hfield_nrow_np[hfieldid]
  nc = hfield_ncol_np[hfieldid]
  sz = hfield_size_np[hfieldid]

  adr = hfield_adr_np[hfieldid]
  data = hfield_data_np[adr : adr + nr * nc].reshape((nr, nc))

  sx, sy, sz_scale = sz[0], sz[1], sz[2]
  width = 0.5 * max(nc - 1, 1)
  height = 0.5 * max(nr - 1, 1)

  # Generate mesh from heightfield
  points_list = []
  indices_list = []
  for r in range(nr - 1):
    for c in range(nc - 1):
      # Four corners of the cell
      x0 = sx * (float(c) / width - 1.0)
      x1 = sx * (float(c + 1) / width - 1.0)
      y0 = sy * (float(r) / height - 1.0)
      y1 = sy * (float(r + 1) / height - 1.0)

      z00 = float(data[r, c]) * sz_scale
      z01 = float(data[r, c + 1]) * sz_scale
      z10 = float(data[r + 1, c]) * sz_scale
      z11 = float(data[r + 1, c + 1]) * sz_scale

      base_idx = len(points_list)
      points_list.extend(
        [
          [x0, y0, z00],
          [x1, y0, z01],
          [x0, y1, z10],
          [x1, y1, z11],
        ]
      )
      # Two triangles per cell
      indices_list.extend([base_idx, base_idx + 1, base_idx + 3])
      indices_list.extend([base_idx, base_idx + 3, base_idx + 2])

  points = np.array(points_list, dtype=np.float32)
  indices = np.array(indices_list, dtype=np.int32)

  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = 0.5 * (pmax - pmin)

  points_wp = wp.array(points, dtype=wp.vec3)
  indices_wp = wp.array(indices, dtype=wp.int32)
  mesh = wp.Mesh(points=points_wp, indices=indices_wp, bvh_constructor="sah")

  return mesh, half


# =============================================================================
# BVH Lifecycle Functions
# =============================================================================


def build_ray_bvh(
  m: Model,
  d: Data,
  enabled_geom_groups: Optional[list[int]] = None,
) -> RayBvhContext:
  """Build a BVH for ray casting.

  This function constructs a BVH over all geometries for efficient ray queries.
  Call this once during initialization, then use `refit_ray_bvh()` each physics
  step to update the bounds.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state (device).
    enabled_geom_groups: List of geom groups to include (default: all).

  Returns:
    RayBvhContext containing the BVH data structures.
  """
  # Get numpy arrays for CPU-side operations
  geom_type_np = m.geom_type.numpy()
  geom_dataid_np = m.geom_dataid.numpy()

  # Filter geometries by group if specified
  if enabled_geom_groups is not None:
    geom_group_np = m.geom_group.numpy()
    enabled_geom_ids = [i for i in range(m.ngeom) if geom_group_np[i] in enabled_geom_groups]
  else:
    enabled_geom_ids = list(range(m.ngeom))

  ngeom = len(enabled_geom_ids)

  # Build per-mesh BVHs
  mesh_registry = {}
  mesh_bvh_id = [wp.uint64(0)] * m.nmesh
  mesh_bounds_size = [np.zeros(3, dtype=np.float32) for _ in range(m.nmesh)]

  if m.nmesh > 0:
    mesh_vert_np = m.mesh_vert.numpy()
    mesh_face_np = m.mesh_face.numpy()
    mesh_vertadr_np = m.mesh_vertadr.numpy()
    mesh_vertnum_np = m.mesh_vertnum.numpy()
    mesh_faceadr_np = m.mesh_faceadr.numpy()

    used_mesh_ids = set(
      int(geom_dataid_np[g]) for g in enabled_geom_ids if geom_type_np[g] == GeomType.MESH and int(geom_dataid_np[g]) >= 0
    )

    for meshid in used_mesh_ids:
      mesh, half = _build_mesh_bvh(
        m, mesh_vert_np, mesh_face_np, mesh_vertadr_np, mesh_vertnum_np, mesh_faceadr_np, m.nmeshface, meshid
      )
      mesh_registry[mesh.id] = mesh
      mesh_bvh_id[meshid] = mesh.id
      mesh_bounds_size[meshid] = half

  # Build per-hfield BVHs
  hfield_registry = {}
  hfield_bvh_id = [wp.uint64(0)] * m.nhfield
  hfield_bounds_size = [np.zeros(3, dtype=np.float32) for _ in range(m.nhfield)]

  if m.nhfield > 0:
    hfield_data_np = m.hfield_data.numpy()
    hfield_size_np = m.hfield_size.numpy()
    hfield_nrow_np = m.hfield_nrow.numpy()
    hfield_ncol_np = m.hfield_ncol.numpy()
    hfield_adr_np = m.hfield_adr.numpy()

    used_hfield_ids = set(
      int(geom_dataid_np[g]) for g in enabled_geom_ids if geom_type_np[g] == GeomType.HFIELD and int(geom_dataid_np[g]) >= 0
    )

    for hfieldid in used_hfield_ids:
      mesh, half = _build_hfield_bvh(hfield_data_np, hfield_size_np, hfield_nrow_np, hfield_ncol_np, hfield_adr_np, hfieldid)
      hfield_registry[mesh.id] = mesh
      hfield_bvh_id[hfieldid] = mesh.id
      hfield_bounds_size[hfieldid] = half

  # Allocate BVH arrays
  lower = wp.zeros(d.nworld * ngeom, dtype=wp.vec3)
  upper = wp.zeros(d.nworld * ngeom, dtype=wp.vec3)
  group = wp.zeros(d.nworld * ngeom, dtype=int)
  group_root = wp.zeros(d.nworld, dtype=int)
  enabled_geom_ids_wp = wp.array(enabled_geom_ids, dtype=int)
  mesh_bvh_id_wp = wp.array(mesh_bvh_id, dtype=wp.uint64)
  mesh_bounds_size_wp = wp.array(mesh_bounds_size, dtype=wp.vec3)
  hfield_bvh_id_wp = wp.array(hfield_bvh_id, dtype=wp.uint64)
  hfield_bounds_size_wp = wp.array(hfield_bounds_size, dtype=wp.vec3)

  # Compute initial bounds
  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=d.nworld * ngeom,
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      d.nworld,
      ngeom,
      enabled_geom_ids_wp,
      mesh_bounds_size_wp,
      hfield_bounds_size_wp,
      lower,
      upper,
      group,
    ],
  )

  # Build BVH with SAH constructor
  bvh = wp.Bvh(lower, upper, groups=group, constructor="sah")

  # Compute group roots
  wp.launch(
    kernel=_compute_bvh_group_roots,
    dim=d.nworld,
    inputs=[bvh.id],
    outputs=[group_root],
  )

  return RayBvhContext(
    bvh=bvh,
    bvh_id=bvh.id,
    lower=lower,
    upper=upper,
    group=group,
    group_root=group_root,
    ngeom=ngeom,
    enabled_geom_ids=enabled_geom_ids_wp,
    mesh_bvh_id=mesh_bvh_id_wp,
    mesh_bounds_size=mesh_bounds_size_wp,
    hfield_bvh_id=hfield_bvh_id_wp,
    hfield_bounds_size=hfield_bounds_size_wp,
    mesh_registry=mesh_registry,
    hfield_registry=hfield_registry,
  )


def refit_ray_bvh(m: Model, d: Data, ctx: RayBvhContext):
  """Refit the BVH after a physics step.

  This function updates the AABB bounds for all geometries based on their
  new positions. It's much cheaper than rebuilding the BVH from scratch
  because it only updates bounds without restructuring the tree.

  This must be called after each physics step (or any operation that moves
  geometries) before casting rays, otherwise rays may miss geometries or
  report incorrect intersections.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state (device).
    ctx: The BVH context to update (modified in-place).

  Note:
    This only updates bounds for geometries that were included when the BVH
    was built. If geometries are added or removed, call build_ray_bvh() instead.
  """
  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=d.nworld * ctx.ngeom,
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      d.nworld,
      ctx.ngeom,
      ctx.enabled_geom_ids,
      ctx.mesh_bounds_size,
      ctx.hfield_bounds_size,
      ctx.lower,
      ctx.upper,
      ctx.group,
    ],
  )

  ctx.bvh.refit()


# =============================================================================
# BVH-Accelerated Ray Kernel
# =============================================================================


@wp.func
def _ray_geom_bvh(
  # Model:
  nmeshface: int,
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_vertadr: wp.array(dtype=int),
  mesh_faceadr: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_face: wp.array(dtype=wp.vec3i),
  hfield_size: wp.array(dtype=wp.vec4),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_adr: wp.array(dtype=int),
  hfield_data: wp.array(dtype=float),
  # BVH context:
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomid: int,
  max_dist: float,
) -> Tuple[float, wp.vec3]:
  """Compute ray-geometry intersection for a single geometry."""
  pos = geom_xpos_in[worldid, geomid]
  mat = geom_xmat_in[worldid, geomid]
  gtype = geom_type[geomid]

  if gtype == GeomType.MESH:
    # Use per-mesh BVH if available, otherwise fall back to brute-force
    mesh_id = geom_dataid[geomid]
    bvh_id = mesh_bvh_id[mesh_id]
    if bvh_id != wp.uint64(0):
      # BVH-accelerated mesh query
      lpnt, lvec = _ray_map(pos, mat, pnt, vec)
      t = float(-1.0)
      u = float(0.0)
      v = float(0.0)
      sign = float(0.0)
      n = wp.vec3(0.0, 0.0, 0.0)
      f = int(-1)
      hit = wp.mesh_query_ray(bvh_id, lpnt, lvec, max_dist, t, u, v, sign, n, f)
      if hit:
        normal = mat @ n
        normal = wp.normalize(normal)
        return t, normal
      return -1.0, wp.vec3()
    else:
      # Fall back to brute-force mesh intersection
      return ray_mesh(
        nmeshface,
        mesh_vertadr,
        mesh_faceadr,
        mesh_vert,
        mesh_face,
        mesh_id,
        pos,
        mat,
        pnt,
        vec,
      )
  elif gtype == GeomType.HFIELD:
    # Use per-hfield BVH if available
    hfield_id = geom_dataid[geomid]
    bvh_id = hfield_bvh_id[hfield_id]
    if bvh_id != wp.uint64(0):
      # BVH-accelerated hfield query
      lpnt, lvec = _ray_map(pos, mat, pnt, vec)
      t = float(-1.0)
      u = float(0.0)
      v = float(0.0)
      sign = float(0.0)
      n = wp.vec3(0.0, 0.0, 0.0)
      f = int(-1)
      hit = wp.mesh_query_ray(bvh_id, lpnt, lvec, max_dist, t, u, v, sign, n, f)
      if hit:
        normal = mat @ n
        normal = wp.normalize(normal)
        return t, normal
      return -1.0, wp.vec3()
    else:
      # Fall back to brute-force hfield intersection
      return _ray_hfield(
        geom_type,
        geom_dataid,
        hfield_size,
        hfield_nrow,
        hfield_ncol,
        hfield_adr,
        hfield_data,
        pos,
        mat,
        pnt,
        vec,
        geomid,
      )
  else:
    # Use analytic ray-geometry test
    return ray_geom(pos, mat, geom_size[worldid % geom_size.shape[0], geomid], pnt, vec, gtype)


@wp.kernel
def _ray_bvh(
  # Model:
  nmeshface: int,
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mesh_vertadr: wp.array(dtype=int),
  mesh_faceadr: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_face: wp.array(dtype=wp.vec3i),
  hfield_size: wp.array(dtype=wp.vec4),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_adr: wp.array(dtype=int),
  hfield_data: wp.array(dtype=float),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # BVH context:
  bvh_id: wp.uint64,
  group_root: wp.array(dtype=int),
  ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array(dtype=int),
  # Out:
  dist_out: wp.array2d(dtype=float),
  geomid_out: wp.array2d(dtype=int),
  normal_out: wp.array2d(dtype=wp.vec3),
):
  """BVH-accelerated ray casting kernel."""
  worldid, rayid = wp.tid()

  ray_origin = pnt[worldid, rayid]
  ray_dir = vec[worldid, rayid]
  body_exclude = bodyexclude[rayid]

  min_dist = float(wp.inf)
  min_geomid = int(-1)
  min_normal = wp.vec3()

  # Query BVH for potential intersections
  query = wp.bvh_query_ray(bvh_id, ray_origin, ray_dir, group_root[worldid])
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, min_dist):
    # Map BVH index to actual geometry ID
    bvh_local = bounds_nr - (worldid * ngeom)
    geomid = enabled_geom_ids[bvh_local]

    # Check filtering criteria
    if _ray_eliminate(
      body_weldid,
      geom_bodyid,
      geom_matid[worldid % geom_matid.shape[0]],
      geom_group,
      geom_rgba[worldid % geom_rgba.shape[0]],
      mat_rgba[worldid % mat_rgba.shape[0]],
      geomid,
      geomgroup,
      flg_static,
      body_exclude,
    ):
      continue

    # Narrow-phase intersection test
    dist, normal = _ray_geom_bvh(
      nmeshface,
      geom_type,
      geom_bodyid,
      geom_dataid,
      geom_size,
      mesh_vertadr,
      mesh_faceadr,
      mesh_vert,
      mesh_face,
      hfield_size,
      hfield_nrow,
      hfield_ncol,
      hfield_adr,
      hfield_data,
      mesh_bvh_id,
      hfield_bvh_id,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      ray_origin,
      ray_dir,
      geomid,
      min_dist,
    )

    if dist >= 0.0 and dist < min_dist:
      min_dist = dist
      min_geomid = geomid
      min_normal = normal

  # Write outputs
  if wp.isinf(min_dist):
    dist_out[worldid, rayid] = -1.0
  else:
    dist_out[worldid, rayid] = min_dist
  geomid_out[worldid, rayid] = min_geomid
  normal_out[worldid, rayid] = min_normal


def rays_bvh(
  m: Model,
  d: Data,
  ctx: RayBvhContext,
  pnt: wp.array2d,
  vec: wp.array2d,
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array,
  dist: wp.array2d,
  geomid: wp.array2d,
  normal: wp.array2d,
):
  """BVH-accelerated ray casting.

  This function performs ray casting using the BVH for acceleration.
  It's a drop-in replacement for `rays()` with the addition of the
  BVH context parameter.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state (device).
    ctx: The BVH context from `build_ray_bvh()`.
    pnt: Ray origin points, shape (nworld, nrays).
    vec: Ray directions, shape (nworld, nrays).
    geomgroup: Group inclusion/exclusion mask.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Per-ray body exclusion IDs, shape (nrays,).
    dist: Output distances, shape (nworld, nrays).
    geomid: Output geometry IDs, shape (nworld, nrays).
    normal: Output normals, shape (nworld, nrays).
  """
  wp.launch(
    _ray_bvh,
    dim=(d.nworld, pnt.shape[1]),
    inputs=[
      m.nmeshface,
      m.body_weldid,
      m.geom_type,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_matid,
      m.geom_group,
      m.geom_size,
      m.geom_rgba,
      m.mesh_vertadr,
      m.mesh_faceadr,
      m.mesh_vert,
      m.mesh_face,
      m.hfield_size,
      m.hfield_nrow,
      m.hfield_ncol,
      m.hfield_adr,
      m.hfield_data,
      m.mat_rgba,
      ctx.bvh_id,
      ctx.group_root,
      ctx.ngeom,
      ctx.enabled_geom_ids,
      ctx.mesh_bvh_id,
      ctx.hfield_bvh_id,
      d.geom_xpos,
      d.geom_xmat,
      pnt,
      vec,
      geomgroup,
      flg_static,
      bodyexclude,
      dist,
      geomid,
      normal,
    ],
  )
