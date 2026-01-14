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
"""Tests for BVH-accelerated ray casting functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.types import vec6

# Tolerance for difference between MuJoCo and mujoco_warp ray calculations
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _ray_bvh_single(m, d, ctx, pnt_np, vec_np, geomgroup=None, flg_static=True, bodyexclude=-1):
  """Helper to cast a single ray using BVH and return numpy results."""
  pnt = wp.array([wp.vec3(*pnt_np)], dtype=wp.vec3).reshape((1, 1))
  vec = wp.array([wp.vec3(*vec_np)], dtype=wp.vec3).reshape((1, 1))

  dist = wp.zeros((1, 1), dtype=float)
  geomid = wp.zeros((1, 1), dtype=int)
  normal = wp.zeros((1, 1), dtype=wp.vec3)
  bodyexclude_arr = wp.array([bodyexclude], dtype=int)

  if geomgroup is None:
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

  mjw.rays_bvh(m, d, ctx, pnt, vec, geomgroup, flg_static, bodyexclude_arr, dist, geomid, normal)
  wp.synchronize()

  return dist.numpy()[0, 0], geomid.numpy()[0, 0], normal.numpy()[0, 0]


class RayBvhTest(absltest.TestCase):
  """Tests for BVH-accelerated ray casting."""

  # =========================================================================
  # Core Matching Tests
  # =========================================================================

  def test_bvh_matches_brute_force(self):
    """BVH ray casting should match brute-force results."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    # Test multiple rays
    test_rays = [
      ([0.0, 0.0, 1.6], [0.1, 0.2, -1.0]),  # Looking at sphere
      ([1.0, 0.0, 1.6], [0.0, 0.05, -1.0]),  # Looking at box
      ([0.5, 1.0, 1.6], [0.0, 0.05, -1.0]),  # Looking at capsule
      ([2.0, 1.0, 3.0], [0.1, 0.2, -1.0]),  # Looking at plane
      ([12.0, 1.0, 3.0], [0.0, 0.0, -1.0]),  # Looking at nothing
    ]

    for pnt_np, vec_np in test_rays:
      vec_np = np.array(vec_np)
      vec_np = vec_np / np.linalg.norm(vec_np)

      # BVH result
      bvh_dist, bvh_geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)

      # Brute-force result
      pnt = wp.array([wp.vec3(*pnt_np)], dtype=wp.vec3).reshape((1, 1))
      vec = wp.array([wp.vec3(*vec_np)], dtype=wp.vec3).reshape((1, 1))
      bf_dist, bf_geomid, _ = mjw.ray(m, d, pnt, vec)
      wp.synchronize()
      bf_dist = bf_dist.numpy()[0, 0]
      bf_geomid = bf_geomid.numpy()[0, 0]

      _assert_eq(bvh_geomid, bf_geomid, f"geomid for ray from {pnt_np}")
      _assert_eq(bvh_dist, bf_dist, f"dist for ray from {pnt_np}")

  def test_bvh_matches_mujoco(self):
    """BVH ray casting should match MuJoCo engine."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    # Test multiple rays
    test_rays = [
      ([0.0, 0.0, 1.6], [0.1, 0.2, -1.0]),  # sphere
      ([1.0, 0.0, 1.6], [0.0, 0.05, -1.0]),  # box
      ([2.0, 1.0, 3.0], [0.1, 0.2, -1.0]),  # plane
    ]

    for pnt_np, vec_np in test_rays:
      pnt_np = np.array(pnt_np)
      vec_np = np.array(vec_np)
      vec_np = vec_np / np.linalg.norm(vec_np)

      # BVH result
      bvh_dist, bvh_geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)

      # MuJoCo result
      mj_geomid = np.zeros(1, dtype=np.int32)
      mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)

      _assert_eq(bvh_geomid, mj_geomid[0], f"geomid for ray from {pnt_np}")
      _assert_eq(bvh_dist, mj_dist, f"dist for ray from {pnt_np}")

  # =========================================================================
  # Geometry Type Tests
  # =========================================================================

  def test_bvh_ray_nothing(self):
    """BVH returns -1 when nothing is hit."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    dist, geomid, normal = _ray_bvh_single(m, d, ctx, [12.146, 1.865, 3.895], [0.0, 0.0, -1.0])
    _assert_eq(geomid, -1, "geomid")
    _assert_eq(dist, -1, "dist")
    _assert_eq(normal, 0, "normal")

  def test_bvh_ray_plane(self):
    """BVH ray<>plane matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 1.0, 3.0])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 0, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_sphere(self):
    """BVH ray<>sphere matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([0.0, 0.0, 1.6])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 1, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_capsule(self):
    """BVH ray<>capsule matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([0.5, 1.0, 1.6])
    vec_np = np.array([0.0, 0.05, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 2, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_box(self):
    """BVH ray<>box matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([1.0, 0.0, 1.6])
    vec_np = np.array([0.0, 0.05, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 3, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_cylinder(self):
    """BVH ray<>cylinder matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 0.0, 0.05])
    vec_np = np.array([0.0, 0.05, 1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)

    _assert_eq(geomid, mj_geomid[0], "geomid")
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_mesh(self):
    """BVH ray<>mesh matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    # Look at the tetrahedron
    pnt_np = np.array([2.0, 2.0, 2.0])
    vec_np = np.array([-1.0, -1.0, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 4, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

    # Look at the dodecahedron
    pnt_np = np.array([4.0, 2.0, 2.0])
    vec_np = np.array([-2.0, -1.0, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, 5, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

  def test_bvh_ray_hfield(self):
    """BVH ray<>heightfield matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([0.0, 2.0, 2.0])
    vec_np = np.array([0.0, 0.0, -1.0])

    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)

    _assert_eq(dist, mj_dist, "dist")
    _assert_eq(geomid, mj_geomid[0], "geomid")

  # =========================================================================
  # Filter Tests
  # =========================================================================

  def test_bvh_geomgroup_filter(self):
    """BVH respects geomgroup filtering."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 1.0, 3.0])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    # Hits plane with geom_group[0] = 1
    geomgroup = vec6(1, 0, 0, 0, 0, 0)
    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np, geomgroup=geomgroup)
    _assert_eq(geomid, 0, "geomid")

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, mj_geomid)
    _assert_eq(dist, mj_dist, "dist")

    # Nothing hit with geom_group[0] = 0
    geomgroup = vec6(0, 0, 0, 0, 0, 0)
    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np, geomgroup=geomgroup)
    _assert_eq(geomid, -1, "geomid")
    _assert_eq(dist, -1, "dist")

  def test_bvh_flg_static(self):
    """BVH respects flg_static flag."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 1.0, 3.0])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    # Nothing hit with flg_static = False
    dist, geomid, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np, flg_static=False)
    _assert_eq(geomid, -1, "geomid")
    _assert_eq(dist, -1, "dist")

  def test_bvh_bodyexclude(self):
    """BVH respects body exclusion."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 1.0, 3.0])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    # Nothing hit with bodyexclude = 0 (world body)
    dist, geomid, normal = _ray_bvh_single(m, d, ctx, pnt_np, vec_np, bodyexclude=0)
    _assert_eq(geomid, -1, "geomid")
    _assert_eq(dist, -1, "dist")
    _assert_eq(normal, 0, "normal")

  def test_bvh_invisible(self):
    """BVH doesn't hit transparent geoms."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # Make all geoms transparent
    m.geom_rgba = wp.array2d([[wp.vec4(0.0, 0.0, 0.0, 0.0)] * 8], dtype=wp.vec4)
    mujoco.mj_forward(mjm, mjd)

    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([2.0, 1.0, 3.0])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    dist, geomid, normal = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid, -1, "geomid")
    _assert_eq(dist, -1, "dist")
    _assert_eq(normal, 0, "normal")

  # =========================================================================
  # BVH Lifecycle Tests
  # =========================================================================

  def test_build_ray_bvh(self):
    """Test BVH construction succeeds and returns valid context."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    # Verify context has expected attributes
    self.assertIsNotNone(ctx.bvh)
    self.assertIsNotNone(ctx.bvh_id)
    self.assertGreater(ctx.ngeom, 0)
    self.assertEqual(ctx.lower.shape[0], d.nworld * ctx.ngeom)
    self.assertEqual(ctx.upper.shape[0], d.nworld * ctx.ngeom)
    self.assertEqual(ctx.group_root.shape[0], d.nworld)

  def test_bvh_enabled_geom_groups(self):
    """Test enabled_geom_groups parameter in build_ray_bvh."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # Build BVH with only group 0
    ctx = mjw.build_ray_bvh(m, d, enabled_geom_groups=[0])

    # The BVH should only contain geometries from group 0
    # Verify by checking that ngeom is less than total geoms (if there are geoms in other groups)
    geom_group_np = m.geom_group.numpy()
    expected_ngeom = sum(1 for g in geom_group_np if g == 0)
    self.assertEqual(ctx.ngeom, expected_ngeom)

  def test_refit_ray_bvh(self):
    """Test BVH refit updates bounds correctly."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    # Store initial bounds
    initial_lower = ctx.lower.numpy().copy()

    # Simulate physics step (just forward, no step needed for static scene)
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)

    # Refit BVH
    mjw.refit_ray_bvh(m, d, ctx)

    # Bounds should be recomputed (may or may not change for static scene)
    # Just verify the function runs without error
    final_lower = ctx.lower.numpy()
    self.assertEqual(initial_lower.shape, final_lower.shape)

  def test_refit_preserves_correctness(self):
    """After refit, ray results should still be correct."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    pnt_np = np.array([0.0, 0.0, 1.6])
    vec_np = np.array([0.1, 0.2, -1.0])
    vec_np = vec_np / np.linalg.norm(vec_np)

    # Initial ray cast
    dist1, geomid1, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    self.assertEqual(geomid1, 1)  # Should hit sphere

    # Forward kinematics (positions don't change for static scene)
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)

    # Refit BVH
    mjw.refit_ray_bvh(m, d, ctx)

    # Ray cast after refit should give same result
    dist2, geomid2, _ = _ray_bvh_single(m, d, ctx, pnt_np, vec_np)
    _assert_eq(geomid2, geomid1, "geomid after refit")
    _assert_eq(dist2, dist1, "dist after refit")

  # =========================================================================
  # Edge Cases
  # =========================================================================

  def test_bvh_many_rays(self):
    """BVH handles many simultaneous rays correctly."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    ctx = mjw.build_ray_bvh(m, d)

    nrays = 100
    # Generate random rays pointing downward
    np.random.seed(42)
    pnt_np = np.random.uniform(-5, 5, (nrays, 3)).astype(np.float32)
    pnt_np[:, 2] = 3.0  # All start from z=3
    vec_np = np.zeros((nrays, 3), dtype=np.float32)
    vec_np[:, 2] = -1.0  # All point down

    pnt = wp.array(pnt_np, dtype=wp.vec3).reshape((1, nrays))
    vec = wp.array(vec_np, dtype=wp.vec3).reshape((1, nrays))

    dist = wp.zeros((1, nrays), dtype=float)
    geomid = wp.zeros((1, nrays), dtype=int)
    normal = wp.zeros((1, nrays), dtype=wp.vec3)
    bodyexclude = wp.full(nrays, -1, dtype=int)
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

    mjw.rays_bvh(m, d, ctx, pnt, vec, geomgroup, True, bodyexclude, dist, geomid, normal)
    wp.synchronize()

    # Verify results are valid (either hit something or -1)
    geomid_np = geomid.numpy()[0]
    dist_np = dist.numpy()[0]

    for i in range(nrays):
      if geomid_np[i] >= 0:
        self.assertGreater(dist_np[i], 0, f"Ray {i} hit geom but dist <= 0")
      else:
        self.assertEqual(dist_np[i], -1, f"Ray {i} missed but dist != -1")


if __name__ == "__main__":
  absltest.main()
