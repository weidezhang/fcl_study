# FCL — Design, Mathematical Background, and Use in Manipulation

**Scope.** This document explains (1) the geometry and math that FCL is built on,
(2) the capabilities it actually exposes in *this* tree, including the gaps, and
(3) how a manipulation stack — motion planning and reactive control — consumes it.

**Version in this tree:** `0.7.0` (`package.xml`), C++ (CMake ≥ 3.14).
**Dependencies:** Eigen3 ≥ 3.0.5 (required), libccd (required), octomap (optional,
gates `OcTree` behind `FCL_HAVE_OCTOMAP`).

**Code layout at a glance:**

```
include/fcl/
  common/      scalar typedefs (types.h), exception, profiler, time
  math/        Eigen aliases, BV types, motion models, Taylor/interval models,
               samplers, RNG, geometry helpers (eigen decomposition, PCA, etc.)
  geometry/    CollisionGeometry hierarchy: shapes, BVH (meshes), octree
  narrowphase/ collide() / distance() / continuousCollide() + all the solvers
  broadphase/  N-body acceleration structures (managers)
```

Nearly everything is a template on the scalar `S` with the implementation in a
parallel `*-inl.h` file; `src/` holds the explicit instantiations for `float` and
`double` so the library ships as a real binary (`libfcl`). 312 headers, 143 of
them `-inl.h`.

---

## 1. What FCL is, and what it deliberately is not

FCL answers **proximity queries between two posed rigid geometries**:

| Query | Entry point | Returns |
|---|---|---|
| Discrete collision | `fcl::collide()` | number of contacts; optionally contact points, normals, penetration depths |
| Distance / nearest points | `fcl::distance()` | minimum distance; optionally the witness point pair and which primitive they lie on |
| Continuous collision | `fcl::continuousCollide()` | whether a swept motion collides, and the time of first contact `toc ∈ [0,1]` |
| N-body culling | `BroadPhaseCollisionManager` | candidate pairs, via a user callback |

**It is not:**

- a kinematics library — there is no joint, link, or forward-kinematics concept.
  You compute `Transform3<S>` per body yourself and hand it in.
- a dynamics or contact-resolution engine — it reports contacts; it does not
  integrate them. (`computeMomentofInertia()`/`computeVolume()` exist on the
  geometries as a convenience for whoever *does* the dynamics.)
- differentiable — no analytic derivative of distance w.r.t. pose is returned.
  `math/constants.h` is written to tolerate `Eigen::AutoDiffScalar` as `S` (see
  `detail::ScalarTrait`, and `test/test_fcl_auto_diff.cpp`), but this is partial:
  most of the library has only `float`/`double` instantiations, and the `libccd`
  solver path is plain `double` C code. Treat autodiff as experimental.
- a collision *policy* — allowed-collision matrices, link padding, and
  self-collision filtering live in the layer above (MoveIt, Drake, your planner).

---

## 2. Conventions you must internalise

### 2.1 Scalars and linear algebra

`include/fcl/common/types.h` is a thin alias layer over Eigen:

```cpp
template <typename S> using Vector3    = Eigen::Matrix<S, 3, 1>;
template <typename S> using Matrix3    = Eigen::Matrix<S, 3, 3>;
template <typename S> using Quaternion = Eigen::Quaternion<S>;
template <typename S> using Transform3 = Eigen::Transform<S, 3, Eigen::Isometry>;
```

Two consequences worth stating out loud:

1. **`Transform3` is `Isometry`, not `Affine`.** A pose is a rigid
   transform in SE(3): rotation + translation, **no scale, no shear**. If you
   want a scaled mesh you must scale the vertex data. Eigen will happily let you
   write a non-orthonormal `linear()` block, and FCL's math then silently
   produces garbage — several routines exploit `R⁻¹ = Rᵀ`.
2. Per-scalar aliases exist throughout (`Boxd`, `CollisionObjectf`,
   `DistanceRequestd`, …). `double` is the well-trodden path.

Tolerances come from `fcl::constants<S>`:

```cpp
constants<S>::pi(), ::phi()
constants<S>::eps()      // machine epsilon of S
constants<S>::eps_78()   // eps^(7/8)   ← tightest practical tolerance
constants<S>::eps_34()   // eps^(3/4)
constants<S>::eps_12()   // eps^(1/2)   ← loosest; "half the digits"
```

The fractional-power scheme is deliberate: a hard-coded `1e-9` is meaningless for
`float` and wasteful for `double`, whereas `eps^(3/4)` *scales* with the
representation. Use these rather than literals when you extend the library.

### 2.2 Frames

Every `CollisionGeometry` is defined in its own **canonical local frame G**:

- Solids are **centred at the origin**. `Box(x,y,z)` spans `[-x/2, x/2] × …`.
- Solids of revolution (`Capsule`, `Cone`, `Cylinder`) have their **axis along +z**
  and length `lz`. `Cone` is the exception for the centroid: `computeCOM()`
  returns a point at `z = -lz/4`, because the centroid of a cone is a quarter of
  the way up from the base, while the *geometry* is still centred on the origin.
- `Halfspace`/`Plane` are `{x : n·x = d}`; for `Halfspace` the interior is
  `n·x < d`. These are infinite, which is why their `computeLocalAABB()` and
  their role in BV hierarchies are special-cased.

A `CollisionObject` is `(shared_ptr<CollisionGeometry>, Transform3 X_WG)`. So

```
p_W = X_WG * p_G
```

**Reduction to a relative frame is the central performance trick.** Rather than
transforming both objects to the world, almost every traversal reduces to
"object 1 in configuration `(R, T)`, object 2 at the identity":

```
X_12 = X_W1⁻¹ · X_W2
```

`relativeTransform()` in `math/geometry.h` does this. That is why the BV overlap
and distance free functions have signatures like

```cpp
bool overlap(const Matrix3<S>& R0, const Vector3<S>& T0,
             const OBB<S>& b1, const OBB<S>& b2);
```

The newer `RSS` class documents this with monogram notation (`R_FT`, `p_FoTo_F`)
and is the best-documented BV in the tree — read it first.

### 2.3 Request / Result

Every query takes a `…Request` (what to compute, how hard to try) and fills a
`…Result`. Results are **accumulators**: `DistanceResult::update()` keeps the
running minimum; `CollisionResult::addContact()` appends. `request.isSatisfied(result)`
is the early-exit predicate the traversals consult. **Clear your result object
between queries** or you will keep a stale minimum.

---

## 3. The mathematics, layer by layer

### 3.1 Support functions and the Minkowski difference

The convex machinery rests on the **support function** of a convex set `A`:

```
h_A(d) = max { d·x : x ∈ A }        s_A(d) = argmax { d·x : x ∈ A }
```

`detail::getSupport()` (`convexity_based_algorithm/minkowski_diff.h`,
`support.h`) implements `s_A` in closed form per shape — for a box it is a sign
test per axis, for a sphere it is `r·d̂`, for a cylinder a cap/rim decision, and
for `Convex` it is `Convex::findExtremeVertex()`.

`Convex::findExtremeVertex()` is worth a look: for meshes with ≥ 32 vertices
(`kMinVertCountForEdgeWalking`) and a validated watertight topology, it **hill-climbs
the vertex adjacency graph** instead of scanning all vertices — O(√n)-ish in
practice vs O(n). The adjacency is stored in a single flat `std::vector<int>`
(offset table followed by `count, n₀, n₁, …` runs) specifically for cache
locality. If `ValidateTopology()` fails, `find_extreme_via_neighbors_` is cleared
and it falls back to a linear scan — i.e. **a malformed convex mesh silently costs
you performance**, not correctness.

The key identity: for the **Minkowski difference** `A ⊖ B = {a − b}`,

```
h_{A⊖B}(d) = h_A(d) + h_B(−d)
dist(A, B) = dist(0, A ⊖ B)
A ∩ B ≠ ∅  ⟺  0 ∈ A ⊖ B
```

`MinkowskiDiff<S>` stores the two shape pointers plus `toshape0`/`toshape1`
(the relative transforms) so the support of the difference is evaluated without
materialising anything. It also exposes `support(d, v)` — the support of a shape
*translating at velocity v* — which is how the sweep-based CCD reuses the same
code.

### 3.2 GJK — distance between convex sets

Gilbert–Johnson–Keerthi searches for the point of `A ⊖ B` closest to the origin by
maintaining a simplex (1–4 vertices) of support points and iteratively:

1. pick a direction `d` (toward the origin from the current simplex),
2. add `s_{A⊖B}(d)` to the simplex,
3. reduce the simplex to the face containing the closest point
   (`detail::Project`, `math/detail/project.h`, gives barycentric projections of
   the origin onto point/segment/triangle/tetrahedron),
4. stop when progress `< tolerance` or the origin is enclosed.

`GJK<S>` (`convexity_based_algorithm/gjk.h`) holds `simplices[2]` so the previous
simplex survives for **warm starting**: `getGuessFromSimplex()` plus
`CollisionRequest::enable_cached_gjk_guess` / `cached_gjk_guess` and
`CollisionResult::cached_gjk_guess` let you feed the previous frame's separating
direction into the next query. In a control loop where poses change slightly per
tick, this routinely cuts GJK iterations to one or two. **This is the single
cheapest optimisation available to a 1 kHz consumer of FCL.**

### 3.3 EPA — penetration depth

When the origin is inside `A ⊖ B`, distance is zero and the interesting quantity is
the **penetration depth**

```
δ = min { ‖t‖ : (A + t) ∩ B = ∅ }
```

which equals the distance from the origin to the *boundary* of `A ⊖ B`. The
Expanding Polytope Algorithm (`epa.h`) starts from the GJK termination simplex,
repeatedly takes the polytope face nearest the origin, and expands through it with
a new support point until the face stops moving. The face normal is the contact
normal, and the barycentric weights give the witness points. `alloc.h` / `list.h`
are a hand-rolled pooled linked list to keep this allocation-free.

EPA is the numerically delicate part of any collision library: the result depends
on the quality of the initial simplex, and for deep or face-face-degenerate
penetration it can return a normal that flips between frames. Budget for that in
a controller (see §14.5).

### 3.4 Separating-axis theorem — boxes and OBBs

For two convex polytopes, disjointness is certified by a single axis `a` on which
their projections don't overlap. For two boxes it suffices to test 15 axes: the 3
face normals of each box plus the 9 pairwise cross products. `obbDisjoint()` in
`math/bv/OBB-inl.h` is the classic Gottschalk/RAPID formulation: with `B = R_12`,
`T = p_12`, half-extents `a`, `b`, and `Bf = |B| + ε`,

```
reject if  |T·aᵢ| > aᵢ + Bf.row(i)·b       (3 face axes of box 1)
reject if  |T·(B·bⱼ)| > bⱼ + Bf.col(j)·a   (3 face axes of box 2)
reject if  … 9 cross-product axes …
```

`reps = 1e-6` is added to `|B|` to make the cross-product tests robust to
near-parallel axes. Early exit on the first separating axis is why OBB overlap is
so cheap in practice.

### 3.5 Swept-sphere volumes and the rectangle-distance kernel

`RSS` (Rectangle Swept Sphere) = Minkowski sum of an axis-aligned rectangle (in
its own frame T, one corner at T's origin, long side along +x, lying in `z = 0`)
with a ball of radius `r`. The point of this shape is that **distance between two
RSSs is closed-form**:

```
dist(RSS₁, RSS₂) = max(0, rectDistance(X_12, l₁, l₂) − r₁ − r₂)
```

`rectDistance()` reduces rectangle-rectangle distance to edge-edge distance via
`segCoords()` (Lumelsky's 1985 closest-points-on-two-segments) plus `inVoronoi()`
Voronoi-region tests. That is the whole reason RSS exists: **AABB and OBB give you
cheap overlap; RSS gives you cheap *distance*.** This is the pivot in choosing a
BV for a distance-driven controller.

`kIOS` takes the complementary approach: up to 5 spheres whose intersection bounds
the geometry, plus a cached `OBB`. Sphere-sphere overlap is trivially
rotation-invariant, so `kIOS` is cheap to transform; `overlap()` screens with the
spheres then falls back to the OBB.

`kDOP<S,N>` (N = 16, 18, 24) stores signed distances from the origin to `N/2`
pairs of parallel planes with fixed normals: the 3 axis directions plus edge/corner
diagonals like `(±1, ±1, 0)` and `(±1, ±1, ±1)`. Overlap is `N/2` interval
comparisons. kDOPs are tighter than AABBs and cheaper than OBBs, but they are
**axis-fixed**: rotating the object invalidates them, so they are only useful for
geometry that does not rotate relative to the hierarchy. `KDOP::distance()` is
explicitly **not implemented**.

### 3.6 Triangle kernels

Meshes bottom out in two routines:

- `detail::TriangleDistance<S>::triDistance()` — distance between two triangles
  plus witness points. Tests all 9 edge-edge pairs, then the vertex-face cases.
- `detail::Intersect<S>::intersect_Triangle()` — triangle-triangle intersection,
  with contact points, penetration depth and normal; there is also
  `intersect_Triangle_ODE_style()`. `Intersect` additionally provides the
  continuous primitives `intersect_VF` (moving vertex vs moving face) and
  `intersect_EE` (moving edge vs moving edge), each solving the resulting cubic
  in time (`math/detail/polysolver.h`) with an optional Newton polish and
  interval-based filters (`intersect_VF_filtered`, `intersect_EE_filtered`).

### 3.7 Interval arithmetic and Taylor models

`math/motion/taylor_model/` implements a first-order-plus-remainder Taylor model
over a time interval:

```
f(t) ≈ c₀ + c₁t + c₂t² + c₃t³ + R,   R an Interval, t ∈ TimeInterval
```

with `Interval`, `IntervalVector3`, `IntervalMatrix3`, `TaylorModel`,
`TVector3`, `TMatrix3`. `MotionBase::getTaylorModel(TMatrix3&, TVector3&)`
returns a **conservative enclosure of the body's trajectory**, which
`TBVMotionBoundVisitor` turns into a bounding volume that contains the body over
the whole interval. This is how FCL bounds *rotating* motion without sampling it:
rotations are nonlinear in time, so you bound them rather than evaluate them.

### 3.8 Motion bounds and conservative advancement

This is the mathematical heart of FCL's continuous collision detection, and the
piece most worth understanding if you care about CCD.

Given a direction `n̂` (the current closest-approach direction) and a motion, the
**motion bound** `μ` is an upper bound on how fast any point of the body can
approach along `n̂`. For `InterpMotion` (constant linear velocity `v` + constant
angular velocity `ω` about axis `â` through reference point `p_ref`), the bound
for a triangle with vertices `a, b, c` is
(`triangle_motion_bound_visitor-inl.h`):

```
μ = v·n̂  +  ‖â × n̂‖ · ω · max_{p ∈ {a,b,c}} ‖ (R(t)(p − p_ref)) × â ‖
```

The second term is the classic "lever arm × angular rate" bound: the largest
distance from the rotation axis over the primitive's vertices, times the angular
speed, times the component of the rotation that actually moves along `n̂`. The RSS
version (`tbv_motion_bound_visitor-inl.h`) does the same over the rectangle's four
corners and then adds the swept radius:

```
μ = v·n̂ + ‖â × n̂‖ · ω · (r + c_proj_max)
```

**Conservative advancement** then iterates (see
`MeshConservativeAdvancementTraversalNode::leafTesting`):

```
t ← 0
loop:
  d  ← distance between the two bodies at time t          (BVH-accelerated)
  n̂  ← unit vector between the witness points
  μ  ← μ₁(n̂) + μ₂(n̂)                                     (both motion bounds)
  if μ ≤ d:  Δt = 1                 # can't close the gap in the remaining time
  else:      Δt = d / μ             # provably safe step
  t ← t + Δt
  until d < tolerance or t ≥ 1
```

Because `μ` bounds the approach *rate*, advancing by `d/μ` can never skip a
collision. It is exact in the limit and never tunnels — unlike sampling. The
per-leaf `Δt` is minimised over all leaf pairs, so the tree traversal tightens the
step as it descends.

### 3.9 PCA, variance, and BV fitting

`math/geometry.h` provides the fitting primitives:

- `eigen()` / `eigen_old()` — symmetric 3×3 eigendecomposition (Jacobi).
- `axisFromEigen()` — order the eigenvectors into a BV axis frame, longest
  principal direction first (which is exactly the convention `OBB::axis` and
  `RSS::axis` document).
- `getExtentAndCenter()`, `getRadiusAndOriginAndRectangleSize()`,
  `maximumDistance()`, `circumCircleComputation()` — compute the actual BV
  parameters for a primitive subset given the axes.
- `generateCoordinateSystem()` — complete an orthonormal frame from one axis.
- `hat()` — the skew-symmetric cross-product matrix `[v]×`.

`Variance3<S>` plus `BVHExpand()` (`geometry/bvh/BVH_utility.h`) inflate a
hierarchy's BVs by `r` standard deviations of per-node point covariance — the hook
for treating *uncertain* geometry (noisy point clouds) conservatively.

### 3.10 Spatial hashing and Morton codes

For the broadphase: `detail::SpatialHash` maps an AABB to the set of grid cells it
overlaps (`broadphase/detail/spatial_hash.h`), with `SimpleHashTable` /
`SparseHashTable` backends. `detail::morton` computes Morton (Z-order) codes,
used to build the hierarchy tree bottom-up with good spatial coherence.

---

## 4. Type architecture

```
                         ┌──────────────────────────┐
                         │     CollisionGeometry<S> │  abstract; local AABB,
                         │                          │  aabb_center/radius,
                         │  computeLocalAABB()      │  cost_density,
                         │  computeVolume/COM/      │  occupancy thresholds
                         │    MomentofInertia()     │
                         └───────────┬──────────────┘
          ┌──────────────────────────┼────────────────────────┐
          ▼                          ▼                        ▼
   ShapeBase<S>  (OT_GEOM)    BVHModel<BV> (OT_BVH)      OcTree<S> (OT_OCTREE)
   Box Sphere Ellipsoid       triangle soup or              octomap-backed
   Capsule Cone Cylinder      point cloud + a
   Convex Plane Halfspace     hierarchy of BV
   TriangleP

                         ┌──────────────────────────┐
                         │   CollisionObject<S>     │ = geometry + X_WG
                         │   getAABB() (world AABB) │   + cached world AABB
                         └──────────────────────────┘
                         ┌──────────────────────────────────────┐
                         │ ContinuousCollisionObject<S>         │
                         │   = geometry + MotionBase<S>         │
                         └──────────────────────────────────────┘
```

`OBJECT_TYPE` ∈ {`OT_GEOM`, `OT_BVH`, `OT_OCTREE`} is the coarse family;
`NODE_TYPE` ∈ {`GEOM_BOX` … `GEOM_OCTREE`, `BV_AABB` … `BV_KDOP24`} is the fine
type used as the **dispatch key**.

### 4.1 Dispatch: the function matrices

There is no virtual `collide()` on the geometries. Instead, three
`NODE_COUNT × NODE_COUNT` tables of function pointers are built once (static
local, lazily initialised in `getCollisionFunctionLookTable<Solver>()`):

| Table | Header | Purpose |
|---|---|---|
| `CollisionFunctionMatrix` | `detail/collision_func_matrix.h` | discrete collision |
| `DistanceFunctionMatrix` | `detail/distance_func_matrix.h` | distance |
| `ConservativeAdvancementFunctionMatrix` | `detail/conservative_advancement_func_matrix.h` | CCD |

`collide()` looks up `matrix[node_type1][node_type2]`, and if the pair is
`(OT_GEOM, OT_BVH)` it **swaps the arguments** and uses `matrix[type2][type1]`
instead — the tables only populate the BVH-first orientation for mesh/shape pairs.

**An unpopulated cell is not an error.** It prints
`"Warning: collision function between node type X and node type Y is not supported"`
to `stderr` and returns `0` / `max()`. A silent `0` from `collide()` means
"no collision" *or* "unsupported pair". If you are wiring up a new geometry
combination, check the table before trusting a negative result.

Two narrow-phase solvers exist, selected by `GJKSolverType` on the request:

| | `GST_LIBCCD` (default) | `GST_INDEP` |
|---|---|---|
| Implementation | wraps the external `libccd` C library (`gjk_libccd.h/-inl.h`) | FCL's own `GJK` + `EPA` (`gjk.h`, `epa.h`) |
| Tuning | `collision_tolerance`, `distance_tolerance`, `max_*_iterations` | `gjk_tolerance`, `epa_tolerance`, max iterations |
| Signed distance for primitive pairs | native (`shapeSignedDistance`), witness points **on the surfaces** | via penetration-depth workaround, witness points **not** on the surfaces |
| Warm start | `enableCachedGuess`/`setCachedGuess` present but documented as unsupported in the stream operator | supported |

`CollisionRequest::gjk_tolerance` defaults to `1e-6` purely for backward
compatibility — the header itself notes future code should pick something
consistent with `S`'s precision.

Low-level failures throw `detail::FailedAtThisConfiguration`, which callers that
hold the objects and transforms are expected to catch and re-throw as a
`std::logic_error` carrying a copy-pasteable dump of the poses
(`WriteCommaSeparated`). That is a real debugging affordance: when a GJK call
blows up in a 500-link scene, the exception tells you which two poses to
reproduce. Pair it with `Shape::representation(precision)`, which prints a
constructor call that recreates the shape.

---

## 5. Shape catalogue

All in `geometry/shape/`. `getBoundVertices(tf)` returns a convex point set that
bounds the shape in configuration `tf` — used to build BVs and to feed the
conservative-advancement bound. It is absent on `Plane`/`Halfspace` (unbounded).

| Shape | Parameters | Local frame | Volume | Inertia | `getBoundVertices` |
|---|---|---|---|---|---|
| `Box` | `Vector3 side` | centred, axis-aligned | ✔ | ✔ | ✔ (8 corners) |
| `Sphere` | `S radius` | centred | ✔ | ✔ | ✔ |
| `Ellipsoid` | `Vector3 radii` | centred | ✔ | ✔ | ✔ |
| `Capsule` | `radius`, `lz` | axis +z, centred | ✔ | ✔ | ✔ |
| `Cone` | `radius`, `lz` | axis +z, centred; COM at `z=-lz/4` | ✔ | ✔ | ✔ |
| `Cylinder` | `radius`, `lz` | axis +z, centred | ✔ | ✔ | ✔ |
| `Convex` | shared vertices + face encoding | arbitrary | ✔ | ✔ | ✔ |
| `Plane` | `n`, `d` | `n·x = d` | — | — | — |
| `Halfspace` | `n`, `d` | interior `n·x < d` | — | — | — |
| `TriangleP` | `a`, `b`, `c` | explicit points | — | — | ✔ |

### 5.1 `Convex` in detail

`Convex` is the workhorse for real manipulation geometry (convex-decomposed
meshes). Its contract, enforced partially:

- vertices `V`, and `faces` as a **flat int array**: for each face,
  `[k, i₀, i₁, …, i_{k-1}]`, indices **counter-clockwise viewed from outside**.
- each face must be planar and lie on a supporting plane `πᵢ` with
  `πᵢ(v) ≤ 0 ∀ v ∈ V` (that *is* convexity).
- faces may be arbitrary n-gons, not just triangles.
- `vertices` and `faces` are `shared_ptr<const …>` and must not change for the
  object's lifetime.

`throw_if_invalid` triggers `ValidateMesh()`, which checks watertightness (every
edge shared by exactly two faces) and that every vertex belongs to a face. The
header is honest that planarity, the ≥4-vertex minimum, and the "all other
vertices behind each face" test are **not yet** validated — see upstream issue
326. A non-convex mesh handed to `Convex` produces wrong answers quietly.

### 5.2 Shape → BV, and shape → mesh

- `computeBV<BV, Shape>(shape, tf, bv)` (`geometry/shape/utility.h`) specialises
  per (BV, shape) pair: the tight bound of a posed primitive.
- `constructBox(bv[, tf_bv], box, tf)` goes the other way — turn a BV back into a
  `Box` + pose. This is how octree cells and BVH nodes are fed to the shape-shape
  narrow phase.
- `generateBVHModel(model, shape, pose, …)` (`geometry/geometric_shape_to_BVH_model.h`)
  tessellates a primitive into a `BVHModel`, with tessellation knobs per shape
  (`seg`/`ring` for spheres and ellipsoids, or `n_faces_for_unit_sphere` which
  keeps per-triangle *area* roughly constant across radii). `FinalizeModel::DONT`
  lets you accumulate several primitives into one model — which then represents
  their **union**, with no memory of the originals.

---

## 6. Bounding volumes: choosing one

| BV | Storage | Overlap test | Distance test | Rotation-invariant | Notes |
|---|---|---|---|---|---|
| `AABB` | 2 × `Vector3` | 6 interval compares | ✔ closed-form + witness points | ✖ (must refit) | also the broadphase currency |
| `OBB` | `axis` + `To` + `extent` | 15-axis SAT | **not implemented** | ✔ | tightest cheap overlap |
| `RSS` | `axis` + `To` + `l[2]` + `r` | via `rectDistance` | ✔ closed-form + witness points | ✔ | the distance BV |
| `OBBRSS` | an `OBB` + an `RSS` | OBB's | RSS's | ✔ | best of both, 2× memory |
| `kIOS` | ≤5 spheres + `OBB` | spheres then OBB | ✔ approximate (no witness points) | ✔ | header marks `overlap` "not efficient" |
| `kDOP<16/18/24>` | `N` scalars | `N/2` interval compares | **not implemented** | ✖ | tight, static orientation only |

Practical reading of that table:

- **Collision only, static mesh, arbitrary rotation:** `OBB`.
- **Distance or CCD:** `RSS`, or `OBBRSS` if you do both on the same model.
- **Deformable / frequently refit:** `AABB` (cheap refit) or `kDOP`.
- Note the gaps: `BVHModel<OBB>` and `BVHModel<KDOP<…>>` have **no mesh-mesh
  distance entry** in `DistanceFunctionMatrix` — only `AABB`, `RSS`, `kIOS`, and
  `OBBRSS` do. If you build an OBB tree and then call `distance()` on two meshes,
  you get the "not supported" warning. This is the single most common surprise
  for new FCL users.

---

## 7. BVH: meshes and point clouds

`BVHModel<BV>` (`geometry/bvh/BVH_model.h`) is a triangle soup (or point cloud,
`BVH_MODEL_POINTCLOUD`) plus a binary tree of `BVNode<BV>`.

### 7.1 Construction

A strict state machine (`BVHBuildState` in `BVH_internal.h`), with integer return
codes (`BVHReturnCode`) rather than exceptions:

```
empty ──beginModel()──▶ begun ──addVertex/addTriangle/addSubModel──▶ begun
                                        │
                                   endModel()
                                        ▼
                                    processed ◀──┐
                      ┌──beginReplaceModel()─────┤   (same topology, new vertices)
                      │  replaceVertex/Triangle  │
                      │  endReplaceModel(refit, bottomup)
                      └──────────────────────────┤
                      ┌──beginUpdateModel()──────┤   (same, but keeps the previous
                      │  updateVertex/Triangle   │    frame in prev_vertices — this
                      │  endUpdateModel(…)       │    is what CCD on deformables uses)
                      └────▶ updated ────────────┘
```

`endModel()` calls `buildTree()` → `recursiveBuildTree()`: top-down, at each node
fit a BV to the primitive subset, then partition.

- **Fitting:** `detail::BVFitter<BV>::fit()` — specialised per BV; for oriented BVs
  it runs the PCA path from §3.9.
- **Splitting:** `detail::BVSplitter<BV>` with `SplitMethodType`:
  - `SPLIT_METHOD_MEAN` — split at the mean projection on the split axis,
  - `SPLIT_METHOD_MEDIAN` — split at the median (balanced tree, more build cost),
  - `SPLIT_METHOD_BV_CENTER` — split at the BV centre (cheapest, can be unbalanced).
  For axis-aligned BVs the split axis is a world axis index (`split_axis`); for
  oriented BVs it is a full `split_vector`, which gives better partitions.

Both are `shared_ptr` members (`bv_splitter`, `bv_fitter`) on the model, so you can
swap in your own policies without touching FCL.

### 7.2 Refit vs rebuild, and `makeParentRelative()`

`endReplaceModel(refit, bottomup)` / `endUpdateModel(refit, bottomup)`:

- `refit=false` → full rebuild: tightest tree, most expensive.
- `refit=true, bottomup=true` → `refitTree_bottomup()`: fast, looser BVs.
- `refit=true, bottomup=false` → `refitTree_topdown()`: slower, more compact.

For a deforming mesh at interactive rates, bottom-up refit is the right default;
rebuild when the deformation has drifted far enough that traversal costs rise.

`makeParentRelative()` stores each node's BV **relative to its parent** instead of
in model coordinates, saving one transform composition per node during traversal
of oriented hierarchies. For plain (axis-aligned) BVs only the translation is
stored relative; `OBB`/`RSS`/`OBBRSS` have full specialisations
(`MakeParentRelativeRecurseImpl`).

### 7.3 Traversal

`narrowphase/detail/traversal/` holds a node type per (geometry pair, query)
combination — `MeshCollisionTraversalNode`, `MeshShapeDistanceTraversalNode`,
`ShapeMeshConservativeAdvancementTraversalNode`, the octree variants, and so on.
Each exposes the same five hooks consumed by `traversal_recurse.h`:

```
isFirstNodeLeaf(b) / isSecondNodeLeaf(b)   firstOverSecond(b1,b2)  → descend which side
BVTesting(b1, b2)   → overlap or BV distance (the pruning test)
leafTesting(b1, b2) → the real primitive-primitive work
canStop(c)          → early exit (request satisfied, or distance already too big)
```

For distance queries the recursion is **best-first**: `BVTesting` returns the BV
distance, and the traversal always descends into the closer child, so the running
minimum tightens as fast as possible and prunes the rest.

`BVHFrontList` (`geometry/bvh/detail/BVH_front.h`) records the "front" — the set
of node pairs where the previous traversal stopped. Feeding it back into the next
query lets a slightly-moved pair resume from where it left off instead of
re-descending from the roots. See `test/test_fcl_frontlist.cpp`. For a controller
re-querying the same pair every tick, this is the BVH-level analogue of the GJK
cached guess.

---

## 8. Narrow phase: what is actually computed

### 8.1 Closed-form primitive pairs

`narrowphase/detail/primitive_shape_algorithm/` bypasses GJK entirely for pairs
where an exact, cheaper formula exists:

| File | Handles |
|---|---|
| `sphere_sphere` | centre distance vs radius sum |
| `sphere_box`, `sphere_cylinder`, `sphere_capsule` | closest point on the solid to the centre |
| `sphere_triangle` | point-triangle closest point |
| `capsule_capsule` | segment-segment distance (Lumelsky) |
| `box_box` | full SAT with contact manifold generation |
| `halfspace`, `plane` | signed-distance tests against every other primitive |
| `triangle_distance` | the `TriangleDistance<S>` kernel of §3.6 |

These are the paths worth staying on: they are exact, branch-predictable, and have
no iteration count. **A manipulation collision model built from spheres, capsules,
and boxes is dramatically cheaper and more numerically stable than one built from
meshes** — this is why so many production robot collision models are capsule
approximations.

Everything else routes to GJK/EPA via `ShapeShapeCollide` / `ShapeShapeDistance`.

### 8.2 Distance semantics — read this before trusting a negative number

`DistanceRequest::enable_signed_distance` governs behaviour on overlap. The table
in `distance_request.h` is authoritative:

| | `GST_LIBCCD` | `GST_INDEP` |
|---|---|---|
| primitive shapes | SD_1, NP | SD_2, NP_X |
| mesh and octree | SD_2, NP_X | SD_2, NP_X |

- **SD_1** — signed distance computed natively by the convex method.
- **SD_2** — positive distance from GJK, but **negative** distance obtained by a
  workaround: run `collide()` with contacts enabled, take the **maximum**
  penetration depth over the reported contacts, and negate it
  (`distance-inl.h`, around the `enable_signed_distance` block).
- **NP** — witness points are guaranteed on the objects' surfaces.
- **NP_X** — they are **not**. In the SD_2 workaround both `nearest_points[0]` and
  `[1]` are set to the *same* contact position.

With `enable_signed_distance = false` (the default), `min_distance` on overlap is
**implementation-defined, usually `-1`**. Do not feed that into a controller.

Other result fields:

- `nearest_points[2]` are in **world** coordinates.
- `b1`/`b2` identify the witness primitive: triangle/point index for a BVH,
  `NONE (-1)` for a primitive shape, and an opaque **query cell id** for an octree
  — resolve it with `OcTree::getNodeByQueryCellId(id, point, …)`, which needs the
  witness point to find the cell efficiently.

### 8.3 Collision result

`Contact<S>` carries `o1`, `o2`, `b1`, `b2`, `normal` (**pointing from o1 to o2**),
`pos` (world), `penetration_depth`. `collide()` returns the contact count;
`CollisionRequest::num_max_contacts` caps it and `enable_contact` decides whether
the expensive manifold work happens at all. For a binary "is this configuration
valid?" check, leave `enable_contact = false` and `num_max_contacts = 1` — that is
the single biggest knob for planner throughput.

`CostSource` / `enable_cost` is a separate, lightly-documented facility: it reports
AABB regions weighted by `CollisionGeometry::cost_density` so a planner can treat
collision as a soft cost rather than a hard constraint. The header itself carries a
TODO asking for a clear definition; treat it as experimental.

---

## 9. Continuous collision detection

### 9.1 Motion models

`MotionBase<S>` (`math/motion/`) is the interface:

```cpp
bool integrate(S dt) const;                        // advance internal t to dt
void getCurrentTransform(Transform3<S>&) const;    // X(t)
S    computeMotionBound(const BVMotionBoundVisitor<S>&) const;
S    computeMotionBound(const TriangleMotionBoundVisitor<S>&) const;
void getTaylorModel(TMatrix3<S>&, TVector3<S>&) const;
```

| Class | `CCDMotionType` | Trajectory |
|---|---|---|
| `TranslationMotion` | `CCDM_TRANS` | pure translation, fixed orientation |
| `InterpMotion` | `CCDM_LINEAR` | constant linear + constant angular velocity about a reference point: `R(t)(p − p_ref) + p_ref + T(t)` |
| `ScrewMotion` | `CCDM_SCREW` | screw motion (coupled rotation + translation about one axis) |
| `SplineMotion` | `CCDM_SPLINE` | polynomial spline in position and orientation |

`getMotionBase(tf_beg, tf_end, motion_type)` is the factory. Note the default on
`ContinuousCollisionRequest` is `CCDM_TRANS` — i.e. **by default FCL interpolates
your two poses as a pure translation and ignores the rotation**. For a robot link
swinging through an arc that is wrong, and you want `CCDM_LINEAR`.

### 9.2 Solvers

`CCDSolverType` on the request:

| Value | Status in this tree |
|---|---|
| `CCDC_NAIVE` (default) | **implemented.** Samples `min(num_max_iterations, ceil(1/toc_err))` times in `[0,1]`, calls `collide()` at each. Simple; **can tunnel**. |
| `CCDC_CONSERVATIVE_ADVANCEMENT` | **implemented.** The §3.8 algorithm. Provably no tunnelling. The one you want. |
| `CCDC_RAY_SHOOTING` | **empty stub.** The `OT_GEOM`+`CCDM_TRANS` branch has no body and falls through returning `-1`; anything else warns. |
| `CCDC_POLYNOMIAL_SOLVER` | partially implemented: BVH-vs-BVH with `CCDM_TRANS` only, via `continuousCollideBVHPolynomial`; warns for other BV types. |

So: the default request (`CCDC_NAIVE`, `CCDM_TRANS`, 10 iterations, `toc_err=1e-4`)
is the *least* capable configuration. For real CCD:

```cpp
fcl::ContinuousCollisionRequestd req;
req.ccd_solver_type  = fcl::CCDC_CONSERVATIVE_ADVANCEMENT;
req.ccd_motion_type  = fcl::CCDM_LINEAR;     // honour rotation
req.num_max_iterations = 100;
req.toc_err = 1e-4;
```

### 9.3 CCD coverage gaps

`ConservativeAdvancementFunctionMatrix` covers shape-shape (box, sphere, capsule,
cone, cylinder, convex, plane, halfspace), BVH-shape for every BV, and BVH-BVH for
every BV. It does **not** cover:

- **`Ellipsoid`** — absent from the CA matrix entirely.
- **`OcTree`** — no octree entries at all. You cannot conservatively-advance
  against perceived octree geometry; convert it to boxes
  (`OcTree::toBoxes()`) or use discrete checks.

`ContinuousCollisionResult` returns `is_collide`, `time_of_contact ∈ [0,1]`, and
the two contact transforms `contact_tf1`/`contact_tf2` — the latter are what you
want for backing a trajectory off to the last safe point.

---

## 10. Broadphase: N bodies

Pairwise queries are O(N²). `BroadPhaseCollisionManager<S>` culls using each
object's **world AABB** (`CollisionObject::computeAABB()`), then invokes your
callback on surviving pairs.

| Manager | Structure | Best for |
|---|---|---|
| `NaiveCollisionManager` | all pairs | N small, or a correctness baseline |
| `SSaPCollisionManager` | sweep-and-prune on a sorted axis | static or slowly-moving sets |
| `SaPCollisionManager` | incremental sweep-and-prune | high temporal coherence |
| `IntervalTreeCollisionManager` | interval trees per axis | wide AABB size spread |
| `SpatialHashingCollisionManager` | uniform grid hash | uniform object size, known cell size |
| `DynamicAABBTreeCollisionManager` | dynamic AABB (BVH) tree | **the general-purpose default** |
| `DynamicAABBTreeCollisionManager_Array` | same, array-backed nodes | same, better cache locality |

API:

```cpp
registerObject(obj); registerObjects(vec); unregisterObject(obj);
setup();                       // after bulk registration
update();                      // all objects moved
update(obj); update(vec);      // only these moved  ← prefer this
collide(void* data, CollisionCallBack<S> cb);                  // self (N²-pruned)
collide(obj,  void* data, cb);                                  // one vs all
collide(other_manager, void* data, cb);                         // set vs set
distance(…)  // same three shapes
```

Callbacks are **raw function pointers**, not `std::function`:

```cpp
template <typename S>
using CollisionCallBack = bool (*)(CollisionObject<S>*, CollisionObject<S>*, void* cdata);
template <typename S>
using DistanceCallBack  = bool (*)(CollisionObject<S>*, CollisionObject<S>*, void* cdata, S& dist);
```

Returning `true` means **stop the traversal**. State goes through the `void*`.
`broadphase/default_broadphase_callbacks.h` gives you ready-made ones:

```cpp
fcl::DefaultCollisionData<double> cdata;
cdata.request.enable_contact = false;
manager->collide(&cdata, fcl::DefaultCollisionFunction<double>);
if (cdata.result.isCollision()) { /* ... */ }
```

`DefaultCollisionFunction` stops early once `numContacts() >= num_max_contacts`
(and cost is disabled); `DefaultDistanceFunction` stops as soon as distance ≤ 0.
Note the `DistanceCallBack`'s `S& dist` out-parameter: the manager uses it to
shrink its own pruning radius as the minimum tightens, so a callback that fails
to write it defeats the acceleration.

`DynamicAABBTreeCollisionManager` also exposes `octree_as_geometry_collide` /
`octree_as_geometry_distance`: when false (default) the manager descends *into* the
octree during broadphase culling; when true it treats the whole octree as one
opaque geometry and hands it to the narrow phase.

The continuous analogue is `BroadPhaseContinuousCollisionManager<S>` over
`ContinuousCollisionObject<S>`.

---

## 11. Octree and uncertain geometry

`OcTree<S>` wraps a `shared_ptr<const octomap::OcTree>`. Beyond the tree it adds
FCL-side occupancy semantics:

```cpp
setOccupancyThres(d); setFreeThres(d); setCellDefaultOccupancy(d);
bool isNodeOccupied/isNodeFree/isNodeUncertain(const OcTreeNode*) const;
```

These are **FCL's** thresholds, not octomap's. A cell is *occupied*, *free*, or
*uncertain*; `CollisionGeometry::isOccupied()/isFree()/isUncertain()` and
`threshold_occupied`/`threshold_free` are the generic hooks. Traversals use them to
skip free space and to treat uncertain space conservatively — which is the point:
**FCL can reason about sensor data that is only probably there.**

`detail::OcTreeSolver` (`traversal/octree/octree_solver.h`) implements octree-vs-
{shape, mesh, octree} for both collision and distance by recursive descent, with
`computeChildBV()` giving a child cell's AABB from its parent's. `toBoxes()`
flattens occupied cells to `[x,y,z,size,cost,threshold]` sextuples if you need to
hand the scene to something else.

Remember from §9.3: **octree has no CCD support.**

---

## 12. Sampling and RNG

`math/sampler/` and `math/rng.h` exist because FCL was written alongside sampling
planners:

| Sampler | Space |
|---|---|
| `SamplerR<S, N>` | ℝⁿ box |
| `SamplerSE2`, `SamplerSE2_disk` | SE(2), rectangular or disk bounds |
| `SamplerSE3Euler`, `SamplerSE3Quat` | SE(3), Euler or quaternion parameterisation |
| `SamplerSE3Euler_ball`, `SamplerSE3Quat_ball` | SE(3) restricted to a ball of radius `r` |

`RNG<S>` provides uniform, Gaussian, half-normal, quaternion, and Euler-SO(3)
sampling; `detail::seed` centralises seeding for reproducibility. In practice most
users let OMPL or their own planner own sampling, but the `_ball` variants are
genuinely handy for local perturbation around a nominal pose.

---

## 13. Putting it together: canonical usage

### 13.1 Pairwise collision

```cpp
#include "fcl/narrowphase/collision.h"
#include "fcl/geometry/shape/box.h"
#include "fcl/geometry/shape/sphere.h"
using namespace fcl;

auto box    = std::make_shared<Boxd>(0.1, 0.2, 0.3);
auto sphere = std::make_shared<Sphered>(0.05);

Transform3d X_WB = Transform3d::Identity();
Transform3d X_WS = Transform3d::Identity();
X_WS.translation() << 0.12, 0.0, 0.0;

CollisionRequestd req(/*num_max_contacts=*/4, /*enable_contact=*/true);
CollisionResultd  res;
std::size_t n = collide(box.get(), X_WB, sphere.get(), X_WS, req, res);

for (std::size_t i = 0; i < res.numContacts(); ++i) {
  const Contactd& c = res.getContact(i);
  // c.pos (world), c.normal (o1→o2), c.penetration_depth
}
```

### 13.2 Signed distance with witness points

```cpp
DistanceRequestd req(/*enable_nearest_points=*/true,
                     /*enable_signed_distance=*/true);
req.gjk_solver_type = GST_LIBCCD;    // native signed distance for primitive pairs
DistanceResultd res;
double d = distance(objA, objB, req, res);
// res.nearest_points[0], [1]  — world frame
// d < 0 ⇒ penetration depth (negated); see §8.2 for the guarantees
```

### 13.3 Mesh model

```cpp
auto mesh = std::make_shared<BVHModel<OBBRSSd>>();   // OBBRSS: collide AND distance
mesh->beginModel(num_tris, num_verts);
mesh->addSubModel(vertices, triangles);              // std::vector<Vector3d>, <Triangle>
mesh->endModel();
mesh->computeLocalAABB();
```

Pick the BV by query: `OBB` for collision only, `RSS`/`OBBRSS` if you also need
distance (`OBB` and `kDOP` have no mesh-mesh distance entry — §6).

### 13.4 Broadphase, robot vs world

```cpp
DynamicAABBTreeCollisionManagerd robot, world;
for (auto* o : robot_links) robot.registerObject(o);
for (auto* o : obstacles)   world.registerObject(o);
robot.setup(); world.setup();

// every control tick, after forward kinematics:
for (auto* o : robot_links) { o->setTransform(X_WL[o]); o->computeAABB(); }
robot.update(robot_links);                 // only the movers

DefaultCollisionData<double> cd;
robot.collide(&world, &cd, DefaultCollisionFunction<double>);
```

`setTransform()` does **not** refresh the cached world AABB — you must call
`computeAABB()`, then tell the manager via `update()`. Forgetting either is the
classic "my collision checks use last frame's poses" bug.

### 13.5 Continuous collision

```cpp
ContinuousCollisionRequestd req;
req.ccd_solver_type = CCDC_CONSERVATIVE_ADVANCEMENT;
req.ccd_motion_type = CCDM_LINEAR;
req.num_max_iterations = 100;
req.toc_err = 1e-4;

ContinuousCollisionResultd res;
continuousCollide(geomA.get(), X_WA_beg, X_WA_end,
                  geomB.get(), X_WB_beg, X_WB_end, req, res);
if (res.is_collide) {
  double toc = res.time_of_contact;        // in [0,1]
  // res.contact_tf1 / contact_tf2: the poses at first contact
}
```

---

## 14. How manipulation uses FCL

FCL is the *geometric kernel* under a manipulation stack. Everything
robot-specific lives above it. The interface between the two is small and worth
stating precisely:

```
      ┌────────────────────────────────────────────────┐
      │  robot model + forward kinematics              │   you / MoveIt / Drake /
      │     q  ──FK──▶  { X_WL : link L }              │   Pinocchio / KDL
      └──────────────────────┬─────────────────────────┘
                             │  Transform3<S> per body
                             ▼
      ┌────────────────────────────────────────────────┐
      │  FCL: CollisionObject per body, managers,       │
      │  collide() / distance() / continuousCollide()   │
      └──────────────────────┬─────────────────────────┘
                             │  bool | d, p₁, p₂, n̂ | toc
                             ▼
      ┌────────────────────────────────────────────────┐
      │  planner (sampling/optimisation) or controller  │
      │  (QP / CBF / potential field / impedance)       │
      └────────────────────────────────────────────────┘
```

### 14.1 Building the collision model

One `CollisionObject` per collision body: robot links, the gripper, the grasped
payload (re-parented to the gripper once grasped), static environment, and
perceived obstacles. Practical choices:

- **Approximate links with primitives.** Capsules and boxes hit the closed-form
  kernels of §8.1: no iteration, exact witness points, stable normals. A capsule
  model of a 7-DoF arm is typically an order of magnitude faster *and* better
  behaved than the visual meshes.
- **If you must use meshes, make them convex.** Convex decomposition offline,
  then one `Convex` per piece. `Convex` plus GJK is far better conditioned than a
  BVH of a concave soup, and `findExtremeVertex()`'s hill-climb rewards you for
  watertightness.
- **Pad the geometry**, don't rely on a distance threshold alone. Inflating a
  capsule's radius by the safety margin turns "clearance ≥ ε" into a plain
  collision query, which is cheaper and avoids the signed-distance pitfalls.
- Hold geometries in `shared_ptr` and **share them across identical links** — the
  BVH/`Convex` data is the expensive part, the pose is free.

Two managers is the usual layout: one for the robot, one for the world, queried
against each other. Self-collision is the robot manager's self-`collide()`, with
adjacent-link and known-safe pairs filtered in your callback (the
allowed-collision-matrix concept is *yours*, not FCL's — FCL gives you
`CollisionObject::setUserData()` to hang the link identity off, and the
`void* cdata` to carry the filter).

### 14.2 Sampling-based planning

The planner needs one predicate: `isValid(q)`.

```cpp
bool isValid(const Eigen::VectorXd& q) {
  fk(q, X_WL);                                   // your kinematics
  for (auto* o : links) { o->setTransform(X_WL[o]); o->computeAABB(); }
  robot_mgr.update(links);

  DefaultCollisionData<double> cd;
  cd.request.num_max_contacts = 1;
  cd.request.enable_contact   = false;           // ← binary query: cheapest path
  robot_mgr.collide(&world_mgr, &cd, DefaultCollisionFunction<double>);
  if (cd.result.isCollision()) return false;

  DefaultCollisionData<double> self;
  self.request.num_max_contacts = 1;
  self.request.enable_contact   = false;
  robot_mgr.collide(&self, &MyFilteredSelfCollisionFn);   // honours your ACM
  return !self.result.isCollision();
}
```

This call sits in the planner's innermost loop — it is typically **80–95 % of
planning time**. The levers, in order of impact:

1. `enable_contact = false`, `num_max_contacts = 1`, `enable_cost = false`.
2. Primitive geometry over meshes (§14.1).
3. `DynamicAABBTreeCollisionManager` with `update(moved_only)`.
4. Early-exit callbacks — return `true` the instant you know the answer.
5. `GST_LIBCCD` vs `GST_INDEP`: benchmark both on *your* geometry; neither wins
   universally.

**Edge validation** is where correctness is usually lost. Interpolating `q` and
checking samples can tunnel through thin obstacles. Three options, increasing
cost and rigour:

- dense discrete sampling with step chosen from the maximum link swept speed —
  cheap, no guarantee;
- `continuousCollide()` per link pair with `CCDC_CONSERVATIVE_ADVANCEMENT` and
  `CCDM_LINEAR` — a guarantee *for that pose-interpolated motion*, and `toc` tells
  you where to split the edge;
- a swept-volume geometry (hull of the link over the segment) checked discretely —
  conservative, and constructible with `generateBVHModel(..., FinalizeModel::DONT)`.

Note the honest caveat on the middle option: FCL's CCD interpolates **Cartesian
pose**, while the robot actually moves along a **joint-space** path. For a single
segment of a joint-interpolated trajectory the two differ. The usual engineering
answer is to subdivide in joint space until the Cartesian interpolation error is
below the safety margin, and let CCD cover the residual.

### 14.3 Optimisation-based planning (CHOMP / TrajOpt / STOMP style)

These need a **signed distance field with gradients**, which is the point where
FCL stops handing you the answer and starts handing you the ingredients.

FCL gives, per body pair, `d` and witness points `p₁ ∈ A`, `p₂ ∈ B`. From those:

```
n̂ = (p₂ − p₁) / ‖p₂ − p₁‖              (for d > 0)
```

and by the envelope theorem — the witness points are the active constraint —

```
∂d/∂q = n̂ᵀ ( J_B(p₂) − J_A(p₁) )
```

where `J_X(p)` is the translational Jacobian of point `p` on body `X`. For a robot
link `A` against a static obstacle `B` this collapses to `−n̂ᵀ J_A(p₁)`. **That one
line is how essentially every distance-based trajectory optimiser couples FCL to
its solver**, and it is why `enable_nearest_points` matters so much more than it
looks.

Caveats that bite in practice:

- **Nonsmoothness.** `d(q)` is only piecewise differentiable. The witness pair
  jumps when the active feature changes (vertex → edge → face), so the gradient is
  discontinuous. Optimisers handle this with trust regions, smoothing, or by
  keeping several nearby pairs active instead of only the minimum.
- **Validity in penetration.** Inside contact the formula needs the *penetration
  normal*, and FCL's negative distance for meshes comes from the EPA workaround
  (§8.2) with witness points that are **not on the surfaces** and are both set to
  the same point. The direction is still usable; the two points are not.
  For primitive pairs with `GST_LIBCCD` you get proper signed distance and surface
  witness points — another argument for primitive collision models.
- **Everything is per-pair.** You call `distance()` for each pair you want a
  gradient for, so your cost function's sparsity pattern is yours to manage. Use
  the broadphase to pick the pairs within the cost's influence radius, then do the
  narrow-phase distance calls only on those.

### 14.4 Reactive control

In a 100 Hz–1 kHz loop the geometry query must be bounded-time. The typical shape:

```
each tick:
  FK from measured q            → X_WL
  update poses + AABBs + manager
  broadphase cull               → candidate pairs within influence radius r_infl
  narrow-phase distance per pair → (dᵢ, p₁ᵢ, p₂ᵢ, n̂ᵢ)
  build constraints / forces from (dᵢ, n̂ᵢ, Jᵢ)
  solve QP (or evaluate the field) → q̇ / τ
```

How the geometric output enters the controller:

- **Velocity-level QP / inverse kinematics.** Per pair, a linear inequality that
  keeps the approach rate bounded:
  `n̂ᵢᵀ J_i q̇ ≥ −α (dᵢ − d_safe)`. This is a discrete-time control-barrier-function
  constraint with `h(q) = d(q) − d_safe`; `α` sets how aggressively the robot is
  allowed to close the remaining gap. FCL supplies `dᵢ` and `n̂ᵢ`; you supply `Jᵢ`.
- **Artificial potential fields / RMPs.** A repulsive potential
  `U = ½ k (1/d − 1/r_infl)²` for `d < r_infl`, with force `−∂U/∂q` assembled from
  the same `n̂ᵢᵀJᵢ`.
- **Impedance / compliance on contact.** `Contact::normal` and
  `penetration_depth` from `collide()` drive the virtual spring. Here is where
  EPA's normal instability under deep penetration shows up as chatter — clamp the
  depth, low-pass the normal, and prefer primitive geometry.
- **Safety supervisor.** `continuousCollide()` over the *commanded* motion for the
  next control interval, rejecting or scaling the command when `toc < 1`. Note
  §9.3: no octree support, so perceived obstacles must be converted to boxes or
  primitives first.

What FCL does **not** give you, and you must build:

- temporal smoothing/hysteresis on the witness pair and normal;
- any guarantee that `d(q)` is continuous across feature changes;
- derivatives (see §14.3);
- a bound on query time — GJK/EPA iteration counts are data-dependent. If you need
  hard real-time, use primitive-only models (closed-form, fixed cost) and cap the
  pair count from the broadphase.

Two FCL-specific optimisations matter a lot here, because a controller re-queries
*nearly the same configuration* every tick:

1. **GJK warm start** — carry `CollisionResult::cached_gjk_guess` into the next
   tick's `CollisionRequest::cached_gjk_guess` with `enable_cached_gjk_guess = true`.
2. **BVH front lists** — reuse `BVHFrontList` across ticks for mesh pairs
   (`test/test_fcl_frontlist.cpp` shows the pattern).

### 14.5 Grasping, placement, and perception

- **Grasp / placement feasibility:** `collide()` with `enable_contact = true`
  between gripper geometry and object, and between object and support surface.
  Penetration depth tells you how far to retract. Once grasped, make the object a
  child of the gripper frame: update its `CollisionObject` transform from the
  gripper's FK and *remove* the gripper-object pair from the checked set.
- **Perceived scene:** `OcTree` consumes octomap directly and keeps occupancy
  semantics, so free space is skipped and uncertain space can be treated
  conservatively. `OcTree::getNodeByQueryCellId()` maps a query result back to the
  specific cell. For CCD or for anything that needs gradients, flatten with
  `toBoxes()` and register the boxes in a `DynamicAABBTreeCollisionManager`.
- **Uncertain geometry:** `Variance3` + `BVHExpand()` inflate a BVH by `r`
  standard deviations — a principled way to be conservative about noisy mesh
  reconstructions.

### 14.6 Configuration cheat sheet

| Use case | Geometry | BV | Solver | Request settings |
|---|---|---|---|---|
| Planner validity check | capsules/boxes | — | `GST_LIBCCD` | `enable_contact=false`, `num_max_contacts=1` |
| Planner edge check | capsules/boxes | — | — | `CCDC_CONSERVATIVE_ADVANCEMENT` + `CCDM_LINEAR` |
| Trajectory optimisation | primitives or `Convex` | `OBBRSS` if mesh | `GST_LIBCCD` | `enable_nearest_points=true`, `enable_signed_distance=true` |
| Reactive control (distance) | primitives only | — | `GST_LIBCCD` | as above + cached GJK guess |
| Contact/impedance | primitives | — | `GST_LIBCCD` | `enable_contact=true`, clamp depth |
| Mesh-mesh distance | `BVHModel` | `RSS` or `OBBRSS` | either | `enable_nearest_points=true` |
| Mesh-mesh collision only | `BVHModel` | `OBB` | either | `enable_contact=false` |
| Perceived obstacles, discrete | `OcTree` | — | either | tune `setOccupancyThres` |
| Perceived obstacles, CCD | boxes from `toBoxes()` | — | either | octree has no CCD |

---

## 15. Pitfalls, collected

1. **`setTransform()` does not update the cached AABB.** Call `computeAABB()`,
   then `manager->update()`.
2. **Unsupported type pairs warn to `stderr` and return "no collision".** Check the
   function matrix for exotic combinations; `BVHModel<OBB>` and
   `BVHModel<KDOP<…>>` have no mesh-mesh *distance*.
3. **Negative distance is `-1` unless you ask.** Set
   `enable_signed_distance = true`, and know which of SD_1/SD_2 and NP/NP_X your
   pair falls under (§8.2).
4. **Default CCD is the weakest CCD.** `CCDC_NAIVE` + `CCDM_TRANS` samples and
   ignores rotation. `CCDC_RAY_SHOOTING` is an empty stub.
5. **No CCD for `OcTree` or `Ellipsoid`.**
6. **`Transform3` is an isometry.** No scaling. A non-orthonormal `linear()` block
   produces silent garbage.
7. **`Convex` validation is incomplete.** Non-planar faces, wrong winding, and
   non-convexity are not all caught; a failed topology check silently degrades
   `findExtremeVertex()` to a linear scan.
8. **Results accumulate.** Reuse a `DistanceResult` without `clear()` and you keep
   a stale minimum.
9. **kDOPs are orientation-locked.** Fine for non-rotating geometry, wrong for a
   rotating link.
10. **`gjk_tolerance` defaults to `1e-6` for backward compatibility**, which is
    loose for `double` and tight for `float`. Prefer `constants<S>::eps_34()`-style
    scaling.
11. **EPA normals can flip** under deep or degenerate penetration. Filter before
    feeding a controller.
12. **Broadphase distance callbacks must write `dist`** — the manager uses it to
    prune.

---

## 16. Where to look in the code

| You want | Read |
|---|---|
| The public API | `narrowphase/collision.h`, `distance.h`, `continuous_collision.h` |
| What pairs are supported | `narrowphase/detail/{collision,distance,conservative_advancement}_func_matrix-inl.h` |
| Scalar/Eigen conventions | `common/types.h`, `math/constants.h` |
| Frame conventions, best-documented BV | `math/bv/RSS.h` |
| SAT | `math/bv/OBB-inl.h` (`obbDisjoint`) |
| Support functions / Minkowski | `narrowphase/detail/convexity_based_algorithm/{minkowski_diff,support}.h` |
| GJK / EPA | same directory, `gjk.h`, `epa.h`, `gjk_libccd-inl.h` |
| Exact primitive pairs | `narrowphase/detail/primitive_shape_algorithm/` |
| Mesh build & refit | `geometry/bvh/BVH_model-inl.h`, `detail/BV_{splitter,fitter}.h` |
| Traversal skeleton | `narrowphase/detail/traversal/traversal_recurse-inl.h` |
| Motion bounds (the CCD math) | `math/motion/{tbv,triangle}_motion_bound_visitor-inl.h` |
| Conservative advancement loop | `.../traversal/distance/mesh_conservative_advancement_traversal_node-inl.h` |
| Broadphase default callbacks | `broadphase/default_broadphase_callbacks.h` |
| Worked examples | `test/` — `test_fcl_distance.cpp`, `test_fcl_signed_distance.cpp`, `test_fcl_broadphase_*.cpp`, `test_fcl_frontlist.cpp`, `test_fcl_octomap_*.cpp` |

---

## 17. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest
```

Relevant options: `FCL_WITH_OCTOMAP` (gates `OcTree`), `FCL_ENABLE_PROFILING`
(`common/profiler.h`), `FCL_TREAT_WARNINGS_AS_ERRORS`, `FCL_HIDE_ALL_SYMBOLS`,
`BUILD_TESTING`. Default build type is `Release` — and it matters: this is a
template-heavy library whose performance depends entirely on inlining.
