"""GH Python | Link Grid Display ONLY (no filtering, no sorting)

Inputs:
    links: DataTree (branches = sets/clusters, items = closed polyline curves)
    max_sets: int (<=0 means all)
    enable: bool
    facets: int (unused by Brep.CreatePipe; kept if you want later)
Outputs:
    pipes: DataTree[object]  (Brep)
    colors: DataTree[object] (Color)
    pipe_r_used: float
    cluster_pts: list of Point3d center points for each displayed cluster
"""

import Rhino.Geometry as rg
from System.Drawing import Color
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import math

# ----------------------------
# Defaults / coercion
# ----------------------------
def _as_int(x, default):
    try: return int(x)
    except: return default

def _as_float(x, default):
    try: return float(x)
    except: return default

def _as_bool(x, default):
    try: return bool(x)
    except: return default

max_sets = _as_int(globals().get("max_sets", 0), 0)
enable   = _as_bool(globals().get("enable", True), True)
links    = globals().get("links", None)

# ----------------------------
# Helpers
# ----------------------------
def to_curve(obj):
    if obj is None:
        return None
    if isinstance(obj, rg.Curve):
        return obj
    if isinstance(obj, rg.Polyline):
        return rg.PolylineCurve(obj)
    if isinstance(obj, rg.PolylineCurve):
        return obj
    try:
        return rg.Curve.TryConvert(obj)
    except:
        return None

def curve_points(curve):
    try:
        ok, pl = curve.TryGetPolyline()
        if ok:
            return list(pl)
    except:
        pass
    return None

def min_adjacent_dist(pts):
    if not pts or len(pts) < 2:
        return None
    m = None
    n = len(pts)
    for i in range(n):
        d = pts[i].DistanceTo(pts[(i + 1) % n])
        if d <= 0:
            continue
        if (m is None) or (d < m):
            m = d
    return m

def cluster_bbox(curves):
    bb = rg.BoundingBox.Empty
    for c in curves:
        if c:
            bb = rg.BoundingBox.Union(bb, c.GetBoundingBox(True))
    return bb

def make_pipe_breps(curve, radius):
    if curve is None or (not curve.IsValid):
        return []
    try:
        breps = rg.Brep.CreatePipe(curve, radius, True, rg.PipeCapMode.Flat, True, 0.01, 0.01)
        return list(breps) if breps else []
    except:
        return []

def palette_color(i):
    pal = [
        Color.FromArgb(230,  60,  60),
        Color.FromArgb( 60, 170, 230),
        Color.FromArgb( 80, 200, 120),
        Color.FromArgb(240, 190,  60),
        Color.FromArgb(170,  90, 220),
        Color.FromArgb(255, 120,  40),
        Color.FromArgb( 60, 220, 200),
        Color.FromArgb(220,  90, 140),
        Color.FromArgb(140, 140, 140),
    ]
    return pal[i % len(pal)]

# ----------------------------
# Outputs
# ----------------------------
pipes = DataTree[object]()
colors = DataTree[object]()
pipe_r_used = 0.0
cluster_pts = []

if (not enable) or (links is None):
    pass
else:
    # Read clusters from DataTree
    clusters = []
    for bi in range(links.BranchCount):
        br = links.Branch(bi)
        cs = []
        for item in br:
            c = to_curve(item)
            if c:
                cs.append(c)
        clusters.append(cs)

    # Apply cap (NO sorting)
    if max_sets and max_sets > 0:
        clusters = clusters[:max_sets]

    if clusters and clusters[0]:
        pts0 = curve_points(clusters[0][0])
        dmin = min_adjacent_dist(pts0)
        if dmin and dmin > 0:
            pipe_r_used = 0.35 * dmin
    if pipe_r_used <= 0:
        pipe_r_used = 0.6

    # Square grid size
    N = len(clusters)
    grid_n = int(math.ceil(math.sqrt(N))) if N > 0 else 1

    # Cell size from max bbox XY span
    bbs = [cluster_bbox(cs) for cs in clusters]
    max_w = 0.0
    max_h = 0.0
    for bb in bbs:
        if not bb.IsValid:
            continue
        dx = bb.Max.X - bb.Min.X
        dy = bb.Max.Y - bb.Min.Y
        if dx > max_w: max_w = dx
        if dy > max_h: max_h = dy

    base = max(max_w, max_h)
    pad = base * 0.25 if base > 0 else 5.0
    cell_w = max_w + pad
    cell_h = max_h + pad

    # Place in +X/+Y, first at (0,0)
    for i, curves in enumerate(clusters):
        if not curves:
            continue

        row = i // grid_n
        col = i % grid_n

        bb = bbs[i]
        if not bb.IsValid:
            continue

        xform = rg.Transform.Translation(
            -bb.Min.X + col * cell_w,
            -bb.Min.Y + row * cell_h,
            -bb.Min.Z
        )

        for j, c in enumerate(curves):
            cc = c.DuplicateCurve()
            cc.Transform(xform)

            breps = make_pipe_breps(cc, pipe_r_used)
            out_path = GH_Path(i, j)

            colr = palette_color(j)
            for b in breps:
                pipes.Add(b, out_path)
                colors.Add(colr, out_path)

        # Center point of the translated cluster bounding box
        tbb = rg.BoundingBox.Empty
        for c in curves:
            cc = c.DuplicateCurve()
            cc.Transform(xform)
            tbb = rg.BoundingBox.Union(tbb, cc.GetBoundingBox(True))
        if tbb.IsValid:
            cluster_pts.append(tbb.Center)
