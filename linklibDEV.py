"""A library with function to normalize and analyze links.

  A link is given as a list of paths, each path consisting of a list of 3D points
  with integer coordinates.
  Paths are implicilty closed, without duplicating the first point at the end.
  Adjacent points in the path must be adjacent in the lattice
  (no lattice points skipped on a longer segment; this assumption is needed to ensure
  proper detection of intersections).
  Paths on external input need not be in the primary sublattice
  (relevant for FCC and BCC), but they are moved there where relevant. 
  
  TODO: Assumptions per function need to be reconsidered.
  
  This file needs to be put in the Rhino Lib folder.
  On my Mac its location is: /Applications/Rhino 7.app/Contents/Frameworks/RhCore.framework/Versions/Current/Resources/ManagedPlugins/RhinoDLR_Python.rhp/Lib
  
  In GH Python components, do
  
  import linklib as ll
  reload(ll)  # for debugging; when you update linklib.py without restarting Rhino
"""
_version = "2024.08.06"

#from typing import Any, Callable
from itertools import groupby, islice
from functools import reduce
try: 
    import Rhino.Geometry as rg
    import Grasshopper as gh
    import ghpythonlib.treehelpers as th
except:
    print("Working outside Rhino/Grasshopper.  Some functionality is not available")

import ctypes
import os as _os
_dll_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "normalize_link_c.dll")
_normalize_c = None
try:
    _lib = ctypes.CDLL(_dll_path)
    _lib.normalize_links_batch_c.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    _lib.normalize_links_batch_c.restype = ctypes.c_int
    _normalize_c = _lib
    print("Loaded C accelerator:", _dll_path)
except Exception as _e:
    print("C accelerator not available:", _e)

Polyline = "rg.Polyline"
Vector = "tuple[int, int, int]"
Path = "list[Vector]"
Link = "list[Path]"
Symmetry = "Callable[[int, int, int], tuple[int, int, int]]"
PermSign = "tuple[tuple[int, int, int], tuple[int, int, int]]"
Group = "list[int]"

ID_PERM_SIGN = ((0, 1, 2), (1, 1, 1))  # identity tranformation
SYMM_INDEX = range(48)

PERM_SIGN = [
    ((0, 1, 2), (1, 1, 1)),
    ((0, 1, 2), (1, 1, -1)),
    ((0, 1, 2), (1, -1, 1)),
    ((0, 1, 2), (1, -1, -1)),
    ((0, 1, 2), (-1, 1, 1)),
    ((0, 1, 2), (-1, 1, -1)),
    ((0, 1, 2), (-1, -1, 1)),
    ((0, 1, 2), (-1, -1, -1)),
    ((0, 2, 1), (1, 1, 1)),
    ((0, 2, 1), (1, 1, -1)),
    ((0, 2, 1), (1, -1, 1)),
    ((0, 2, 1), (1, -1, -1)),
    ((0, 2, 1), (-1, 1, 1)),
    ((0, 2, 1), (-1, 1, -1)),
    ((0, 2, 1), (-1, -1, 1)),
    ((0, 2, 1), (-1, -1, -1)),
    ((1, 0, 2), (1, 1, 1)),
    ((1, 0, 2), (1, 1, -1)),
    ((1, 0, 2), (1, -1, 1)),
    ((1, 0, 2), (1, -1, -1)),
    ((1, 0, 2), (-1, 1, 1)),
    ((1, 0, 2), (-1, 1, -1)),
    ((1, 0, 2), (-1, -1, 1)),
    ((1, 0, 2), (-1, -1, -1)),
    ((1, 2, 0), (1, 1, 1)),
    ((1, 2, 0), (1, 1, -1)),
    ((1, 2, 0), (1, -1, 1)),
    ((1, 2, 0), (1, -1, -1)),
    ((1, 2, 0), (-1, 1, 1)),
    ((1, 2, 0), (-1, 1, -1)),
    ((1, 2, 0), (-1, -1, 1)),
    ((1, 2, 0), (-1, -1, -1)),
    ((2, 0, 1), (1, 1, 1)),
    ((2, 0, 1), (1, 1, -1)),
    ((2, 0, 1), (1, -1, 1)),
    ((2, 0, 1), (1, -1, -1)),
    ((2, 0, 1), (-1, 1, 1)),
    ((2, 0, 1), (-1, 1, -1)),
    ((2, 0, 1), (-1, -1, 1)),
    ((2, 0, 1), (-1, -1, -1)),
    ((2, 1, 0), (1, 1, 1)),
    ((2, 1, 0), (1, 1, -1)),
    ((2, 1, 0), (1, -1, 1)),
    ((2, 1, 0), (1, -1, -1)),
    ((2, 1, 0), (-1, 1, 1)),
    ((2, 1, 0), (-1, 1, -1)),
    ((2, 1, 0), (-1, -1, 1)),
    ((2, 1, 0), (-1, -1, -1)),
]

INDEX = {
    ((0, 1, 2), (1, 1, 1)): 0,
    ((0, 1, 2), (1, 1, -1)): 1,
    ((0, 1, 2), (1, -1, 1)): 2,
    ((0, 1, 2), (1, -1, -1)): 3,
    ((0, 1, 2), (-1, 1, 1)): 4,
    ((0, 1, 2), (-1, 1, -1)): 5,
    ((0, 1, 2), (-1, -1, 1)): 6,
    ((0, 1, 2), (-1, -1, -1)): 7,
    ((0, 2, 1), (1, 1, 1)): 8,
    ((0, 2, 1), (1, 1, -1)): 9,
    ((0, 2, 1), (1, -1, 1)): 10,
    ((0, 2, 1), (1, -1, -1)): 11,
    ((0, 2, 1), (-1, 1, 1)): 12,
    ((0, 2, 1), (-1, 1, -1)): 13,
    ((0, 2, 1), (-1, -1, 1)): 14,
    ((0, 2, 1), (-1, -1, -1)): 15,
    ((1, 0, 2), (1, 1, 1)): 16,
    ((1, 0, 2), (1, 1, -1)): 17,
    ((1, 0, 2), (1, -1, 1)): 18,
    ((1, 0, 2), (1, -1, -1)): 19,
    ((1, 0, 2), (-1, 1, 1)): 20,
    ((1, 0, 2), (-1, 1, -1)): 21,
    ((1, 0, 2), (-1, -1, 1)): 22,
    ((1, 0, 2), (-1, -1, -1)): 23,
    ((1, 2, 0), (1, 1, 1)): 24,
    ((1, 2, 0), (1, 1, -1)): 25,
    ((1, 2, 0), (1, -1, 1)): 26,
    ((1, 2, 0), (1, -1, -1)): 27,
    ((1, 2, 0), (-1, 1, 1)): 28,
    ((1, 2, 0), (-1, 1, -1)): 29,
    ((1, 2, 0), (-1, -1, 1)): 30,
    ((1, 2, 0), (-1, -1, -1)): 31,
    ((2, 0, 1), (1, 1, 1)): 32,
    ((2, 0, 1), (1, 1, -1)): 33,
    ((2, 0, 1), (1, -1, 1)): 34,
    ((2, 0, 1), (1, -1, -1)): 35,
    ((2, 0, 1), (-1, 1, 1)): 36,
    ((2, 0, 1), (-1, 1, -1)): 37,
    ((2, 0, 1), (-1, -1, 1)): 38,
    ((2, 0, 1), (-1, -1, -1)): 39,
    ((2, 1, 0), (1, 1, 1)): 40,
    ((2, 1, 0), (1, 1, -1)): 41,
    ((2, 1, 0), (1, -1, 1)): 42,
    ((2, 1, 0), (1, -1, -1)): 43,
    ((2, 1, 0), (-1, 1, 1)): 44,
    ((2, 1, 0), (-1, 1, -1)): 45,
    ((2, 1, 0), (-1, -1, 1)): 46,
    ((2, 1, 0), (-1, -1, -1)): 47,
}

CUBE_SYMMETRY = [
    lambda x, y, z: (x, y, z),
    lambda x, y, z: (x, y, -z),
    lambda x, y, z: (x, -y, z),
    lambda x, y, z: (x, -y, -z),
    lambda x, y, z: (-x, y, z),
    lambda x, y, z: (-x, y, -z),
    lambda x, y, z: (-x, -y, z),
    lambda x, y, z: (-x, -y, -z),
    lambda x, y, z: (x, z, y),
    lambda x, y, z: (x, z, -y),
    lambda x, y, z: (x, -z, y),
    lambda x, y, z: (x, -z, -y),
    lambda x, y, z: (-x, z, y),
    lambda x, y, z: (-x, z, -y),
    lambda x, y, z: (-x, -z, y),
    lambda x, y, z: (-x, -z, -y),
    lambda x, y, z: (y, x, z),
    lambda x, y, z: (y, x, -z),
    lambda x, y, z: (y, -x, z),
    lambda x, y, z: (y, -x, -z),
    lambda x, y, z: (-y, x, z),
    lambda x, y, z: (-y, x, -z),
    lambda x, y, z: (-y, -x, z),
    lambda x, y, z: (-y, -x, -z),
    lambda x, y, z: (y, z, x),
    lambda x, y, z: (y, z, -x),
    lambda x, y, z: (y, -z, x),
    lambda x, y, z: (y, -z, -x),
    lambda x, y, z: (-y, z, x),
    lambda x, y, z: (-y, z, -x),
    lambda x, y, z: (-y, -z, x),
    lambda x, y, z: (-y, -z, -x),
    lambda x, y, z: (z, x, y),
    lambda x, y, z: (z, x, -y),
    lambda x, y, z: (z, -x, y),
    lambda x, y, z: (z, -x, -y),
    lambda x, y, z: (-z, x, y),
    lambda x, y, z: (-z, x, -y),
    lambda x, y, z: (-z, -x, y),
    lambda x, y, z: (-z, -x, -y),
    lambda x, y, z: (z, y, x),
    lambda x, y, z: (z, y, -x),
    lambda x, y, z: (z, -y, x),
    lambda x, y, z: (z, -y, -x),
    lambda x, y, z: (-z, y, x),
    lambda x, y, z: (-z, y, -x),
    lambda x, y, z: (-z, -y, x),
    lambda x, y, z: (-z, -y, -x),
]

INVERSE = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    9,
    11,
    12,
    14,
    13,
    15,
    16,
    17,
    20,
    21,
    18,
    19,
    22,
    23,
    32,
    36,
    33,
    37,
    34,
    38,
    35,
    39,
    24,
    26,
    28,
    30,
    25,
    27,
    29,
    31,
    40,
    44,
    42,
    46,
    41,
    45,
    43,
    47,
]

# Precomputed lookup: maps tuple of symmetry matrix rows -> index
# Built from CUBE_SYMMETRY so symmetry_to_index() is O(1) instead of O(48)
SYMMETRY_MATRIX_TO_INDEX = {
    (f(1, 0, 0), f(0, 1, 0), f(0, 0, 1)): i
    for i, f in enumerate(CUBE_SYMMETRY)
}


def compose(*perm_sign):
    # type: (*PermSign) -> PermSign
    """Compose zero or more symmetries in perm-sign representation.
    """
    result_p, result_s = ID_PERM_SIGN
    
    for p, s in perm_sign:
        result_p = tuple(result_p[i] for i in p)
        result_s = tuple(result_s[p[i]] * s[i] for i in range(3))
    
    return result_p, result_s


def compose_symmetries(symm1, symm2):
    # type: (Symmetry, Symmetry) -> Symmetry
    """First symm1, then symm2.
    """
    def result(x, y, z):
        return symm2(*symm1(x, y, z))
        
    return result


def inverse_symmetry(symm):
    # type: (Symmetry) -> Symmetry
    """Inverse of symm.
    """
    return CUBE_SYMMETRY[symmetry_to_index(symm)]


def symmetry_matrix(symm):
    # type: (Symmetry) -> list[Vector]
    """Apply symmetry to vectors (1, 0, 0), (0, 1, 0), and (0, 0, 1).
    
    >>> symmetry_matrix(CUBE_SYMMETRY[0])
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    >>> symmetry_matrix(CUBE_SYMMETRY[-1])
    [(0, 0, 1), (1, 0, 0), (0, -1, 0)]
    """
    return [symm(1, 0, 0), symm(0, 1, 0), symm(0, 0, 1)]


def symmetry_to_index(symm):
    # type: (Symmetry) -> Optional[int]
    """Determine the index of symm, if it is a cube symmetry,
    else None.
    
    Note that its inverse is CUBE_SYMMETRY[index].
    
    The images of vectors (1, 0, 0), (0, 1, 0), and (0, 0, 1)
    uniquely determine the symmetry.
    
    >>> symmetry_to_index(CUBE_SYMMETRY[0])
    0
    >>> symmetry_to_index(CUBE_SYMMETRY[-1])
    47
    """
    key = (symm(1, 0, 0), symm(0, 1, 0), symm(0, 0, 1))
    return SYMMETRY_MATRIX_TO_INDEX.get(key)


def transform_group(group, f):
    # type: (Group, Index) -> Group
    perm_sign = PERM_SIGN[f]
    inv = PERM_SIGN[INVERSE[f]]
    return sorted(INDEX[compose(perm_sign, PERM_SIGN[g], inv)] for g in group)    


def normalize_group(group):
    # type: (Group) -> tuple[Group, Index]
    return min(((transform_group(group, f), f) for f in SYMM_INDEX), key=lambda t:t[0])
    
    
def add_vectors(vec1, vec2):
    # type: (Vector, Vector) -> Vector
    """Add two vectors.
    
    >>> add_vectors((1, 3, 5), (0, -1, 1))
    (1, 2, 6)
    """
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return (x1 + x2, y1 + y2, z1 + z2)


def subtract_vectors(vec1, vec2):
    # type: (Vector, Vector) -> Vector
    """Subtract two vectors: vec1 - vec2.
    
    >>> subtract_vectors((1, 3, 7), (0, -1, 1))
    (1, 4, 6)
    """
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return (x1 - x2, y1 - y2, z1 - z2)


def min_vectors(vec1, vec2):
    # type: (Vector, Vector) -> Vector
    """Componentwise minimum of two vectors.
    
    >>> min_vectors((1, 2, 5), (1, 3, 4))
    (1, 2, 4)
    """
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return (min(x1, x2), min(y1, y2), min(z1, z2))


def max_vectors(vec1, vec2):
    # type: (Vector, Vector) -> Vector
    """Componentwise minimum of two vectors.
    
    >>> max_vectors((1, 2, 5), (1, 3, 4))
    (1, 3, 5)
    """
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return (max(x1, x2), max(y1, y2), max(z1, z2))


def square_length(vec):
    # type: (Vector) -> int
    """Compute square of vector length.
    
    >>> square_length((0, 6, 8))
    100
    """
    x, y, z = vec
    return x * x + y * y + z * z


def detect_lattice(path):
    # type: (Path) -> int
    """Detect lattice of path.
    result: 1 -> SC, 2 -> FCC, 3 -> BCC

    Assumptions:
    * path is in a single lattice
    * path makes only unit steps in lattice
    * len(path) >= 2
    
    >>> detect_lattice([(0, 0, 0), (1, 0, 0)])
    1
    >>> detect_lattice([(0, 0, 0), (1, 1, 0)])
    2
    >>> detect_lattice([(0, 0, 0), (1, 1, 1)])
    3
    """
    p0, p1 = path[0:2]
    return abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) + abs(p1[2] - p0[2])


def is_in_lattice(vec, lattice):
    # type: (Vec, int) -> bool
    """Determine whether vec is in lattice.
    lattice: 1 -> SC, 2 -> FCC, 3 -> BCC
    
    Assumptions:
    * lattice in [1, 2, 3]
    
    >>> is_in_lattice((0, 0, 0), 1)
    True
    >>> is_in_lattice((0, 0, 0), 2)
    True
    >>> is_in_lattice((0, 0, 0), 3)
    True
    >>> is_in_lattice((1, 0, 0), 1)
    True
    >>> is_in_lattice((1, 0, 0), 2)
    False
    >>> is_in_lattice((1, 1, 0), 2)
    True
    >>> is_in_lattice((1, 0, 0), 3)
    False
    >>> is_in_lattice((1, 1, 0), 3)
    False
    >>> is_in_lattice((1, 1, 1), 3)
    True
    """
#     return sum(map(lambda x: x % 2, vec)) % lattice == 0
    x, y, z = vec
    return (x % 2 + y % 2 + z % 2) % lattice == 0


def snap_to_lattice(vec, lattice):
    # type: (Vector, int) -> Vector
    """Determine lattice point nearest to vec.
    lattice:  1 -> SC, 2 -> FCC, 3 -> BCC
    
    >>> snap_to_lattice((0, 0, 0), 1)
    (0, 0, 0)
    >>> snap_to_lattice((0, 0, 0), 2)
    (0, 0, 0)
    >>> snap_to_lattice((0, 0, 0), 3)
    (0, 0, 0)
    >>> snap_to_lattice((1, 0, 0), 1)
    (1, 0, 0)
    >>> snap_to_lattice((1, 0, 0), 2)
    (0, 0, 0)
    >>> snap_to_lattice((1, 0, 0), 3)
    (0, 0, 0)
    >>> snap_to_lattice((0, 1, 0), 1)
    (0, 1, 0)
    >>> snap_to_lattice((0, 1, 0), 2)
    (-1, 1, 0)
    >>> snap_to_lattice((0, 1, 0), 3)
    (0, 0, 0)
    >>> snap_to_lattice((1, 1, 0), 1)
    (1, 1, 0)
    >>> snap_to_lattice((1, 1, 0), 2)
    (1, 1, 0)
    >>> snap_to_lattice((1, 1, 0), 3)
    (1, 1, -1)
    """
    if lattice == 1 or is_in_lattice(vec, lattice):
        return vec
    vec_x = add_vectors(vec, (-1, 0, 0))
    if lattice == 2 or is_in_lattice(vec_x, lattice):
        return vec_x
    vec_y = add_vectors(vec, (0, -1, 0))
    if is_in_lattice(vec_y, lattice):
        return vec_y
    vec_z = add_vectors(vec, (0, 0, -1))
#     if is_in_lattice(vec_z, lattice):
    return vec_z


def import_path(path):
    # type: (Polyline) -> Path
    """Convert a Rhino.Geometry.Polyline to a Path.
    
    Drop last point of path, and convert coordinates to int.
    """
    return [(int(point.X), int(point.Y), int(point.Z)) for point in path][:-1]


def import_link(link):
    # type: (list[Polyline]) -> list[Link]
    """Convert from Rhino.GH format to internal format.
    """
    return [import_path(path) for path in link]


def import_links(links):
    # type: (th.DataTree[Polyline]) -> list[Link]
    """Convert from Rhino.GH format to internal format.
    
    Turn tree into list of lists,
    drop last point of paths, and convert coordinates to int.
    """
    return [import_link(link)
            for link in th.tree_to_list(links, retrieve_base=None)
           ]


def export_path(path):
    # type: (Path) -> Polyline
    """Convert path to Rhino.
    """
    return rg.Polyline(rg.Point3d(*point) for point in path + [path[0]])


def export_paths(paths):
    # type: (list[Path]) -> list[Polyline]
    """Convert paths for GH.
    """
    return [export_path(path) for path in paths]


def export_link(link):
    # type: (Link) -> list[Polyline]
    """Convert internal format to Rhino.GH.
    
    Close paths (append first point) and put them in Array.
    """
    return [export_path(path) for path in link]


def create_transform(matrix):
    """Create a Transform object from a 3x3 orthonormal matrix.
    """
    # Create an identity Transform object from a value on the diagonal
    transform = rg.Transform(1.0)
    
    # Fill in the values from matrix
    for i in range(3):
        for j in range(3):
            transform[i, j] = matrix[i][j]
    
    return transform


def export_links(links):
    # type: (list[Link]) -> list[Polyline]
    """Convert internal format to Rhino.GH.
    
    Close paths (append first point) and put them in Array.
    """
    # the following does not work because Polyline is treated as list of points,
    # and expanded
#    return th.list_to_tree([export_link(link) for link in links], False, source=[])

    # create a DataTree from scratch
    result = gh.DataTree[object]()
    
    for branch_index, link in enumerate(links):
        # Create a new path for this branch
        path = gh.Kernel.Data.GH_Path(branch_index)
        
        # Add each Polyline in the branch to the data tree at the created path
        for polyline in link:
            result.Add(export_path(polyline), path)
            
    return result


def min_path(path):
    # type: (Path) -> Vector
    """Determine minimum corner of bounding box of path.
    
    >>> min_path([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    (0, 0, 0)
    """
    return reduce(min_vectors, path)


def max_path(path):
    # type: (Path) -> Vector
    """Determine maximum corner of bounding box of  path.
    
    >>> max_path([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    (1, 1, 1)
    """
    return reduce(max_vectors, path)


def transform_link(f, link):
    # type: (Callable[[Vector], Vector], Link) -> Link
    """Transform all points in the link by function f.
    
    Assumption: f preserves the lattice
    
    >>> link = [[(0, 0, 0), (1, 0, 1), (0, 1, 1)],
    ...         [(2, 0, 0), (3, 1, 0), (2, 2, 0), (1, 1, 0)]]
    >>> transform_link(lambda x, y, z: (x + 1, y + 3, z + 6), link)
    [[(1, 3, 6), (2, 3, 7), (1, 4, 7)], [(3, 3, 6), (4, 4, 6), (3, 5, 6), (2, 4, 6)]]
    """
    return [[f(*point) for point in path] for path in link]


def min_link(link):
    # type: (Link) -> Vector
    """Determine minimum corner of bounding box of link.

    >>> link = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...         [(0, 0, 0), (1, 1, 1), (0, 0, 2), (-1, -1, 1)]]
    >>> min_link(link)
    (-1, -1, 0)
    """
    return min_path(min_path(path) for path in link)


def max_link(link):
    # type: (Link) -> Vector
    """Determine maximum corner of bounding box of  link.

    >>> link = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...         [(0, 0, 0), (1, 1, 1), (0, 0, 2), (-1, -1, 1)]]
    >>> max_link(link)
    (1, 1, 2)
    """
    return max_path(max_path(path) for path in link)


def shift_path(path, vec):
    # type: (Path, Vector) -> Path
    """Shift all points in the path by subtracting the vector.
    Thus, the point at vec moves to the origin.
    
    >>> shift_path([(1, 0, 0), (0, 1, 0), (0, 0, 1)], (1, 3, 5))
    [(0, -3, -5), (-1, -2, -5), (-1, -3, -4)]
    """
    dx, dy, dz = vec
    return [(x - dx, y - dy, z - dz) for x, y, z in path]


def shift_link(link, vec):
    # type: (Link, Vector) -> Link
    """Shift all points in the link by subtracting the vector.
    Thus, the point at vec moves to the origin.

    >>> link = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...         [(0, 0, 0), (1, 1, 1), (0, 0, 2), (-1, -1, 1)]]
    >>> shift_link(link, (1, 3, 5))
    [[(0, -3, -5), (-1, -2, -5), (-1, -3, -4)], [(-1, -3, -5), (0, -2, -4), (-1, -3, -3), (-2, -4, -4)]]
    """
    return [shift_path(path, vec) for path in link]


def snap_path(path, lattice):
    # type: (Path, int) -> Path
    """Snap path to lattice.
    
    >>> snap_path([(2, 0, 0), (1, 1, 0), (1, 0, 1)], 2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1)]
    >>> snap_path([(1, 0, 0), (0, 1, 0), (0, 0, 1)], 2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1)]
    """
    if not is_in_lattice(path[0], lattice):
        v = snap_to_lattice(path[0], lattice)
        path = shift_path(path, subtract_vectors(path[0], v))
    return path


def snap_link(link, lattice):
    # type: (Link, int) -> Path
    """Snap link to lattice.
    
    >>> snap_link([[(2, 0, 0), (1, 1, 0), (1, 0, 1)]], 2)
    [[(2, 0, 0), (1, 1, 0), (1, 0, 1)]]
    >>> snap_link([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 2)
    [[(2, 0, 0), (1, 1, 0), (1, 0, 1)]]
    """
    if not is_in_lattice(link[0][0], lattice):
        v = snap_to_lattice(link[0][0], lattice)
        link = shift_link(link, subtract_vectors(link[0][0], v))
    return link


def align_path(path, lattice=1):
    # type: (Path) -> Path
    """Translate path such that minimum coordinates are 0,
    but do snap it to the lattice, avoiding negative coordinates.
    
    >>> align_path([(0, -3, -5), (-1, -2, -5), (-1, -3, -4)])
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    >>> align_path([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
    [(2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 0, 0)]
    """
    return shift_path(path, snap_to_lattice(min_path(path), lattice))


def align_link(link, lattice=1):
    # type: (Link) -> Link
    """Translate link such that minimum coordinates are 0,
    but do snap it to the lattice, avoiding negative coordinates.

    >>> link = [[(0, -3, -5), (-1, -2, -5), (-1, -3, -4)],
    ...         [(-1, -3, -5), (0, -2, -4), (-1, -3, -3), (-2, -4, -4)]]
    >>> align_link(link)
    [[(2, 1, 0), (1, 2, 0), (1, 1, 1)], [(1, 1, 0), (2, 2, 1), (1, 1, 2), (0, 0, 1)]]
    """
    return shift_link(link, snap_to_lattice(min_link(link), lattice))


def sum_path(path):
    # type: (Path) -> Vector
    """Sum all points.
    
    >>> sum_path([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    (1, 1, 1)
    """
    return reduce(add_vectors, path)


def path_centroid(path):
    # type: (Path) -> Vector
    """Compute the centroid of path in integer lattice.
    
    >>> path_centroid([(1, 2, 4), (0, 3, 4), (0, 2, 5)])
    (0, 2, 4)
    >>> path_centroid([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
    (0, 0, 0)
    """
    sx, sy, sz =  sum_path(path)
    n = len(path)
    return (sx // n, sy // n, sz // n)


def link_centroid(link):
    # type: (Link) -> Vector
    """Computer the centroid of link in integer lattice.

    >>> link = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...         [(0, 0, 0), (1, 1, 1), (0, 0, 2), (-1, -1, 1)]]
    >>> link_centroid(link)
    (0, 0, 0)
    >>> link = [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    ...         [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1), (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)],
    ...         [(3, 0, 0), (4, -1, 0), (5, 0, 0), (4, 1, 0)]]
    >>> link_centroid(link)
    (2, 0, 0)
    """
    sx, sy, sz = sum_path(sum_path(path) for path in link)
    # N.B. relies on the fact that sum_path() accepts an iterable
    n = sum(len(path) for path in link)
    return (sx // n, sy // n, sz // n)
    

def center_path(path, lattice=1):
    # type: (Path) -> Path
    """Translate path such that its centroid is near the origin.
    But do snap it to the lattice.
    
    Assumption: path is in lattice

    >>> center_path([(0, -3, -5), (-1, -2, -5), (-1, -3, -4)], lattice=2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1)]
    """
    return shift_path(path, snap_to_lattice(path_centroid(path), lattice))


def center_link(link, lattice=1):
    # type: (Link) -> Link
    """Translate path such that its centroid is near the origin.
    But do snap it to the lattice.

    >>> link = [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    ...         [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1), (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)],
    ...         [(3, 0, 0), (4, -1, 0), (5, 0, 0), (4, 1, 0)]]
    >>> center_link(link, lattice=2)
    [[(-1, 0, 0), (-2, 1, 0), (-3, 0, 0), (-2, -1, 0)], [(-2, 0, 0), (-1, 0, 1), (0, 0, 2), (1, 0, 1), (2, 0, 0), (1, 0, -1), (0, 0, -2), (-1, 0, -1)], [(1, 0, 0), (2, -1, 0), (3, 0, 0), (2, 1, 0)]]
    """
    lc = snap_to_lattice(link_centroid(link), lattice)
    return [shift_path(path, lc) for path in link]


def normalize_path(path):
    # type: (Path) -> Path
    """Normalize path.
    
    That is, rotate order of its points such that it starts with the minimum,
    and the next point is less than the last point.
    
    >>> normalize_path([(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)])
    [(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)]
    >>> normalize_path([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
    [(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)]
    >>> normalize_path([(1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)])
    [(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)]
    """
#     m, i = min((point, index) for index, point in enumerate(path))
    i = min(range(len(path)), key=lambda index: path[index])
    if path[(i + 1) % len(path)] < path[i - 1]:
        result = path[i:] + path[:i]
    else:
        result = path[i::-1] + path[:i:-1]
    return result


def path_variations(path, lattice=None):
    # type: (Path) -> tuple[list[Path], list[int]]
    """Determine list of all orientations of path modulo symmetry,
    and all symmetries.
    
    >>> path_variations([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])  # SC, square
    ([[(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)], [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]], [0, 1, 6, 7, 12, 13, 18, 19, 24, 25, 30, 31, 36, 37, 42, 43])
    >>> path_variations([(1, 0, 0), (0, 1, 0), (0, 0, 1)])  # FCC, equilateral triangle
    ([[(0, 0, 0), (0, 1, 1), (1, 0, 1)], [(0, 0, 0), (0, 1, 1), (1, 1, 0)], [(0, 0, 0), (1, 0, 1), (1, 1, 0)], [(0, 1, 1), (1, 0, 1), (1, 1, 0)], [(1, 0, 1), (1, 1, 0), (2, 0, 0)], [(1, 0, 1), (1, 1, 0), (2, 1, 1)], [(1, 0, 1), (2, 0, 0), (2, 1, 1)], [(1, 1, 0), (2, 0, 0), (2, 1, 1)]], [0, 2, 4, 7, 23, 27])
    >>> path_variations([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])  # FCC, square
    ([[(1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 0, 1)], [(0, 0, 0), (1, 0, -1), (2, 0, 0), (1, 0, 1)], [(0, 0, 0), (1, -1, 0), (2, 0, 0), (1, 1, 0)]], [0, 1, 6, 7, 12, 13, 18, 19, 24, 25, 30, 31, 36, 37, 42, 43])
    >>> path_variations([(0, 0, 0), (1, 1, 1), (0, 0, 2), (-1, -1, 1)])  # BCC, rhombus
    ([[(-1, -1, 1), (0, 0, 0), (1, 1, 1), (0, 0, 2)], [(-1, 1, -1), (0, 0, 0), (1, 1, 1), (0, 2, 0)], [(-1, 1, 1), (0, 0, 0), (1, -1, 1), (0, 0, 2)], [(-1, 1, 1), (0, 0, 0), (1, 1, -1), (0, 2, 0)], [(0, 0, 0), (1, -1, -1), (2, 0, 0), (1, 1, 1)], [(0, 0, 0), (1, -1, 1), (2, 0, 0), (1, 1, -1)]], [0, 7, 12, 19, 24, 31, 36, 43])
    """
    if not lattice:
        lattice = detect_lattice(path)
    path = snap_path(path, lattice)
    result = [normalize_path(align_path([f(*point) for point in path], lattice))
              for f in CUBE_SYMMETRY
             ]
    p0 = result[0]
    symmetries = [i for i, p in enumerate(result) if p == p0]
    result.sort()
    return ([center_path(path, lattice) for path, _ in groupby(result)],
            symmetries
           )


def cross_product(a, b):
    # type: (Vector, Vector) -> Vector
    """Cross product of two 3D integer vectors.
    
    >>> cross_product((1, 0, 0), (0, 1, 0))
    (0, 0, 1)
    >>> cross_product((1, 1, 0), (0, 1, 1))
    (1, -1, 1)
    """
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def dot_product(a, b):
    # type: (Vector, Vector) -> int
    """Dot product of two 3D integer vectors.
    
    >>> dot_product((1, 2, 3), (4, 5, 6))
    32
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def is_planar(path):
    # type: (Path) -> tuple[Vector, int] | None
    """Check if a closed path is planar.
    Returns (normal, offset) where dot(normal, point) == offset for all points,
    or None if the path is not planar or degenerate.
    
    >>> is_planar([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
    ((0, 0, 1), 0)
    >>> is_planar([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    ((0, 0, 1), 0)
    >>> is_planar([(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)])
    """
    if len(path) < 3:
        return None
    # Find a non-degenerate normal from edges originating at path[0]
    p0 = path[0]
    e0 = (path[1][0] - p0[0], path[1][1] - p0[1], path[1][2] - p0[2])
    normal = (0, 0, 0)
    for k in range(2, len(path)):
        ek = (path[k][0] - p0[0], path[k][1] - p0[1], path[k][2] - p0[2])
        normal = cross_product(e0, ek)
        if normal != (0, 0, 0):
            break
    if normal == (0, 0, 0):
        return None  # all points are collinear
    offset = dot_product(normal, p0)
    # Check all points lie in this plane
    for pt in path:
        if dot_product(normal, pt) != offset:
            return None
    return (normal, offset)


def _project_to_2d(point, normal):
    # type: (Vector, Vector) -> tuple[float, float]
    """Project a 3D point onto 2D by dropping the axis corresponding to
    the largest component of the normal vector.
    """
    nx, ny, nz = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if nx >= ny and nx >= nz:
        return (point[1], point[2])  # drop x
    elif ny >= nz:
        return (point[0], point[2])  # drop y
    else:
        return (point[0], point[1])  # drop z


def point_in_polygon_2d(px, py, polygon_2d):
    # type: (float, float, list[tuple[float, float]]) -> bool
    """Ray casting algorithm for point-in-polygon test in 2D.
    
    >>> poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    >>> point_in_polygon_2d(1.0, 1.0, poly)
    True
    >>> point_in_polygon_2d(3.0, 1.0, poly)
    False
    """
    n = len(polygon_2d)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon_2d[i][1], polygon_2d[i][0]
        yj, xj = polygon_2d[j][1], polygon_2d[j][0]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_line_segments(polygon, normal, offset, line_dir):
    # type: (Path, Vector, int, Vector) -> list[tuple[float, float]]
    """Compute the segments where a polygon's filled interior intersects a plane.
    
    The cutting plane is defined by (normal, offset): dot(normal, P) = offset.
    line_dir is the direction of the intersection line (cross product of normals).
    
    Returns a sorted list of (t_enter, t_exit) intervals along line_dir
    representing where the polygon interior crosses the plane.
    For non-convex polygons this may be multiple disjoint segments.
    """
    n = len(polygon)
    dists = [dot_product(normal, polygon[i]) - offset for i in range(n)]
    
    # Collect crossing points where polygon edges cross the cutting plane.
    # For vertices exactly on the plane (dist==0), only count as a crossing
    # if the polygon passes through the plane (non-zero neighbors on opposite sides).
    crossings = []
    for i in range(n):
        j = (i + 1) % n
        d1, d2 = dists[i], dists[j]
        if d1 == 0:
            # Vertex on the plane. Find the last non-zero dist before and after.
            prev_d = 0
            for k in range(1, n):
                prev_d = dists[(i - k) % n]
                if prev_d != 0:
                    break
            next_d = 0
            for k in range(1, n):
                next_d = dists[(i + k) % n]
                if next_d != 0:
                    break
            # Count as crossing only if prev and next are on strictly opposite sides
            if prev_d != 0 and next_d != 0 and (prev_d > 0) != (next_d > 0):
                crossings.append(dot_product(line_dir, polygon[i]))
        elif d2 != 0 and (d1 > 0) != (d2 > 0):
            # Edge properly crosses the plane
            t = d1 / (d1 - d2)
            pt = (polygon[i][0] + t * (polygon[j][0] - polygon[i][0]),
                  polygon[i][1] + t * (polygon[j][1] - polygon[i][1]),
                  polygon[i][2] + t * (polygon[j][2] - polygon[i][2]))
            crossings.append(dot_product(line_dir, pt))
        # d2 == 0 is handled when we process vertex j
    
    # Sort crossings and pair them into interior segments
    crossings.sort()
    segments = []
    for k in range(0, len(crossings) - 1, 2):
        segments.append((crossings[k], crossings[k + 1]))
    return segments


def curve_intersects_surface(curve, surface, normal, offset):
    # type: (Path, Path, Vector, int) -> bool
    """Check if any edge of curve crosses through the filled surface polygon.
    
    The surface is a planar polygon with the given normal and offset.
    An edge crosses the surface if it crosses the plane (endpoints on opposite sides)
    and the crossing point is inside the polygon boundary.
    
    >>> surf = [(-2, 0, 2), (-1, 0, 1), (0, 0, 0), (1, 0, -1), (2, 0, 0), (3, 0, 1), (4, 0, 2), (3, 0, 3), (2, 0, 4), (1, 0, 5), (0, 0, 4), (-1, 0, 3)]
    >>> curve = [(1, -3, 0), (1, -2, -1), (1, -1, -2), (1, 0, -3), (1, 1, -2), (1, 2, -1), (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3), (1, -1, 2), (1, -2, 1)]
    >>> n, o = is_planar(surf)
    >>> curve_intersects_surface(curve, surf, n, o)
    True
    """
    poly_2d = [_project_to_2d(pt, normal) for pt in surface]
    nc = len(curve)
    dists = [dot_product(normal, curve[i]) - offset for i in range(nc)]
    for i in range(nc):
        d1 = dists[i]
        if d1 == 0:
            # Vertex is on the plane. If it's inside the polygon, count it.
            px, py = _project_to_2d(curve[i], normal)
            if point_in_polygon_2d(px, py, poly_2d):
                return True
        else:
            d2 = dists[(i + 1) % nc]
            if d2 != 0 and (d1 > 0) != (d2 > 0):
                # Edge properly crosses the plane
                p1 = curve[i]
                p2 = curve[(i + 1) % nc]
                t = d1 / (d1 - d2)
                ix = p1[0] + t * (p2[0] - p1[0])
                iy = p1[1] + t * (p2[1] - p1[1])
                iz = p1[2] + t * (p2[2] - p1[2])
                px, py = _project_to_2d((ix, iy, iz), normal)
                if point_in_polygon_2d(px, py, poly_2d):
                    return True
    return False


def is_linked(link):
    # type: (Link) -> bool
    """Check if every path in a link intersects at least one other path's surface.
    A link is kept only if NO path has zero intersections with all other surfaces.
    Only applies when ALL paths are planar; mixed links pass automatically.
    """
    n = len(link)
    # Precompute planes for all paths
    planes = [is_planar(p) for p in link]
    # If any path is non-planar, skip the filter entirely
    if any(p is None for p in planes):
        return True
    for i in range(n):
        # Does path i intersect any other path's surface?
        has_intersection = False
        for j in range(n):
            if i == j:
                continue
            normal, offset = planes[j]
            if curve_intersects_surface(link[i], link[j], normal, offset):
                has_intersection = True
                break
        if not has_intersection:
            return False  # path i has zero intersections -> filter out
    return True


def filter_unlinked(links):
    # type: (list[Link]) -> list[Link]
    """Remove links where any path has zero intersections with other paths' surfaces.
    """
    result = [link for link in links if is_linked(link)]
    print("Surface intersection filter: %d -> %d links" % (len(links), len(result)))
    return result


def normalize_paths(link):
    # type: (Link) -> Link
    """Normalize each path in link, and then sort them.

    >>> link = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...         [(0, 0, 0), (1, 1, 1), (1, 1, 0), (1, 0, 0)]]
    >>> normalize_paths(link)
    [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)], [(0, 0, 1), (0, 1, 0), (1, 0, 0)]]
    >>> link = [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    ...         [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1), (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)],
    ...         [(3, 0, 0), (4, -1, 0), (5, 0, 0), (4, 1, 0)]]
    >>> normalize_paths(link)
    [[(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -1), (4, 0, 0), (3, 0, 1), (2, 0, 2), (1, 0, 1)], [(3, 0, 0), (4, -1, 0), (5, 0, 0), (4, 1, 0)]]
    """
    return sorted(normalize_path(path) for path in link)


def normalize_link(link, lattice=1):
    # type: (Link) -> Link
    """Normalize link, that is, produce a unique representative,
    that abstracts from it orientation and translation, and its internal structure.
    
    Assumption: all paths have same shape
    
    >>> link = [[(0, 0, 0), (1, 0, 1), (2, 0, 2), (1, 1, 2), (0, 2, 2), (0, 1, 1)],
    ...         [(3, -1, 1), (2, 0, 1), (1, 1, 1), (1, 0, 2), (1, -1, 3), (2, -1, 2)]]
    >>> normalize_link(link, 2)
    
    """
    best = None
    for f in CUBE_SYMMETRY:
        transformed = align_link(transform_link(f, link), lattice)
        # Normalize each path and sort, with early bail-out
        normalized = sorted(normalize_path(path) for path in transformed)
        if best is not None:
            # Lexicographic early termination: compare path by path
            skip = False
            for np, bp in zip(normalized, best):
                if np < bp:
                    break  # candidate is better, keep it
                elif np > bp:
                    skip = True  # candidate is worse, discard
                    break
            if skip:
                continue
        best = normalized
    return best


def link_variations(link, lattice=None):
    # type: (Link) -> tuple[list[Link], list[int]]
    """Determine list of all orientations of link modulo symmetry.

    Assumption: all paths have same shape and are in the same (sub)lattice

    >>> link = [[(0, 0, 0), (1, 0, 1), (2, 0, 2), (1, 1, 2), (0, 2, 2), (0, 1, 1)],
    ...         [(3, -1, 1), (2, 0, 1), (1, 1, 1), (1, 0, 2), (1, -1, 3), (2, -1, 2)]]
    >>> link_variations(link)

    >>> link = [[(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1),
    ...          (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)],
    ...         [(-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0),
    ...          (2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)]]
    >>> link_variations(link)
    ([[[(-1, -1, 0), (0, -2, 0), (1, -3, 0), (2, -2, 0), (3, -1, 0), (2, 0, 0), (1, 1, 0), (0, 0, 0)], [(1, -1, 0), (1, 0, -1), (1, 1, -2), (1, 2, -1), (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 1)]], [[(-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)], [(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -1), (4, 0, 0), (3, 0, 1), (2, 0, 2), (1, 0, 1)]], [[(-1, 0, -1), (0, 0, -2), (1, 0, -3), (2, 0, -2), (3, 0, -1), (2, 0, 0), (1, 0, 1), (0, 0, 0)], [(1, -2, 1), (1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3), (1, -1, 2)]], [[(-2, 0, 0), (-1, 0, -1), (0, 0, -2), (1, 0, -1), (2, 0, 0), (1, 0, 1), (0, 0, 2), (-1, 0, 1)], [(0, 0, 0), (1, -1, 0), (2, -2, 0), (3, -1, 0), (4, 0, 0), (3, 1, 0), (2, 2, 0), (1, 1, 0)]], [[(-1, 0, 1), (0, 0, 0), (1, 0, -1), (2, 0, 0), (3, 0, 1), (2, 0, 2), (1, 0, 3), (0, 0, 2)], [(1, -2, -1), (1, -1, -2), (1, 0, -3), (1, 1, -2), (1, 2, -1), (1, 1, 0), (1, 0, 1), (1, -1, 0)]], [[(-1, 1, 0), (0, 0, 0), (1, -1, 0), (2, 0, 0), (3, 1, 0), (2, 2, 0), (1, 3, 0), (0, 2, 0)], [(1, -3, 0), (1, -2, -1), (1, -1, -2), (1, 0, -1), (1, 1, 0), (1, 0, 1), (1, -1, 2), (1, -2, 1)]]], [0, 11, 13, 22, 29, 31, 40, 42])
    """
    if not lattice:
        lattice = detect_lattice(link[0])
    link = snap_link(link, lattice)
    result = [normalize_paths(align_link(transform_link(f, link), lattice))
              for f in CUBE_SYMMETRY
             ]
    l0 = result[0]
#     from pprint import pprint
#     pprint(list(enumerate(result)))
    symmetries = [i for i, l in enumerate(result) if l == l0]
    result.sort()
#     pprint(result)
    return ([center_link(link, lattice) for link, _ in groupby(result)],
            symmetries
           )


def link_symmetry_count(link, lattice=None):
    # type: (Link, int) -> int
    """Count how many of the 48 cube symmetries map a link to itself.
    
    >>> link = [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]]
    >>> link_symmetry_count(link, 2)
    16
    """
    if not lattice:
        lattice = detect_lattice(link[0])
    link = snap_link(link, lattice)
    canonical = normalize_paths(align_link(link, lattice))
    count = 0
    for f in CUBE_SYMMETRY:
        transformed = normalize_paths(align_link(transform_link(f, link), lattice))
        if transformed == canonical:
            count += 1
    return count


def normalize_links(links, key=None, lattice=1):
    # type: (list[T]) -> list[tuple[T, int, Link]]
    """Normalize all links, retaining the originals with their index.
    If key is not None, then key returns the link from the list items.
    
    The index is included in the result, to ensure that sorting/grouping the tuples
    will preserve the original order.

    Assumptions:
    * all paths have same shape
    * if key is None then T == Link else key has type Callable[[T], Link]
    
    >>> link = [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    ...         [(0, 0, 0), (1, 0, 1), (2, 0, 0), (1, 0, -1)]]
    >>> link_ = transform_link(CUBE_SYMMETRY[6], link)
    >>> normalize_links([link, link_])
    [([[(0, 1, 1), (1, 0, 1), (2, 1, 1), (1, 2, 1)], [(1, 1, 1), (1, 2, 0), (1, 3, 1), (1, 2, 2)]], 0, [[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], [(0, 0, 0), (1, 0, 1), (2, 0, 0), (1, 0, -1)]]), ([[(0, 1, 1), (1, 0, 1), (2, 1, 1), (1, 2, 1)], [(1, 1, 1), (1, 2, 0), (1, 3, 1), (1, 2, 2)]], 1, [[(0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0)], [(0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 1, -1)]])]
    >>> normalize_links([(link, '0'), (link_, '1')], key=lambda t: t[0])
    [([[(0, 1, 1), (1, 0, 1), (2, 1, 1), (1, 2, 1)], [(1, 1, 1), (1, 2, 0), (1, 3, 1), (1, 2, 2)]], 0, ([[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], [(0, 0, 0), (1, 0, 1), (2, 0, 0), (1, 0, -1)]], '0')), ([[(0, 1, 1), (1, 0, 1), (2, 1, 1), (1, 2, 1)], [(1, 1, 1), (1, 2, 0), (1, 3, 1), (1, 2, 2)]], 1, ([[(0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0)], [(0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 1, -1)]], '1'))]
    """
    if key is not None:
        return [(normalize_link(key(t), lattice), index, t)
                for index, t in enumerate(links)]
    # Try C batch acceleration
    if _normalize_c is not None and links:
        try:
            num_links = len(links)
            num_paths = len(links[0])
            path_len = len(links[0][0])
            link_ints = num_paths * path_len * 3
            total_ints = num_links * link_ints
            arr = (ctypes.c_int * total_ints)()
            idx = 0
            for link in links:
                for path in link:
                    for x, y, z in path:
                        arr[idx] = x; arr[idx+1] = y; arr[idx+2] = z
                        idx += 3
            out = (ctypes.c_int * total_ints)()
            rc = _normalize_c.normalize_links_batch_c(
                arr, num_links, num_paths, path_len, lattice, out)
            if rc == 0:
                result = []
                idx = 0
                for i in range(num_links):
                    norm_link = []
                    for p in range(num_paths):
                        norm_path = []
                        for pt in range(path_len):
                            norm_path.append((out[idx], out[idx+1], out[idx+2]))
                            idx += 3
                        norm_link.append(norm_path)
                    result.append((norm_link, i, links[i]))
                return result
        except Exception:
            pass  # fall through to pure Python
    return [(normalize_link(link, lattice), index, link)
            for index, link in enumerate(links)]


def remove_duplicates(links, key=None, lattice=None):
    # type: (list[T], Optional[Callable[[T]], Link]) -> list[T]
    """Remove duplicate links modulo symmetry.
    
    Assumptions:
    * all paths have same shape
    * if key is None then T == Link
      else key has type Callable[[T], Link] and lattice is not None
    
    >>> link = [[(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1),
    ...          (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)],
    ...         [(-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0),
    ...          (2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)]]
    >>> links = [transform_link(f, link) for f in CUBE_SYMMETRY]
    >>> remove_duplicates(links)
    [[[(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1), (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)], [(-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)]]]
    >>> remove_duplicates([(link, index) for index, link in enumerate(links)],
    ...     key=lambda t: t[0], lattice=2)
    [([[(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1), (4, 0, 0), (3, 0, -1), (2, 0, -2), (1, 0, -1)], [(-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)]], 0)]
    """
    if not links:
        return links
    if not lattice:
        lattice = detect_lattice(links[0][0] if key is None else key(links[0])[0])
        links = [snap_link(link, lattice) for link in links]
    normalized_links = normalize_links(links, key=key, lattice=lattice)
    normalized_links.sort(key=lambda t: t[0])
    result = [next(group)[1:]  # take original from first item in each group
              for _, group in groupby(normalized_links, key=lambda t: t[0])
             ]
    result.sort()
    return [link for _, link in result]  # drop the index
    
    # cheapest solution, which neither preserves orientation nor order
    # when there is no satellite data
#     normalized_links = [normalize_link(link) for link in links]
#     normalized_links.sort()
#     return [link for link, _ in groupby(normalized_links)]


def pointset_path(path):
    # type: (Path) -> set[Vector]
    """Determine set of points for path, including midpoints of segments.
    
    Assumption:
        * path segments are edges in the SC|FCC|BCC lattice

    Coordinates are doubled, so that midpoints have int coordinates.
    
    >>> pointset_path([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]) ^ {(2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0), (-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0)}
    set()
    """
    result = {(2 * x, 2 * y, 2 * z) for x, y, z in path}
    path_rot = path[1:]
    path_rot.append(path[0])
    result.update((x1 + x2, y1 + y2, z1 + z2)
                  for (x1, y1, z1), (x2, y2, z2) in zip(path, path_rot)
                 )
    return result


class BitGrid:
    """Maps 3D doubled-coordinates to bit positions in a Python int.
    
    All paths/prospects must share the same BitGrid instance so that
    bit positions are consistent for intersection checks.
    """
    
    def __init__(self, origin, size_y, size_z):
        # type: (Vector, int, int) -> None
        """origin: minimum (x, y, z) in doubled-coordinate space.
        size_y, size_z: dimensions of the grid in y and z.
        """
        self.ox, self.oy, self.oz = origin
        self.sy = size_y
        self.sz = size_z
    
    def point_to_bit(self, x, y, z):
        # type: (int, int, int) -> int
        """Convert a doubled-coordinate point to a bitmask with one bit set."""
        return 1 << (((x - self.ox) * self.sy + (y - self.oy)) * self.sz + (z - self.oz))
    
    def pointset_to_bits(self, pointset):
        # type: (set[Vector]) -> int
        """Convert a set of doubled-coordinate points to a bitmask."""
        bits = 0
        for x, y, z in pointset:
            bits |= 1 << (((x - self.ox) * self.sy + (y - self.oy)) * self.sz + (z - self.oz))
        return bits


def make_bitgrid(paths):
    # type: (list[Path]) -> BitGrid
    """Create a BitGrid that covers all given paths (in original coordinates).
    
    Computes the bounding box over all doubled-coordinates
    (vertices and edge midpoints) with some margin.
    """
    all_coords = []
    for path in paths:
        for x, y, z in path:
            all_coords.append((2 * x, 2 * y, 2 * z))
        path_rot = path[1:] + [path[0]]
        for (x1, y1, z1), (x2, y2, z2) in zip(path, path_rot):
            all_coords.append((x1 + x2, y1 + y2, z1 + z2))
    
    min_x = min(c[0] for c in all_coords) - 2
    min_y = min(c[1] for c in all_coords) - 2
    min_z = min(c[2] for c in all_coords) - 2
    max_x = max(c[0] for c in all_coords) + 2
    max_y = max(c[1] for c in all_coords) + 2
    max_z = max(c[2] for c in all_coords) + 2
    
    size_y = max_y - min_y + 1
    size_z = max_z - min_z + 1
    
    return BitGrid((min_x, min_y, min_z), size_y, size_z)


def generate_shifts(min_vec, max_vec, offset=0, min_move=0, lattice=1, step_size=1):
    # type: (Vector, Vector, int, int, int, int) -> list[Vector]
    """Generate list of all vectors within the box shell, sorted on length,
    restricted by lattice.
    
    min_move is _square_ of minimum move distance
    lattice: 1 -> SC, 2 -> FCC, 3 -> BCC
    step_size: only keep vectors whose coordinates are all multiples of step_size
    
    >>> generate_shifts((0, 3, 6), (1, 4, 7))
    [(0, 3, 6), (1, 3, 6), (0, 4, 6), (1, 4, 6), (0, 3, 7), (1, 3, 7), (0, 4, 7), (1, 4, 7)]
    >>> generate_shifts((6, 3, 0), (7, 4, 1))
    [(6, 3, 0), (6, 3, 1), (6, 4, 0), (6, 4, 1), (7, 3, 0), (7, 3, 1), (7, 4, 0), (7, 4, 1)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 0)
    [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0), (-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1), (1, -1, 0), (1, 0, -1), (1, 0, 1), (1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 1)
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0), (-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1), (1, -1, 0), (1, 0, -1), (1, 0, 1), (1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 2)
    [(-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1), (1, -1, 0), (1, 0, -1), (1, 0, 1), (1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 3)
    [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 0, 2)
    [(0, 0, 0), (-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1), (1, -1, 0), (1, 0, -1), (1, 0, 1), (1, 1, 0)]
    >>> generate_shifts((0, 0, 0), (0, 0, 0), 1, 0, 3)
    [(0, 0, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    """
    min_x, min_y, min_z = min_vec
    max_x, max_y, max_z = max_vec
#     min_move2 = min_move * min_move
    vectors = [(x, y, z)
               for x in range(min_x - offset, max_x + offset + 1)
               for y in range(min_y - offset, max_y + offset + 1)
               for z in range(min_z - offset, max_z + offset + 1)
               if is_in_lattice((x, y, z), lattice)
               if square_length((x, y, z)) >= min_move
               if x % step_size == 0 and y % step_size == 0 and z % step_size == 0
              ]
    result = sorted(vectors, key=square_length)
    print(len(result), "shift vectors (sorted):", result)
    print("=====")
    return result


def tangle(links, pvs, lattice=1):
    # type: (list[tuple[Link, list[int], int]], list[tuple[Path, int]]) -> list[tuple[Link, list[int], int]]
    """Tangle each path in pvs into each link without intersection,
    removing symmetric results.
    
    Assumptions:
    * links is list of links, path indices, and corresponding bitmasks, and
    * pvs is list of path variations and corresponding bitmasks
    * path segments are edges in the SC|FCC|BCC lattice
    
    >>> links = [(link, [-1], pointset_path(link[0]))
    ...          for link in [[[(0, -1, 0), (0, 0, -1), (0, 1, 0), (0, 0, 1)]]]]
    >>> pvs = [[(-1, 0, 0), (0, 0, -1), (1, 0, 0), (0, 0, 1)]]
    >>> tangle(links, [(p, pointset_path(p)) for p in pvs])
    []
    >>> pvs = [[(1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 0, 1)], [(0, 0, 0), (1, 0, -1), (2, 0, 0), (1, 0, 1)]]
    >>> tangle(links, [(p, pointset_path(p)) for p in pvs])
    [([[(0, -1, 0), (0, 0, -1), (0, 1, 0), (0, 0, 1)], [(1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 0, 1)]], [-1, 0], {(2, -2, 0), (0, -1, -1), (2, 1, 1), (2, 0, 2), (0, -1, 1), (0, 2, 0), (0, -2, 0), (2, 0, -2), (0, 0, 2), (2, -1, -1), (0, 0, -2), (2, -1, 1), (0, 1, -1), (2, 2, 0), (2, 1, -1), (0, 1, 1)}), ([[(0, -1, 0), (0, 0, -1), (0, 1, 0), (0, 0, 1)], [(0, 0, 0), (1, 0, -1), (2, 0, 0), (1, 0, 1)]], [-1, 1], {(0, -1, -1), (2, 0, 2), (1, 0, 1), (0, -1, 1), (1, 0, -1), (0, 2, 0), (0, 0, 0), (0, -2, 0), (2, 0, -2), (4, 0, 0), (0, 0, 2), (0, 0, -2), (3, 0, 1), (0, 1, -1), (3, 0, -1), (0, 1, 1)})]
    """
    if links and isinstance(links[0][2], int):
        # Bit-based collision detection (fast path)
        result = [(link + [path], pvs_indices + [pvs_index], link_bits | path_bits)
                  for link, pvs_indices, link_bits in links
                  for pvs_index, (path, path_bits)
                      in enumerate(pvs[pvs_indices[-1] + 1:], pvs_indices[-1] + 1)
#                       in enumerate(pvs)
                  if link_bits & path_bits == 0
                 ]
    else:
        # Set-based collision detection (backward compat / doctests)
        result = [(link + [path], pvs_indices + [pvs_index], link_ps | path_ps)
                  for link, pvs_indices, link_ps in links
                  for pvs_index, (path, path_ps)
                      in enumerate(pvs[pvs_indices[-1] + 1:], pvs_indices[-1] + 1)
#                       in enumerate(pvs)
                  if link_ps.isdisjoint(path_ps)
                 ]
#     return remove_duplicates(result, lambda t: t[0], lattice)
    return result


def tangle_gen(links, pvs, lattice=1):
    # type: (...) -> Generator[tuple[Link, list[int], int]]
    """Generator version of tangle(). Yields results one at a time
    to avoid building the full list in memory.
    Only supports bit-based collision detection (the fast path).
    """
    for link, pvs_indices, link_bits in links:
        for pvs_index, (path, path_bits) in enumerate(
                pvs[pvs_indices[-1] + 1:], pvs_indices[-1] + 1):
            if link_bits & path_bits == 0:
                yield (link + [path], pvs_indices + [pvs_index],
                       link_bits | path_bits)


def generate_links(paths, n, offset, min_move, prospects_cap, stage_cap=0, lattice=None,
                   filter_planar=False, step_size=1):
    # type: (list[Path] | Path, int, int, int) -> (list[Link], list[Path])
    """Generate all unique links with n copies of each path shape.
    Also return the selected path variations.
    
    Parameter `paths` is a single path or list of k path shapes.
    Parameter `n` is the number of copies per shape (total paths = k * n).
    Parameter `offset` (from bounding box) bounds the translations.
    Parameter `min_move` is minimum moving distance (squared)
    Parameter `prospects_cap` caps the number of prospects per shape (0 for no cap)
    Parameter `stage_cap` caps the number of links between stages (0 for no cap)
    Parameter `filter_planar` if True, remove links where no path pierces another's disk
    Parameter `step_size` lattice step unit size (1 or 2)
    
    Assumptions:
    * n >= 1
    * path segments are edges in the SC|FCC|BCC lattice
    
    It uses a breadth first approach.
    
    >>> generate_links([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], 0, 0, 0, 100)
    Traceback (most recent call last):
        ...
    AssertionError: n >= 1 violated (n == 0)
    >>> generate_links([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], 1, 0, 0, 1)
    ([[[(1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 0, 1)]]], [[(0, 0, 0), (0, 1, -1), (0, 2, 0), (0, 1, 1)]])
    >>> generate_links([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], 2, 0, 0, 2)
    ([[[(1, -1, 0), (1, 0, -1), (1, 1, 0), (1, 0, 1)], [(0, 0, 0), (0, 1, -1), (0, 2, 0), (0, 1, 1)]]], [[(0, 0, 0), (0, 1, -1), (0, 2, 0), (0, 1, 1)], [(0, -1, 1), (0, 0, 0), (0, 1, 1), (0, 0, 2)]])
    """
    assert n >= 1, "n >= 1 violated (n == %d)" % n
    
    # Backward compat: single path -> wrap in list
    if paths and isinstance(paths[0], tuple):
        paths = [paths]
    
    k = len(paths)  # number of distinct shapes
    
    if not lattice:
        lattice = detect_lattice(paths[0])
    
    # Generate path variations for each shape
    shape_pvs = []
    nodes_per_shape = []
    for path in paths:
        path = snap_path(path, lattice)
        pv, _ = path_variations(path, lattice)
        shape_pvs.append(pv)
        nodes_per_shape.append(len(path))
    
    # Use first shape's first variation as the base path
    base_path = shape_pvs[0][0]
    base_ps = pointset_path(base_path)
    
    # Generate prospect pools per shape: shifted variations that don't collide with base
    all_pvs_sets = []
    variations_per_shape = []
    shifts_per_shape = []
    for si, pv in enumerate(shape_pvs):
        shifts = generate_shifts(min_link(pv), max_link(pv), offset, min_move,
                                 lattice, step_size)
        shifts_per_shape.append(len(shifts))
        variations_per_shape.append(len(pv))
        pvs_sets = [(shifted_path, ps)
                    for vec in shifts
                    for p in pv
                    for shifted_path in [shift_path(p, vec)]
                    for ps in [pointset_path(shifted_path)]
                    if ps.isdisjoint(base_ps)
                   ]
        if prospects_cap:
            pvs_sets = pvs_sets[:prospects_cap]
        all_pvs_sets.append(pvs_sets)
    
    # Build one BitGrid covering base_path and all prospects from all shapes
    all_paths_for_grid = [base_path]
    for pvs_sets in all_pvs_sets:
        all_paths_for_grid.extend(sp for sp, _ in pvs_sets)
    grid = make_bitgrid(all_paths_for_grid)
    
    # Convert each shape's prospects to bitmasks
    base_bits = grid.pointset_to_bits(base_ps)
    shape_pools = []
    for pvs_sets in all_pvs_sets:
        pool = [(sp, grid.pointset_to_bits(ps)) for sp, ps in pvs_sets]
        shape_pools.append(pool)
    
    # Start with base path (first copy of first shape)
    result = [([base_path], [-1], base_bits)]
    path_count = 1
    funnel = []  # (path_count, link_count) at each stage
    
    # Phase loop: for each shape, add n copies (first shape starts at n-1 since base is already in)
    for si in range(k):
        copies_needed = n if si > 0 else n - 1
        pool = shape_pools[si]
        for ci in range(copies_needed):
            path_count += 1
            # Reset pvs_indices for new shape phases (so tangle starts from index 0)
            if ci == 0 and si > 0:
                result = [(link, [-1], bits) for link, _, bits in result]
            if stage_cap:
                result = list(islice(tangle_gen(result, pool, lattice), stage_cap))
            else:
                result = tangle(result, pool, lattice)
            funnel.append((path_count, len(result)))
            print("number of links with", path_count, "paths:", len(result))
    
    before_dedup = len(result)
    links = remove_duplicates([link for link, _, _ in result], lattice=lattice)
    after_dedup = len(links)
    if filter_planar:
        links = filter_unlinked(links)
    after_filter = len(links)
    
    # Compute per-link symmetry counts and sort by descending symmetry
    sym_counts = [link_symmetry_count(link, lattice) for link in links]
    paired = sorted(zip(sym_counts, links), key=lambda x: -x[0])
    sym_counts = [s for s, _ in paired]
    links = [l for _, l in paired]
    
    # Build stats string
    total_paths = k * n
    total_nodes = sum(nodes_per_shape) * n
    lines = []
    lines.append("=== Link Search Stats ===")
    lines.append("Shapes: %d, Copies per shape: %d, Total paths per link: %d" % (k, n, total_paths))
    for si in range(k):
        candidates = shifts_per_shape[si] * variations_per_shape[si]
        lines.append("  Shape %d: %d nodes, %d variations, %d shifts, %d candidates -> %d prospects"
                      % (si + 1, nodes_per_shape[si], variations_per_shape[si],
                         shifts_per_shape[si], candidates, len(shape_pools[si])))
    lines.append("Total nodes per link: %d" % total_nodes)
    lines.append("--- Search Funnel ---")
    lines.append("  Stage 1: 1 link (base path)")
    for pc, lc in funnel:
        lines.append("  Stage %d: %d links (before dedup)" % (pc, lc))
    lines.append("  After dedup: %d" % after_dedup)
    if filter_planar:
        lines.append("  After planar filter: %d" % after_filter)
    lines.append("========================")
    stats = "\n".join(lines)
    print(stats)
    
    all_prospects = []
    for pool in shape_pools:
        all_prospects.extend(p for p, _ in pool)
    return (links, all_prospects, stats, sym_counts)


if __name__ == "__main__":
    import doctest
    
    print('linklib.py', _version)
    print("doctest:", doctest.testmod())
