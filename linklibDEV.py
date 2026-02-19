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
from itertools import groupby
from functools import reduce
try: 
    import Rhino.Geometry as rg
    import Grasshopper as gh
    import ghpythonlib.treehelpers as th
except:
    print("Working outside Rhino/Grasshopper.  Some functionality is not available")

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
    
    Warning: Not efficient
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
    
    Warning: This is not efficient.
    
    The images of vectors (1, 0, 0), (0, 1, 0), and (0, 0, 1)
    uniquely determine the symmetry.
    
    >>> symmetry_to_index(CUBE_SYMMETRY[0])
    0
    >>> symmetry_to_index(CUBE_SYMMETRY[-1])
    47
    """
    symm_matrix = symmetry_matrix(symm)
    
    for index, f in enumerate(CUBE_SYMMETRY):
        if symmetry_matrix(f) == symm_matrix:
            return index

    return None


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
    return min(normalize_paths(align_link(transform_link(f, link), lattice))
               for f in CUBE_SYMMETRY
              )


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
    return ([(normalize_link(link, lattice), index, link)
             for index, link in enumerate(links)
            ]
        if key is None
        else [(normalize_link(key(t), lattice), index, t)
              for index, t in enumerate(links)
             ]
    )


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


def generate_shifts(min_vec, max_vec, offset=0, min_move=0, lattice=1):
    # type: (Vector, Vector, int, int) -> list[Vector]
    """Generate list of all vectors within the box shell, sorted on length,
    restricted by lattice.
    
    min_move is _square_ of minimum move distance
    lattice: 1 -> SC, 2 -> FCC, 3 -> BCC
    
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
              ]
    result = sorted(vectors, key=square_length)
    print(len(result), "shift vectors (sorted):", result)
    print("=====")
    return result


def tangle(links, pvs, lattice=1):
    # type: (list[tuple[Link, list[int], set[Vector]]], list[tuple[Path, set[Vector]]]) -> list[tuple[Link, list[int], set[Vector]]]
    """Tangle each path in pvs into each link without intersection,
    removing symmetric results.
    
    Assumptions:
    * links is list of links, path indices, and corresponding pointsets, and
    * pvs is list of path variations and corresponding pointsets
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
    result = [(link + [path], pvs_indices + [pvs_index], link_ps | path_ps)
              for link, pvs_indices, link_ps in links
              for pvs_index, (path, path_ps)
                  in enumerate(pvs[pvs_indices[-1] + 1:], pvs_indices[-1] + 1)
#                   in enumerate(pvs)
              if link_ps.isdisjoint(path_ps)
             ]
#     return remove_duplicates(result, lambda t: t[0], lattice)
    return result


def generate_links(path, n, offset, min_move, prospects_cap, stage_cap=0, lattice=None):
    # type: (Path, int, int, int) -> (list[Link], list[Path])
    """Generate all unique links with n copies of path.
    Also return the selected path variations.
    
    Parameter `offset` (from bounding box) bounds the translations.
    Parameter `min_move` is minimum moving distance (squared)
    Parameter `prospects_cap` caps the number of prospects (0 for no cap)
    Parameter `stage_cap` caps the number of links between stages (0 for no cap)
    
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
    # Steps:
    # 1. Generate path variations
    # 2. Generate path translations
    # 3. Repeatedly add another path variation and remove duplicate links
    #    New path should not intersect/overlap others
    
    if not lattice:
        lattice = detect_lattice(path)
    path = snap_path(path, lattice)
    pv, _ = path_variations(path, lattice)  # ignore returned symmetries
    base_path = pv[0]
#     print("base path:", base_path)
    base_ps = pointset_path(base_path)
    pvs = [(shifted_path, ps)
           for vec in generate_shifts(min_link(pv), max_link(pv), offset, min_move,
                                      lattice)
           for p in pv
           for shifted_path in [shift_path(p, vec)]
           for ps in [pointset_path(shifted_path)]
           if ps.isdisjoint(base_ps)
          ]
    if prospects_cap:
        pvs = pvs[:prospects_cap]
#     print("len(pvs):", len(pvs))
#     from pprint import pprint
#     pprint(pvs)
    result = [([base_path], [-1], base_ps)]
    
    # TODO: add statistics output
    # TODO: add extra cap on results of each stage
    # TODO: check that duplicate removal after each stage does not miss links
    for k in range(2, n + 1):
        result = tangle(result, pvs, lattice)
        print("number of links before capping with", k, "paths:", len(result))
        if stage_cap:
            result = result[:stage_cap]  # no sorting yet
        print("number of links with", k, "paths:", len(result))
        
#    return ([link for link, _, _ in result], [p for p, _ in pvs])
    return (remove_duplicates([link for link, _, _ in result], lattice=lattice), [p for p, _ in pvs])


if __name__ == "__main__":
    import doctest
    
    print('linklib.py', _version)
    print("doctest:", doctest.testmod())
