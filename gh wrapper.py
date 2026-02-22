#! python 3
# env: C:\Users\Owner\Documents\link finder\
"""Provides a scripting component Link Generator.
    Inputs:
        paths: path shapes to be linked (List Access with Type hint Polyline)
        n: number of copies per shape (Item Access with Type hint int)
        offset: steps beyond bounding box for path translations (Item Access with Type hint int)
        min_move: minimum distance to move paths (Item Access with Type hint int)
        prospects_cap: cap on number of prospects (Item Access with Type hint int)
        stage_cap: cap on number of links per stage (Item Access with Type hint int)
        filter_planar: filter out links with no disk piercing (Item Access with Type hint bool)
        step_size: lattice step unit size, 1 or 2 (Item Access with Type hint int)
        enable: if not, copy input to output (Item Access with Type hint bool)
        reload_linklib: whether to reload linklib (Item Access with Type hint bool)
    Output:
        links: The output variable (DataTree with Polyline)
        prospects: Secondary output variable (List Access with Polyline)
        stats: search statistics summary (Item Access with str)
        symmetry_counts: symmetry count per link (List Access with int)
"""

__author__ = "Tom Verhoeff"
__version__ = "2024.04.02"

import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
import linklibDEV as ll
print("Loaded module:", ll.__name__, "from:", ll.__file__)
if reload_linklib:
    import importlib
    importlib.reload(ll)
    print("linklibDEV reloaded:", ll._version)

if enable:
    p = [ll.import_path(path) for path in paths]
    print("Shapes:", len(p), "Points per shape:", [len(s) for s in p])

    ls, pvs, stats, symmetry_counts = ll.generate_links(p, n, offset, min_move, prospects_cap,
                                  stage_cap, lattice,
                                  filter_planar=filter_planar, step_size=step_size)
    print("Links:", len(ls))
    print("Prospects:", len(pvs))

    links = ll.export_links(ls)
    prospects = [ll.export_path(prospect) for prospect in pvs]

else:
    print('DISABLED')
    links = ll.export_links([])
    prospects = []
    stats = ""
    symmetry_counts = []
