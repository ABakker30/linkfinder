"""Provides a scripting component Link Generator.
    Inputs:
        path: to be linked (Item Access with Type hint Polyline)
        n: number of copies to link (Item Access with Type hint int)
        offset: steps beyond bounding box for path translations (Item Access with Type hint int)
        min_move: minimum distance to move paths (Item Access with Type hint int)
        prospects_cap: cap on number of prospects (Item Access with Type hint int)
        stage_cap: cap on number of links per stage (Item Access with Type hint int)
        enable: if not, copy input to output (Item Access with Type hint bool)
        reload_linklib: whether to reload linklib (Item Access with Type hint bool)
    Output:
        links: The output variable (DataTree with Polyline)
        prospects: Secondary output variable (List Access with Polyline)
"""

__author__ = "Tom Verhoeff"
__version__ = "2024.04.02"

import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
import sys
if r"C:\Users\Owner\Documents\link finder" not in sys.path:
    sys.path.insert(0, r"C:\Users\Owner\Documents\link finder")
import linklibDEV as ll
print("Loaded module:", ll.__name__, "from:", ll.__file__)
if reload_linklib:
    reload(ll)
    print("linklibDEV reloaded:", ll._version)

if enable:
    p = ll.import_path(path)
    print("Points in path:", len(p))

    ls, pvs = ll.generate_links(p, n, offset, min_move, prospects_cap, stage_cap, lattice)
    print("Links:", len(ls))
    print("Prospects:", len(pvs))

    links = ll.export_links(ls)
    prospects = [ll.export_path(prospect) for prospect in pvs]

else:
    print('DISABLED')
    links = ll.export_links([])
    prospects = [path]
