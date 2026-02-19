/*
 * normalize_link_c.c - Fast C implementation of normalize_link()
 *
 * Called via ctypes from linklibDEV.py.
 * Exports: normalize_link_c(link_data, num_paths, path_len, lattice, out_data)
 *
 * Link is represented as a flat int array:
 *   [x0,y0,z0, x1,y1,z1, ...] for path 0, then path 1, etc.
 * All paths must have the same length (path_len points).
 */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include <stdlib.h>
#include <string.h>

/* 48 cube symmetries as (perm[3], sign[3]) */
static const int PERM[48][3] = {
    {0,1,2},{0,1,2},{0,1,2},{0,1,2},{0,1,2},{0,1,2},{0,1,2},{0,1,2},
    {0,2,1},{0,2,1},{0,2,1},{0,2,1},{0,2,1},{0,2,1},{0,2,1},{0,2,1},
    {1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
    {1,2,0},{1,2,0},{1,2,0},{1,2,0},{1,2,0},{1,2,0},{1,2,0},{1,2,0},
    {2,0,1},{2,0,1},{2,0,1},{2,0,1},{2,0,1},{2,0,1},{2,0,1},{2,0,1},
    {2,1,0},{2,1,0},{2,1,0},{2,1,0},{2,1,0},{2,1,0},{2,1,0},{2,1,0},
};

static const int SIGN[48][3] = {
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
    { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},{-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1},
};

/* is_in_lattice: check if (x,y,z) is in the given lattice */
static int is_in_lattice(int x, int y, int z, int lattice) {
    if (lattice == 1) return 1;
    /* For lattice 2 (FCC) and 3 (BCC): (x%2 + y%2 + z%2) % lattice == 0 */
    int mx = ((x % 2) + 2) % 2;
    int my = ((y % 2) + 2) % 2;
    int mz = ((z % 2) + 2) % 2;
    return (mx + my + mz) % lattice == 0;
}

/* snap_to_lattice: find nearest lattice point to (x,y,z) */
static void snap_to_lattice(int x, int y, int z, int lattice, int *ox, int *oy, int *oz) {
    if (lattice == 1 || is_in_lattice(x, y, z, lattice)) {
        *ox = x; *oy = y; *oz = z; return;
    }
    if (lattice == 2 || is_in_lattice(x-1, y, z, lattice)) {
        *ox = x-1; *oy = y; *oz = z; return;
    }
    if (is_in_lattice(x, y-1, z, lattice)) {
        *ox = x; *oy = y-1; *oz = z; return;
    }
    *ox = x; *oy = y; *oz = z-1;
}

/* Compare two points lexicographically. Returns <0, 0, >0 */
static int cmp_point(const int *a, const int *b) {
    if (a[0] != b[0]) return a[0] - b[0];
    if (a[1] != b[1]) return a[1] - b[1];
    return a[2] - b[2];
}

/* Compare two paths (arrays of path_len*3 ints) lexicographically */
static int cmp_path(const int *a, const int *b, int path_len) {
    for (int i = 0; i < path_len * 3; i++) {
        if (a[i] != b[i]) return a[i] - b[i];
    }
    return 0;
}

/* qsort comparator for paths */
static int g_path_len = 0;
static int cmp_path_qsort(const void *a, const void *b) {
    return cmp_path((const int *)a, (const int *)b, g_path_len);
}

/*
 * normalize_path: rotate point order so it starts at the minimum point,
 * with direction tie-breaking (next < last => forward, else reverse).
 * Input: path of path_len points (each 3 ints), output written to out.
 */
static void normalize_path(const int *path, int path_len, int *out) {
    /* Find index of minimum point */
    int min_idx = 0;
    for (int i = 1; i < path_len; i++) {
        if (cmp_point(path + i*3, path + min_idx*3) < 0)
            min_idx = i;
    }
    /* Compare next vs prev to decide direction */
    int next_idx = (min_idx + 1) % path_len;
    int prev_idx = (min_idx - 1 + path_len) % path_len;
    if (cmp_point(path + next_idx*3, path + prev_idx*3) < 0) {
        /* Forward: path[min_idx], path[min_idx+1], ... */
        for (int i = 0; i < path_len; i++) {
            int src = (min_idx + i) % path_len;
            out[i*3]   = path[src*3];
            out[i*3+1] = path[src*3+1];
            out[i*3+2] = path[src*3+2];
        }
    } else {
        /* Reverse: path[min_idx], path[min_idx-1], ... */
        for (int i = 0; i < path_len; i++) {
            int src = (min_idx - i + path_len) % path_len;
            out[i*3]   = path[src*3];
            out[i*3+1] = path[src*3+1];
            out[i*3+2] = path[src*3+2];
        }
    }
}

/*
 * normalize_link_c: Main entry point.
 *
 * link_data: flat array of num_paths * path_len * 3 ints
 *            (path0_pt0_x, path0_pt0_y, path0_pt0_z, path0_pt1_x, ..., path1_pt0_x, ...)
 * num_paths: number of paths in the link
 * path_len:  number of points per path (all paths same length)
 * lattice:   1=SC, 2=FCC, 3=BCC
 * out_data:  output buffer, same size as link_data (num_paths * path_len * 3 ints)
 *
 * Returns 0 on success.
 */
EXPORT int normalize_link_c(const int *link_data, int num_paths, int path_len,
                             int lattice, int *out_data) {
    int total_points = num_paths * path_len;
    int total_ints = total_points * 3;
    int path_ints = path_len * 3;

    /* Workspace: transformed link, normalized paths, best result */
    int *transformed = (int *)malloc(total_ints * sizeof(int));
    int *normalized  = (int *)malloc(total_ints * sizeof(int));
    int *best        = (int *)malloc(total_ints * sizeof(int));
    int *norm_path_buf = (int *)malloc(path_ints * sizeof(int));
    int have_best = 0;

    if (!transformed || !normalized || !best || !norm_path_buf) {
        free(transformed); free(normalized); free(best); free(norm_path_buf);
        return -1;
    }

    for (int s = 0; s < 48; s++) {
        const int *perm = PERM[s];
        const int *sign = SIGN[s];

        /* Step 1: Apply symmetry transform to all points */
        for (int i = 0; i < total_points; i++) {
            const int *pt = link_data + i * 3;
            int *out = transformed + i * 3;
            out[0] = pt[perm[0]] * sign[0];
            out[1] = pt[perm[1]] * sign[1];
            out[2] = pt[perm[2]] * sign[2];
        }

        /* Step 2: align_link - find min of all points, snap, shift */
        int min_x = transformed[0], min_y = transformed[1], min_z = transformed[2];
        for (int i = 1; i < total_points; i++) {
            int x = transformed[i*3], y = transformed[i*3+1], z = transformed[i*3+2];
            if (x < min_x) min_x = x;
            if (y < min_y) min_y = y;
            if (z < min_z) min_z = z;
        }
        int snap_x, snap_y, snap_z;
        snap_to_lattice(min_x, min_y, min_z, lattice, &snap_x, &snap_y, &snap_z);
        /* shift = snap - min (we subtract snap from all points to align) */
        /* Actually: align_link shifts by snap_to_lattice(min_link), which means
           shift_link(link, snap_to_lattice(min_link(link)))
           shift_path subtracts the vector, so new_pt = pt - snap */
        for (int i = 0; i < total_points; i++) {
            transformed[i*3]   -= snap_x;
            transformed[i*3+1] -= snap_y;
            transformed[i*3+2] -= snap_z;
        }

        /* Step 3: normalize each path and copy into normalized buffer */
        for (int p = 0; p < num_paths; p++) {
            normalize_path(transformed + p * path_ints, path_len,
                           normalized + p * path_ints);
        }

        /* Step 4: sort the normalized paths */
        g_path_len = path_len;
        qsort(normalized, num_paths, path_ints * sizeof(int), cmp_path_qsort);

        /* Step 5: compare with best (early termination) */
        if (have_best) {
            int skip = 0;
            /* Compare path by path */
            for (int p = 0; p < num_paths; p++) {
                int c = cmp_path(normalized + p * path_ints,
                                 best + p * path_ints, path_len);
                if (c < 0) break;       /* candidate is better, keep it */
                if (c > 0) { skip = 1; break; }  /* candidate is worse */
            }
            if (skip) continue;
        }

        /* This candidate is the new best */
        memcpy(best, normalized, total_ints * sizeof(int));
        have_best = 1;
    }

    /* Copy best to output */
    memcpy(out_data, best, total_ints * sizeof(int));

    free(transformed);
    free(normalized);
    free(best);
    free(norm_path_buf);
    return 0;
}

/*
 * Batch version: normalize multiple links at once.
 * All links must have the same num_paths and path_len.
 *
 * all_links: flat array of num_links * num_paths * path_len * 3 ints
 * all_out:   output buffer, same size
 *
 * Returns 0 on success.
 */
EXPORT int normalize_links_batch_c(const int *all_links, int num_links,
                                    int num_paths, int path_len,
                                    int lattice, int *all_out) {
    int link_ints = num_paths * path_len * 3;
    for (int i = 0; i < num_links; i++) {
        int rc = normalize_link_c(all_links + i * link_ints,
                                   num_paths, path_len, lattice,
                                   all_out + i * link_ints);
        if (rc != 0) return rc;
    }
    return 0;
}
