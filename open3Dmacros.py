
import open3d as o3d
import gmsh
f = '/home/tom/HEATruns/MEQscans/CAD/test.stl'
f2 = '/home/tom/HEATruns/MEQscans/CAD/test2.stl'
f3 = '/home/tom/HEATruns/MEQscans/CAD/test3.step'

gmsh.initialize()
gmsh.model.occ.importShapes(f3)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write(f2)
gmsh.finalize()

m2 = o3d.io.read_triangle_mesh(f2)
m2.remove_degenerate_triangles()
m2.remove_duplicated_triangles()
m2.remove_duplicated_vertices()
m2.remove_non_manifold_edges()
m2.compute_triangle_normals()
m2.simplify_vertex_clustering(0.1)
o3d.io.write_triangle_mesh(f2, m2)


pcd = m2.sample_points_poisson_disk(number_of_points=15000)
pm = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)[0]
dm = pm.simplify_quadric_decimation(1000)


def check_properties(mesh):
    mesh.compute_vertex_normals()
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")
    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((1, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls
