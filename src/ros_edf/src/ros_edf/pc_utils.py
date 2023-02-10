from typing import Optional, Tuple

import open3d as o3d
import numpy as np

from edf.data import PointCloud

from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point



def pcd_from_numpy(coord: np.ndarray, color: Optional[np.ndarray], voxel_filter_size: Optional[float] = None):
    assert len(coord.shape) == 2, f"coord must be of shape (N_points, 3), but shape {coord.shape} is given."
    if color is None:
        raise NotImplementedError
        color = np.tile(np.array([[0.8, 0.5, 0.8]]), (coord.shape[-2],1)) # (N_points, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)

    if voxel_filter_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_filter_size)

    return pcd

def pcd_to_numpy(pcd: o3d.cuda.pybind.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors

def draw_geometry(geometries):
    if not hasattr(geometries, '__iter__'):
        geometries = [geometries]
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        if type(geometry) == PointCloud:
            geometry = geometry.to_pcd()
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.8])
    viewer.run()
    viewer.destroy_window()


def reconstruct_surface(pcd: o3d.cuda.pybind.geometry.PointCloud) -> o3d.cuda.pybind.geometry.TriangleMesh:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=2.0)
    
    alpha = 0.015
    mesh: o3d.cuda.pybind.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def mesh_o3d_to_ros(mesh: o3d.cuda.pybind.geometry.TriangleMesh) -> Mesh:
    mesh_ros = Mesh()
    mesh_ros.triangles = list(map(lambda x:MeshTriangle(vertex_indices=x), mesh.triangles))
    mesh_ros.vertices = list(map(lambda x:Point(x=x[0], y=x[1], z=x[2]), mesh.vertices))
    
    return mesh_ros