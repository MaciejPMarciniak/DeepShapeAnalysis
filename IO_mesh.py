from vtk import vtkPolyDataWriter, vtkPolyDataReader, vtkPolyData
from vtk.numpy_interface import dataset_adapter

import os
import numpy as np


# --- Mesh generation ---
class IOMesh:

    verbose = False

    @staticmethod
    def create_directory_with_check(directory):
        # Check if directory exists; if not create it
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return directory

    def write_mesh(self, mesh, write_path=r'C:\Data\Cones\Diff2.vtk'):
        if self.verbose:
            print('Writing mesh to {}'.format(write_path))
        writer = vtkPolyDataWriter()
        writer.SetFileName(write_path)
        writer.SetInputData(mesh.GetOutput())
        writer.Write()

    def write_points(self, mesh, write_path):
        point_cloud = vtkPolyData()
        point_cloud.SetPoints(mesh.GetOutput().GetPoints())
        writer = vtkPolyDataWriter()
        writer.SetInputData(point_cloud)
        writer.SetFileName(write_path)
        writer.Update()
        writer.Write()

    def write_points_as_array(self, mesh, write_path):
        numpy_array_of_points = np.array(dataset_adapter.WrapDataObject(mesh.GetOutput()).Points)
        np.savetxt(write_path, numpy_array_of_points, delimiter=",")

    def read_mesh(self, read_path=r'C:\Data\Cones\Endo.vtk'):
        if self.verbose:
            print('Reading mesh from {}'.format(read_path))
        reader = vtkPolyDataReader()
        reader.SetFileName(read_path)
        reader.Update()
        return reader


if __name__ == '__main__':
   pass