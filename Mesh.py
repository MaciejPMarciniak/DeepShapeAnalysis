from vtk import vtkAppendFilter, vtkDataSetSurfaceFilter, vtkCellSizeFilter, vtkMassProperties, vtkTransformFilter, \
    vtkCenterOfMass, vtkTransform

import numpy as np
import os
import pandas as pd
from IO_mesh import IOMesh
from visualisation import Visualisation


# TODO: Extract points instead of meshes

# ----------------------------------------------------------------------------------------------------------------------
class TemplateVentricle(IOMesh):

    ELEMENTS = ['endo', 'epi', 'myo_plane']

    def __init__(self):
        super().__init__()
        self.template_ventricle_mesh = None
        self.template_ventricle_elements = {}
        self.bounds = None

    # --- Build ventricle ---
    @staticmethod
    def unstructured_grid_to_poly_data(mesh):
        print('transforming UG into PD')
        surface = vtkDataSetSurfaceFilter()
        surface.SetInputConnection(mesh.GetOutputPort())
        surface.Update()
        return surface

    def merge_elements(self, element_dictionary):
        print('Merging elements')
        merger = vtkAppendFilter()
        merger.MergePointsOn()
        for _, element in element_dictionary.items():
            merger.AddInputConnection(element.GetOutputPort())
        merger.Update()
        merger_polydata = self.unstructured_grid_to_poly_data(merger)
        return merger_polydata

    def create_ventricle(self):
        print('Reading template meshes: endo, epi and myo')
        self.read_ventricle_elements()
        print('Merging the meshes')
        self.template_ventricle_mesh = self.merge_elements(self.template_ventricle_elements)
        self.bounds = {x: y for x, y in zip(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'],
                                            self.template_ventricle_mesh.GetOutput().GetBounds())}

    # --- Visualisation
    def show_ventricle(self):
        v = Visualisation(self.template_ventricle_mesh)
        v.show_mesh()

    def show_elements(self):
        for element in self.ELEMENTS:
            v = Visualisation(self.template_ventricle_elements[element])
            v.show_mesh()

    # --- IO ---
    def read_ventricle_elements(self):
        self.template_ventricle_elements['endo'] = self.read_mesh(r'Data\TemplateEndoClean.vtk')
        self.template_ventricle_elements['epi'] = self.read_mesh(r'Data\TemplateEpiClean.vtk')
        self.template_ventricle_elements['myo_plane'] = self.read_mesh(r'Data\TemplateMyo.vtk')

    def save_ventricle(self, path):
        self.write_mesh(self.template_ventricle_mesh, path)

    # --- Mesh Information ---
    def update_bounds(self):
        self.bounds = {x: y for x, y in zip(['x_min_bound', 'x_max_bound', 'y_min_bound', 'y_max_bound', 'z_min_bound',
                                             'z_max_bound'], self.template_ventricle_mesh.GetOutput().GetBounds())}

    def measure_average_edge_length(self):
        print('Average edge length')
        size = vtkCellSizeFilter()
        size.SetInputConnection(self.template_ventricle_mesh.GetOutputPort())
        size.Update()
        print(size)

    def find_mesh_center_of_mass(self):
        centerofmass = vtkCenterOfMass()
        centerofmass.SetInputData(self.template_ventricle_mesh.GetOutput())
        centerofmass.Update()
        return np.array(centerofmass.GetCenter())
# ######################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
class TransformedVentricle(TemplateVentricle):

    def __init__(self, template_ventricle_elements, template_ventricle_mesh, output_folder, verbose=False):
        super().__init__()
        self.template_ventricle_elements = template_ventricle_elements
        self.template_ventricle_mesh = template_ventricle_mesh
        self.output_folder = output_folder
        self.transformed_ventricle_elements = template_ventricle_elements
        self.transformed_ventricle_mesh = template_ventricle_mesh
        self.update_bounds()
        self.df_log = pd.DataFrame()
        self.list_log = []
        self.verbose = verbose

    def transform_elements(self, transform_function):
        for element in self.ELEMENTS:
            transformer = vtkTransformFilter()
            transformer.SetInputConnection(self.transformed_ventricle_elements[element].GetOutputPort())
            transformer.SetTransform(transform_function)
            transformer.Update()
            self.transformed_ventricle_elements[element] = transformer

    def transform_mesh(self, transform_function):
        transformer = vtkTransformFilter()
        transformer.SetInputConnection(self.transformed_ventricle_mesh.GetOutputPort())
        transformer.SetTransform(transform_function)
        transformer.Update()
        self.transformed_ventricle_mesh = transformer

    def transform_mesh_and_elements(self, transform_function):
        self.transform_elements(transform_function)
        self.transform_mesh(transform_function)

    def scale(self, factor=(1.0, 1.0, 1.0)):
        if self.verbose:
            print('Scaling along axes\nAxis x: {}\nAxis y: {}\nAxis z: {} '.format(*factor))
        scale = vtkTransform()
        scale.Scale(*factor)
        self.transform_mesh_and_elements(scale)
        self.update_bounds()

    def rotate(self, around_x=0, around_y=0, around_z=0):
        if self.verbose:
            print('Rotating around axes (degrees)\nAround x: {}\nAround y: {}\nAround z: {}'.format(around_x, around_y,
                                                                                                    around_z))
        rotate = vtkTransform()
        rotate.Identity()
        rotate.RotateX(around_x)
        rotate.RotateY(around_y)
        rotate.RotateZ(around_z)
        self.transform_mesh_and_elements(rotate)
        self.update_bounds()

    def change_size(self, size_coefficient):
        if self.verbose:
            print('Changing size')
        self.scale(factor=[size_coefficient]*3)

    def change_sphericity(self, sphericity_coefficient):
        if self.verbose:
            print('Changing sphericity')
        self.scale(factor=(sphericity_coefficient, sphericity_coefficient, 1.0 / sphericity_coefficient))

    def change_ellipticity(self, ellipticity_coefficient):
        if self.verbose:
            print('Changing ellipticity')
        self.scale(factor=(ellipticity_coefficient, 1.0 / ellipticity_coefficient, 1.0))

    def change_orientation(self, rotation_angles):
        if self.verbose:
            print('Changing orientation')
        self.rotate(*rotation_angles)

    def deform_ventricle(self, deformation_coefficients):
        self.change_size(deformation_coefficients['size_coefficient'])
        self.change_sphericity(deformation_coefficients['sphericity_coefficient'])
        self.change_ellipticity(deformation_coefficients['ellipticity_coefficient'])
        self.change_orientation(deformation_coefficients['rotation_angles'])

    @staticmethod
    def generate_rotation_angles():
        around_x = np.random.uniform(-20, 20, 1)[0]
        around_y = np.random.uniform(-20, 20, 1)[0]
        around_z = np.random.uniform(-90, 90, 1)[0]
        return around_x, around_y, around_z

    def generate_deformation_coefficients(self):
        deformation_coefficients = {
            'size_coefficient': np.random.uniform(0.8, 1.2, 1)[0],
            'sphericity_coefficient': np.random.uniform(0.9, 1.2, 1)[0],
            'ellipticity_coefficient': np.random.uniform(0.8, 1.2, 1)[0],
            'rotation_angles': self.generate_rotation_angles()
        }
        return deformation_coefficients

    def save_mesh_and_point_cloud(self, index):
        mesh_path = os.path.join(self.output_folder, 'Meshes')
        self.create_directory_with_check(mesh_path)
        self.write_mesh(self.transformed_ventricle_mesh,
                        os.path.join(mesh_path, 'mesh_' + str(index).zfill(5) + '.vtk'))
        point_clouds_path = os.path.join(self.output_folder, 'PointClouds')
        self.create_directory_with_check(point_clouds_path)
        self.write_points_as_array(self.transformed_ventricle_mesh,
                                   os.path.join(point_clouds_path, 'point_cloud_' + str(index).zfill(5) + '.csv'))

    def generate_meshes(self, n=10000):
        for i in range(n):
            if i % 100 == 0 and i > 0:
                print('Generating mesh no. {}'.format(i))
                self.save_log()
            self.transformed_ventricle_mesh = self.template_ventricle_mesh
            self.transformed_ventricle_elements = self.template_ventricle_elements.copy()
            deformation_coefficients = self.generate_deformation_coefficients()
            self.deform_ventricle(deformation_coefficients)
            self.save_mesh_and_point_cloud(i)
            self.add_log_information(deformation_coefficients, i)

        self.save_log()

    # --- Logging mesh information ---
    def update_bounds(self):
        self.bounds = {x: y for x, y in zip(['x_min_bound', 'x_max_bound', 'y_min_bound', 'y_max_bound', 'z_min_bound',
                                             'z_max_bound'],
                                            self.transformed_ventricle_mesh.GetOutput().GetBounds())}

    def measure_blood_pool_volume(self):
        mass = vtkMassProperties()
        mass.SetInputConnection(self.transformed_ventricle_elements['endo'].GetOutputPort())
        return mass.GetVolume()

    def measure_myocardial_volume(self):
        mass = vtkMassProperties()
        mass.SetInputConnection(self.transformed_ventricle_elements['epi'].GetOutputPort())
        blood_pool_volume = self.measure_blood_pool_volume()
        myocardial_volume = mass.GetVolume() - blood_pool_volume
        return myocardial_volume

    def measure_ventricle_length(self, deformation_coefficients):
        z_min, z_max = self.template_ventricle_elements['endo'].GetOutput().GetBounds()[4:]
        length = (z_max - z_min) * deformation_coefficients['size_coefficient'] \
                 * 1.0 / deformation_coefficients['sphericity_coefficient']
        return length

    def lv_geometric_measurements(self, deformation_coefficients):
        blood_pool_volume = self.measure_blood_pool_volume() / 1000
        mass = self.measure_myocardial_volume() * 1.055 / 1000  # 1.055 from literature. Too thick?
        length = self.measure_ventricle_length(deformation_coefficients)
        if self.verbose:
            print('blood_pool_volume: {:1.1f} ml'.format(blood_pool_volume))
            print('mass: {:1.1f} g'.format(mass))
            print('length: {:1.1f} mm'.format(length))
            print('bounds: {}'.format(self.bounds))
        return blood_pool_volume, mass, length

    def add_log_information(self, deformation_coefficients, index):
        temp_log = {}
        blood_pool_volume, mass, length = self.lv_geometric_measurements(deformation_coefficients)
        temp_log.update({'index': index})
        temp_log.update({'blood_pool_volume': blood_pool_volume, 'mass': mass, 'length': length})
        temp_log.update(self.bounds)
        temp_log.update({key: deformation_coefficients[key] for key in ['size_coefficient',
                                                                        'sphericity_coefficient',
                                                                        'ellipticity_coefficient']})
        temp_log.update({key: value for key, value in zip(['rot_around_x', 'rot_around_y', 'rot_around_z'],
                                                          deformation_coefficients['rotation_angles'])})
        self.list_log.append(temp_log)

    def save_log(self, log_name='MeshDescription.csv'):
        df_log = pd.DataFrame(self.list_log)
        df_log.set_index('index', inplace=True)
        df_log.to_csv(os.path.join(self.output_folder, log_name))

    # --- Visualisation ---
    def show_ventricle(self):
        v = Visualisation(self.transformed_ventricle_mesh)
        v.show_mesh()

    def show_elements(self):
        for element in self.ELEMENTS:
            v = Visualisation(self.transformed_ventricle_elements[element])
            v.show_mesh()
# ######################################################################################################################


if __name__ == '__main__':
    iv = TemplateVentricle()
    iv.create_ventricle()

    tv = TransformedVentricle(iv.template_ventricle_elements, iv.template_ventricle_mesh,
                              output_folder=r'Data/GeneratedMeshes')
    tv.generate_meshes(n=10000)
