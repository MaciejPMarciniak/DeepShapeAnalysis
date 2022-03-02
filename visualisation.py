from vtk import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkNamedColors, vtkPolyDataMapper, vtkActor
import matplotlib.pyplot as plt


# ---Mesh Generation---
class Visualisation:

    def __init__(self, mesh=None, point_cloud=None):
        self.mesh = mesh
        self.point_cloud = point_cloud

    def show_mesh(self):

        ren = vtkRenderer()
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        colors = vtkNamedColors()

        # Set the background color.
        bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
        colors.SetColor("BkgColor", *bkg)

        # The mapper is responsible for pushing the geometry into the graphics
        # library. It may also do color mapping, if scalars or other
        # attributes are defined.
        cylinderMapper = vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(self.mesh.GetOutputPort())

        # The actor is a grouping mechanism: besides the geometry (mapper), it
        # also has a property, transformation matrix, and/or texture map.
        # Here we set its color and rotate it -22.5 degrees.
        cylinderActor = vtkActor()
        cylinderActor.SetMapper(cylinderMapper)
        cylinderActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))

        # Add the actors to the renderer, set the background and size
        ren.AddActor(cylinderActor)
        ren.SetBackground(colors.GetColor3d("BkgColor"))
        renWin.SetSize(300, 300)
        renWin.SetWindowName('Cylinder')

        # This allows the interactor to initalize itself. It has to be
        # called before an event loop.
        iren.Initialize()

        # We'll zoom in a little by accessing the camera and invoking a "Zoom"
        # method on it.
        ren.ResetCamera()
        ren.GetActiveCamera().Zoom(1.5)
        renWin.Render()

        # Start the event loop.
        iren.Start()
# ---END Mesh Generation------------------------------------------------------------------------------------------------


# ---Training results---
def show_point_cloud_plt(point_cloud, index, ax, c='r'):
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color=c)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Mesh {} point cloud plot'.format(index))
    return ax


def show_point_clouds_batch(sample_batched):
    """Show mesh with description for a batch of samples."""
    point_clouds_batch, description_batch = sample_batched['point_cloud'], sample_batched['description']
    batch_size = point_clouds_batch.size(0)
    print(description_batch)

    fig = plt.figure(figsize=(14, 5))

    for i in range(batch_size):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        _ = show_point_cloud_plt(point_cloud=point_clouds_batch[i].numpy().reshape((-1, 3)),
                                 index=description_batch['index'][i].numpy(),
                                 ax=ax)
    plt.show()


def show_reconstruction(original_point_cloud, reconstructed_point_cloud, index):
    """Show original mesh (red) and its reconstruction (blue) through VAE"""
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    _ = show_point_cloud_plt(point_cloud=original_point_cloud, index=index, ax=ax, c='r')
    _ = show_point_cloud_plt(point_cloud=reconstructed_point_cloud, index='reconstruction '+str(index), ax=ax, c='b')

    plt.show()


def show_changes_in_latent_variable(point_clouds, variable_index):
    """Show how changes in latent variable affect the mesh generation"""
    fig = plt.figure(figsize=(14, 10))

    for i, item in enumerate(point_clouds.items()):
        ax = fig.add_subplot(3, 4, i+1, projection='3d')
        _ = show_point_cloud_plt(point_cloud=item[1],
                                 index='Lv index: {}, value: {}'.format(variable_index, item[0]), ax=ax, c='r')
    plt.show()


def show_loss(training_loss, validation_loss):
    """Show the loss history of training process (and validation dataset)"""
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot([x['sum_loss'] for x in training_loss], label='Training loss')
    plt.plot([x['sum_loss'] for x in validation_loss], label='Validation loss')
    plt.legend()
    ax.set_title('Training loss history', color='white')

    ax = fig.add_subplot(1, 3, 2)
    plt.plot([x['sum_loss'] for x in training_loss], label='Training loss')
    plt.plot([x['sum_loss'] for x in validation_loss], label='Validation loss')
    plt.plot([x['reconstruction_loss'] for x in training_loss], label='Training reconstruction loss')
    plt.plot([x['reconstruction_loss'] for x in validation_loss], label='Validation reconstruction loss')
    plt.legend()
    ax.set_title('Reconstruction loss history', color='white')

    ax = fig.add_subplot(1, 3, 3)
    plt.plot([x['sum_loss'] for x in training_loss], label='Training loss')
    plt.plot([x['sum_loss'] for x in validation_loss], label='Validation loss')
    plt.plot([x['kl_loss'] for x in training_loss], label='Training KL loss')
    plt.plot([x['kl_loss'] for x in validation_loss], label='Validation KL loss')
    plt.legend()
    ax.set_title('KL loss history', color='white')
    plt.show()
