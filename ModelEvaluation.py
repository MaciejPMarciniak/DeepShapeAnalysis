import torch
from BetaVAE import BetaVAE, ConvBetaVAE
from DataLoading import DataLoading
from visualisation import show_reconstruction, show_changes_in_latent_variable
import numpy as np
import os

# TODO: Show changes along the latent variable distributions
# TODO: Show loss both in KL and reconstruction


class Evaluate:

    def __init__(self, model_path, model_type, data_path, mesh_info_file):
        self.model = model_type()
        print('loading state dict')
        self.model.load_state_dict(torch.load(model_path))
        print('evaluating state dict')
        self.model.eval()
        self.model_type = model_type
        self.point_cloud = None

        print('initializing data loader')
        self.dataloader = DataLoading(data_path, mesh_info_file, batch_size=1, validation_split=0,
                                      ravel=True if model_type == BetaVAE else False)

    def point_vector_to_point_cloud(self, point_vector):
        order = 'F' if self.model_type == ConvBetaVAE else 'C'
        point_cloud = point_vector.numpy().reshape((-1, 3), order=order)
        return point_cloud

    def inference_from_latent_space(self, latent_vector):
        with torch.no_grad():
            generated_point_cloud = self.model.decoder(latent_vector)
        generated_point_cloud = self.point_vector_to_point_cloud(generated_point_cloud)
        return generated_point_cloud

    def inference_from_scratch(self, point_cloud):
        with torch.no_grad():

            reconstructed_point_vector = self.model.encoder(point_cloud)[0]  # [0] -> mean value
            reconstructed_point_vector = self.model.decoder(reconstructed_point_vector)

        reconstructed_point_cloud = self.point_vector_to_point_cloud(reconstructed_point_vector)
        return reconstructed_point_cloud

    def reconstruct_point_cloud(self, sample_number, show_reconstruction_result=False):
        assert isinstance(sample_number, int) and 0 <= sample_number < 10000, 'Provide an integer between 0 and 10000'

        sample = self.dataloader.point_cloud_dataset[sample_number]
        point_cloud = sample['point_cloud']
        reconstructed_point_cloud = self.inference_from_scratch(point_cloud)

        if point_cloud.ndim == 1:
            point_cloud = self.point_vector_to_point_cloud(point_cloud)

        if show_reconstruction_result:
            show_reconstruction(point_cloud,
                                reconstructed_point_cloud,
                                sample_number)

        return reconstructed_point_cloud

    def generate_point_clouds(self, latent_variable_index):
        latent_space_vector = torch.zeros(30)
        latent_values = np.linspace(-5, 5, 11)
        generated_point_clouds = {}

        for latent_value in latent_values:
            latent_space_vector[latent_variable_index] = latent_value
            generated_point_cloud = self.inference_from_latent_space(latent_space_vector)
            generated_point_clouds[str(latent_value)] = generated_point_cloud

        return generated_point_clouds

    def analyze_latent_variable(self, latent_variable_index):
        generated_point_clouds = self.generate_point_clouds(latent_variable_index)
        show_changes_in_latent_variable(generated_point_clouds, latent_variable_index)


def evaluate_conv_beta_vae(model, epoch):
    model_path = os.path.join(r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Training/models', model,
                              model + '_e_' + str(epoch) + '.pt')
    evaluator = Evaluate(model_path,
                         ConvBetaVAE,
                         data_path=r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Data/GeneratedMeshes',
                         mesh_info_file=r'MeshDescription.csv')
    print('reconstructing')
    for i in range(7200, 7201):
        evaluator.reconstruct_point_cloud(i, show_reconstruction_result=True)
    for i in range(30):
        evaluator.analyze_latent_variable(i)


def evaluate_beta_vae(model, epoch):
    model_path = os.path.join(r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Training/models', model,
                              model + '_e_' + str(epoch) + '.pt')
    evaluator = Evaluate(model_path,
                         BetaVAE,
                         data_path=r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Data/GeneratedMeshes',
                         mesh_info_file=r'MeshDescription.csv')
    print('reconstructing')
    for i in range(7200, 7201):
        evaluator.reconstruct_point_cloud(i, show_reconstruction_result=True)
    for i in range(1):
        evaluator.analyze_latent_variable(i)


if __name__ == '__main__':

    linear_model = 'BetaVAE150WithDropoutBS10'
    convoluted_model = 'ConvBetaVAE150WithDropoutBS10'
    epoch = 106
    evaluate_beta_vae(linear_model, epoch)
    evaluate_conv_beta_vae(convoluted_model, epoch)

