import torch
import os
import numpy as np
from DataLoading import DataLoading
from BetaVAE import BetaVAE, ConvBetaVAE, Conv3DBetaVAE
from visualisation import show_loss


class Train:

    def __init__(self, epochs=200, learning_rate=0.0005, batch_size=10, validation_split=0.4,
                 data_path=r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Data/GeneratedMeshes',
                 description_path=r'MeshDescription.csv',
                 model_name='BetaVAE',
                 model_type=BetaVAE,
                 intermediate_output_path=''):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model_type = model_type
        self.intermediate_output_path = self.create_intermediate_output_path(intermediate_output_path)
        self.data_loading = DataLoading(data_path, description_path, batch_size, validation_split,
                                        ravel=True if model_type == BetaVAE else False)
        self.batch_size = batch_size
        self.running_average_loss_history = []
        self.running_average_val_loss_history = []

    @staticmethod
    def create_intermediate_output_path(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_trainig_and_validation_indices(self, training_loader, validation_loader):
        np.savetxt(os.path.join(self.intermediate_output_path, 'Training_indices.csv'),
                   sorted(training_loader.sampler.indices), fmt='%i', delimiter=',')
        np.savetxt(os.path.join(self.intermediate_output_path, 'Validation_indices.csv'),
                   sorted(validation_loader.sampler.indices), fmt='%i', delimiter=',')

    def train(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = self.model_type().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        training_loader, validation_loader = self.data_loading.build_shuffled_training_and_validation_data()
        self.save_trainig_and_validation_indices(training_loader, validation_loader)

        for e in range(self.epochs):

            losses = {'running_loss': 0.0, 'running_reconstruction_loss': 0.0, 'running_kl_loss': 0.0,
                      'val_running_loss': 0.0, 'val_running_reconstruction_loss': 0.0, 'val_running_kl_loss': 0.0}

            for s, samples in enumerate(training_loader):
                inputs = samples['point_cloud']
                inputs = inputs.to(device)
                outputs, mu, log_var = model(inputs)
                loss, reconstruction_loss, kl_loss = model.loss_function(outputs, inputs, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses['running_loss'] += loss.item()
                losses['running_reconstruction_loss'] += reconstruction_loss.item()
                losses['running_kl_loss'] += kl_loss.item()

            else:
                # validation - executed after the entire training batch is processed
                with torch.no_grad():  # to not affect the model/training process
                    for val_samples in validation_loader:
                        val_inputs = val_samples['point_cloud']
                        val_inputs = val_inputs.to(device)
                        val_outputs, val_mu, val_log_var = model(val_inputs)
                        val_loss, val_reconstruction_loss, val_kl_loss = \
                            model.loss_function(val_outputs, val_inputs, val_mu, val_log_var)

                        losses['val_running_loss'] += val_loss.item()
                        losses['val_running_reconstruction_loss'] += val_reconstruction_loss.item()
                        losses['val_running_kl_loss'] += val_kl_loss.item()

                training_size = float(len(training_loader) * self.batch_size)

                epoch_loss = losses['running_loss'] / training_size
                epoch_reconstruction_loss = losses['running_reconstruction_loss'] / training_size
                epoch_kl_loss = losses['running_kl_loss'] / training_size

                self.running_average_loss_history.append({'epoch': e,
                                                          'sum_loss': epoch_loss,
                                                          'reconstruction_loss': epoch_reconstruction_loss,
                                                          'kl_loss': epoch_kl_loss})

                validation_size = float(len(validation_loader) * self.batch_size)

                val_epoch_loss = losses['val_running_loss'] / validation_size
                val_epoch_reconstruction_loss = losses['val_running_reconstruction_loss'] / validation_size
                val_epoch_kl_loss = losses['val_running_kl_loss'] / validation_size

                self.running_average_val_loss_history.append({'epoch': e,
                                                              'sum_loss': val_epoch_loss,
                                                              'reconstruction_loss': val_epoch_reconstruction_loss,
                                                              'kl_loss': val_epoch_kl_loss})

                print('epoch: {}'.format(e + 1))
                print('training loss: {:.4f}, reconstruction loss: {:.4f}, '
                      'kl_loss: {:.4f}'.format(epoch_loss, epoch_reconstruction_loss, epoch_kl_loss))
                print('validation loss: {:.4f}, validation rec loss: {:.4f}, '
                      'validation kl_loss: {:.4f}'.format(val_epoch_loss, val_epoch_reconstruction_loss,
                                                          val_epoch_kl_loss))
                torch.save(model.state_dict(),
                           os.path.join(self.intermediate_output_path, self.model_name+'_e_{}.pt'.format(e+1)))

        return self.running_average_loss_history, self.running_average_val_loss_history

    def show_loss_plots(self):
        show_loss(self.running_average_loss_history, self.running_average_val_loss_history)

    def save_loss_information(self):
        # TODO: save the loss information as pandas DataFrames
        pass

    # TODO: show best/mean/median/worst cases from training and validation
    # TODO: better visualisation of latent space


if __name__ == '__main__':

    output_path = r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Training/models/ConvBetaVAE150WithDropoutBS10'
    test_model_name = output_path.rsplit('/', 1)[-1]

    training = Train(epochs=150, learning_rate=0.001, batch_size=10, validation_split=0.4,
                     data_path=r'C:/Users/mm18/PycharmProjects/DeepShapeAnalysis/Data/GeneratedMeshes',
                     description_path=r'MeshDescription.csv',
                     model_name=test_model_name,
                     model_type=Conv3DBetaVAE,
                     intermediate_output_path=output_path)

    training.train()
    training.show_loss_plots()
