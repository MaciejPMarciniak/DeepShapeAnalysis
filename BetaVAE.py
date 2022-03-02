import torch
from torch import nn
import torch.nn.functional as F
from visualisation import show_reconstruction


class BetaVAE(nn.Module):
    def __init__(self, x=3246, h1=1024, h2=512, h3=128, z=30, beta=1):
        super().__init__()
        self.beta = beta

        # encoder
        self.fc1 = nn.Linear(x, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_mean = nn.Linear(h3, z)
        self.fc_sd = nn.Linear(h3, z)

        # decoder
        self.fc4 = nn.Linear(z, h3)
        self.fc5 = nn.Linear(h3, h2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(h2, h1)
        self.fc7 = nn.Linear(h1, x)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        return self.fc_mean(x), self.fc_sd(x)  # mu, log_var

    def decoder(self, x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout2(x)
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

    @staticmethod
    def sampling(sample_mu, sample_log_var):
        std = torch.exp(0.5 * sample_log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(sample_mu)

    def loss_function(self, reconstructed_x, x, loss_mu, loss_log_var):
        mse_loss = nn.MSELoss(reduction='sum')
        reconstruction_loss = mse_loss(reconstructed_x, x)
        regularized_term = -0.5 * torch.sum(1 + loss_log_var - loss_mu.pow(2) - loss_log_var.exp())
        return reconstruction_loss + self.beta * regularized_term, reconstruction_loss, self.beta * regularized_term

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class ConvBetaVAE(nn.Module):
    def __init__(self, input_size=1082, h1=512, h2=256, h3=128, h4=64, z=30, beta=1, use_bias=True):
        super().__init__()
        self.beta = beta
        self.z = z
        self.input_size = input_size
        self.use_bias = use_bias

        # encoder
        self.encoder_structure = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=h4, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=h4, out_channels=h3, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=h3, out_channels=h2, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            # nn.Conv1d(in_channels=h2, out_channels=h2, kernel_size=1,
            #           bias=self.use_bias),
            # nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=h2, out_channels=h1, kernel_size=1,
                      bias=self.use_bias),
        )

        self.conv2lin = nn.Sequential(
            nn.Linear(h1, h3, bias=True),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(h3, self.z, bias=True)
        self.std_layer = nn.Linear(h3, self.z, bias=True)

        # decoder
        self.decoder_structure = nn.Sequential(
            nn.Linear(in_features=self.z, out_features=h4, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=h4, out_features=h3, bias=self.use_bias),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=h3, out_features=h1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            # nn.Linear(in_features=h2, out_features=h1, bias=self.use_bias),
            # nn.ReLU(inplace=True),

            nn.Linear(in_features=h1, out_features=self.input_size * 3, bias=self.use_bias),
        )

    def encoder(self, x):
        output_1 = self.encoder_structure(x)
        output_2 = output_1.max(dim=2)[0]
        linear_output = self.conv2lin(output_2)
        return self.mu_layer(linear_output), self.std_layer(linear_output)

    def decoder(self, x):
        output = self.decoder_structure(x)
        return output

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    @staticmethod
    def sampling(sample_mu, sample_log_var):
        std = torch.exp(0.5 * sample_log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(sample_mu)

    def loss_function(self, reconstructed_x, x, loss_mu, loss_log_var):
        reconstructed_x = reconstructed_x.view(x.shape)
        mse_loss = nn.MSELoss(reduction='sum')
        reconstruction_loss = mse_loss(reconstructed_x, x)
        regularized_term = -0.5 * torch.sum(1 + loss_log_var - loss_mu.pow(2) - loss_log_var.exp())
        return reconstruction_loss + self.beta * regularized_term, reconstruction_loss, self.beta * regularized_term


class Conv3DBetaVAE(nn.Module):
    def __init__(self, input_size=1082, h1=512, h2=256, h3=128, h4=64, z=30, beta=1, use_bias=True):
        super().__init__()
        self.beta = beta
        self.z = z
        self.input_size = input_size
        self.use_bias = use_bias

        # encoder
        self.encoder_structure = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=h4, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=h4, out_channels=h3, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=h3, out_channels=h2, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=h2, out_channels=h1, kernel_size=1,
                      bias=self.use_bias),
        )

        self.conv2lin = nn.Sequential(
            nn.Linear(h1, h3, bias=True),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(h3, self.z, bias=True)
        self.std_layer = nn.Linear(h3, self.z, bias=True)

        # decoder
        self.decoder_structure = nn.Sequential(
            nn.Linear(in_features=self.z, out_features=h4, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=h4, out_features=h3, bias=self.use_bias),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=h3, out_features=h1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=h1, out_features=self.input_size * 3, bias=self.use_bias),
        )

    def encoder(self, x):
        output_1 = self.encoder_structure(x)
        output_2 = output_1.max(dim=2)[0]
        linear_output = self.conv2lin(output_2)
        return self.mu_layer(linear_output), self.std_layer(linear_output)

    def decoder(self, x):
        output = self.decoder_structure(x)
        return output

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    @staticmethod
    def sampling(sample_mu, sample_log_var):
        std = torch.exp(0.5 * sample_log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(sample_mu)

    def loss_function(self, reconstructed_x, x, loss_mu, loss_log_var):
        reconstructed_x = reconstructed_x.view(x.shape)
        mse_loss = nn.MSELoss(reduction='sum')
        reconstruction_loss = mse_loss(reconstructed_x, x)
        regularized_term = -0.5 * torch.sum(1 + loss_log_var - loss_mu.pow(2) - loss_log_var.exp())
        return reconstruction_loss + self.beta * regularized_term, reconstruction_loss, self.beta * regularized_term