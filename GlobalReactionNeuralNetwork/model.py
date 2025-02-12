from typing import Sequence, Tuple
import torch

# Global Reaction Neural Network Class, containing all function for ensuring thermodynamic and stoichiometric consistency
class GRNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        data_min: Sequence[float],
        data_max: Sequence[float],
        MStoic: Sequence[Sequence[float]],
        TD_coeffs: Sequence[Sequence[float]],
        R: float,
        p_standard: float
    ) -> None:
        super().__init__()

        # Properties for rescaling of input data
        self.data_min = torch.tensor(data_min)
        self.data_max = torch.tensor(data_max)

        # Properties for GRNN
        self.MStoic = torch.tensor(MStoic) # stoichiometric matrix
        self.TD_coeffs = torch.tensor(TD_coeffs) # thermodynamic coefficients
        self.R = torch.tensor(R) # gas constant
        self.p_standard = torch.tensor(p_standard) # standard pressure

        # Fully connected neural network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))

        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))

        # Output layer
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # get pressure and temperature for thermodynamic calculations
        p = x[:, :-1]
        T = x[:, -1].reshape(-1, 1)

        # transform and scale input data for fully connected neural network
        x = self.transform_and_scale_p_T(x)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)

        ln_rforw = self.layers[-1](x)
        r_forw = torch.exp(ln_rforw) # forward rates are always positive

        s_dot = self.embedded_physics(r_forw, p, T) # calculating species rates using embedded physics
        return s_dot

    # Embedded physics, where the thermodynamic and stoichiometric layers are combined with the global forward rates
    # from the fully connected neural network
    def embedded_physics(self, r_forw: torch.Tensor, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        r_global = self.thermodynamic_layer(r_forw, T, p)
        s_dot = self.stoichiometric_layer(r_global)
        return s_dot

    # Transform and scale the input data for the fully connected neural network
    def transform_and_scale_p_T(self, x: torch.Tensor) -> torch.Tensor:
        # scale physical units to -1 to 1
        p = x[:, :-1]
        T = x[:, -1].reshape(-1, 1)
        p = torch.log(p)
        T = T**-1
        # scale the transformed data
        transformed_x = torch.cat((p, T), dim=1)
        transformed_and_scaled_x = 2 * (transformed_x - self.data_min) / (self.data_max - self.data_min) - 1
        return transformed_and_scaled_x
        
    # Thermodynamic layer. This layer must be tailored to the structure of the TD coefficients.
    #  Here, we use the NASA7 polynomial form:
    #  (A. Burcat, B. Ruscic, Third millenium ideal gas and condensed phase thermochemical database for combustion
    #  (with update from active thermochemical tables), 2005.)
    def thermodynamic_layer(self, r_forw: torch.Tensor, T: torch.Tensor,  p: torch.Tensor) -> torch.Tensor:
        dfH = self.calc_H(T)
        dfS = self.calc_S(T)
        K = self.calc_K(T, dfH, dfS)
        Q = self.calc_Q(p)
        r_global = self.de_Donder(r_forw, Q, K) 
        return r_global

    # De Donder equation to map the forward rates to global rates
    def de_Donder(self, r_forw: torch.Tensor, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        r_global = r_forw * (1 - Q / K)
        return r_global

    def calc_H(self, T: torch.Tensor) -> torch.Tensor:
        dfH = (
            self.TD_coeffs[0, :] * T
            + self.TD_coeffs[1, :] * T**2 / 2
            + self.TD_coeffs[2, :] * T**3 / 3
            + self.TD_coeffs[3, :] * T**4 / 4
            + self.TD_coeffs[4, :] * T**5 / 5
            + self.TD_coeffs[5, :]
        )
        dfH = self.R * dfH
        return dfH

    def calc_S(self, T: torch.Tensor) -> torch.Tensor:
        dfS = (
            self.TD_coeffs[0, :] * torch.log(T)
            + self.TD_coeffs[1, :] * T
            + self.TD_coeffs[2, :] * T**2 / 2
            + self.TD_coeffs[3, :] * T**3 / 3
            + self.TD_coeffs[4, :] * T**4 / 4
            + self.TD_coeffs[6, :]
        )
        dfS = self.R * dfS
        return dfS

    # Calculate the equilibrium constant from the thermodynamic data
    def calc_K(self, T: torch.Tensor, dfH: torch.Tensor, dfS: torch.Tensor) -> torch.Tensor:
        dfG = dfH - T * dfS
        dRG = torch.matmul(dfG, self.MStoic.T)
        K = torch.exp(-dRG / (self.R * T))
        return K

    # Calculate the reaction quotient
    def calc_Q(self, p: torch.Tensor) -> torch.Tensor:
        a = p / self.p_standard  # calculate the activity of each species using the partial pressures and the standard pressure
        Q = torch.exp(torch.matmul(torch.log(a), self.MStoic.T))
        return Q

    # Stoichiometric layer, using the stoichiometric matrix to map the global rates to species rates
    def stoichiometric_layer(self, r_global: torch.Tensor) -> torch.Tensor:
        s_dot = torch.matmul(r_global, self.MStoic)
        return s_dot

