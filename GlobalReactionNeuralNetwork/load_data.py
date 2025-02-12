import numpy as np
import torch
from typing import Optional, Union, Sequence


# Dataset class containing model inputs and outputs
# The inputs consist of partial pressures [Pa] and temperature [K]
# The outputs consist of source terms of gas species [mol/(m^3 s)] at steady state surface coverages.
# This data is generated using a microkinetic model. See Section 3.1.1.1 for details:
# T. Kircher, F.A. Döppel, M. Votsmeier, Chem. Eng. J. (2024) 149863, http://dx.doi.org/10.1016/j.cej.2024.149863
class Dataset:
    def __init__(
        self,
        dataset_type: str,
        data_min: Optional[np.ndarray] = None,
        data_max: Optional[np.ndarray] = None,
        n_data: Optional[int] = None,
        species: Optional[Union[int, Sequence[int]]] = None
    ) -> None:
        if dataset_type == 'train':
            self.conditions: np.ndarray = np.loadtxt(
                'data/training_conditions.csv',
                delimiter=','
            )
            self.sdot: np.ndarray = np.loadtxt(
                'data/training_sdot.csv',
                delimiter=','
            )
        elif dataset_type == 'validation':
            self.conditions = np.loadtxt(
                'data/validation_conditions.csv',
                delimiter=','
            )
            self.sdot = np.loadtxt(
                'data/validation_sdot.csv',
                delimiter=','
            )
        elif dataset_type == 'test':
            self.conditions = np.loadtxt(
                'data/test_conditions.csv',
                delimiter=','
            )
            self.sdot = np.loadtxt(
                'data/test_sdot.csv',
                delimiter=','
            )
        else:
            print('dataset_type must be either "train", "validation" or "test"')
            return

        if species is not None:
            self.sdot = self.sdot[:, species]

        if n_data is not None:
            self.conditions = self.conditions[:n_data, :]
            self.sdot = self.sdot[:n_data, :]

        self.p_i: np.ndarray = self.conditions[:, :-1]  # [Pa] partial pressures
        self.T: np.ndarray = self.conditions[:, -1].reshape(-1, 1)  # [K] temperature

        if dataset_type == 'train':
            # find min and max of log(p_i) and 1/T, so data going into neural network is scaled between -1 and 1
            conditions_scaled: np.ndarray = np.zeros_like(self.conditions)
            conditions_scaled[:, :-1] = np.log(self.conditions[:, :-1])
            conditions_scaled[:, -1] = self.conditions[:, -1] ** -1
            self.data_min: np.ndarray = np.min(conditions_scaled, axis=0)
            self.data_max: np.ndarray = np.max(conditions_scaled, axis=0)
        else:
            self.data_min = data_min
            self.data_max = data_max

        self.x: np.ndarray = self.conditions

        self.x: torch.Tensor = torch.tensor(self.x)
        self.y: torch.Tensor = torch.tensor(self.sdot)

