import numpy as np
from typing import Dict

# Class containing information on the reaction system
class ChemicalReactions:
    def __init__(
        self,
        reactions_dict: Dict[int, str],
        symbolic_species_variables: Dict[str, int],
        MStoic_path: str = 'data/MStoic.csv',
        TD_coeffs_path: str = 'data/TD_coeffs.csv'
    ) -> None:
        # Constants
        self.R: float = 8.314  # [J/(mol*K)] gas constant
        self.p_standard: float = 1.01325e5  # [Pa] standard pressure

        # Load stoichiometric matrix and thermodynamic coefficients
        self.MStoic: np.ndarray = np.loadtxt(
            MStoic_path,
            dtype=np.float64,
            delimiter=','
        )
        self.TD_coeffs: np.ndarray = np.loadtxt(
            TD_coeffs_path,
            dtype=np.float64,
            delimiter=','
        )

        self.reactions: Dict[int, str] = reactions_dict
        self.symbolic_species_variables: Dict[str, int] = symbolic_species_variables

        # Map thermodynamic coefficients to species
        self.TD_coeffs_dict: Dict[str, Dict[str, float]] = (
            self._map_shomate_coefficients()
        )

    def _map_shomate_coefficients(self) -> Dict[str, Dict[str, float]]:
        coefficients = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
        TD_coeffs_dict: Dict[str, Dict[str, float]] = {}
        for species, index in self.symbolic_species_variables.items():
            TD_coeffs_dict[species] = {
                coeff: self.TD_coeffs[i, index]
                for i, coeff in enumerate(coefficients)
            }
        return TD_coeffs_dict

    def print_reactions(self) -> None:
        print("Reactions:")
        for idx, reaction in self.reactions.items():
            print(f"Reaction {idx + 1}: {reaction}: Index {idx}")
        print()

    def print_stoichiometric_matrix(self) -> None:
        print("Stoichiometric Matrix (MStoic):")
        print("Each row corresponds to one reaction")
        print(
            "Each column corresponds to one gas species "
            "(see symbolic chem variables)"
        )
        print(self.MStoic)
        print()

    def print_symbolic_variables(self) -> None:
        print("Symbolic Variables:")
        for species, index in self.symbolic_species_variables.items():
            print(f"{species}: Index {index}")
        print()

    def print_TD_coefficients(self) -> None:
        print("Thermodynamic Coefficients:")
        print("Each row corresponds to one index, a1-a7")
        print(
            "Each column corresponds to one gas species "
            "(see symbolic chem variables)"
        )
        for species, coeffs in self.TD_coeffs_dict.items():
            print(f"{species}:")
            for coeff, value in coeffs.items():
                print(f"  {coeff}: {value}")
        print()

    def print_constants(self) -> None:
        print(f"R: {self.R}")
        print(f"p_standard: {self.p_standard}")
        print()

    def print_all(self) -> None:
        self.print_reactions()
        self.print_symbolic_variables()
        self.print_constants()
        self.print_stoichiometric_matrix()
        self.print_TD_coefficients()

