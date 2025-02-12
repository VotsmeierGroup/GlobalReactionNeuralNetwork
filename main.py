# Description: Main script to train the Global Reaction Neural Network (GRNN) model.
import numpy as np
import torch
import copy

from GlobalReactionNeuralNetwork.chem_struct import ChemicalReactions
from GlobalReactionNeuralNetwork.model import GRNN
from GlobalReactionNeuralNetwork.load_data import Dataset
from GlobalReactionNeuralNetwork.helpers import RMS_RelativeError, RelativeError, check_accuracy

torch.set_default_dtype(torch.double)
torch.manual_seed(0)

# Training parameters
N_EPOCHS = 2500
N_TRAIN_SAMPLES = 1000
N_HIDDEN = 50
N_LAYERS = 1

if __name__ == "__main__":

    # Load data
    train = Dataset('train', n_data=N_TRAIN_SAMPLES)
    validation = Dataset('validation', train.data_min, train.data_max)
    test = Dataset('test', train.data_min, train.data_max)

    # Define chemical structure data
    reactions_dict = {
        0: "H2 + 1/2 O2 <=> H2O",
        1: "CO + 1/2 O2 <=> CO2",
        2: "H2O + CO <=> H2 + CO2"
    }

    # Define symbolic species variables
    symbolic_species_variables = {
        'H2': 0,
        'O2': 1,
        'H2O': 2,
        'CO': 3,
        'CO2': 4
    }

    # Load chemical structure data
    chem_struct = ChemicalReactions(
        reactions_dict=reactions_dict,
        symbolic_species_variables=symbolic_species_variables,
        MStoic_path='data/MStoic.csv',
        TD_coeffs_path='data/TD_coeffs.csv'
    )

    # Initialize GRNN model
    n_input = train.x.shape[1]
    n_global_reactions = chem_struct.MStoic.shape[0]
    model = GRNN(
        input_size=n_input,
        hidden_size=N_HIDDEN,
        output_size=n_global_reactions,
        num_layers=N_LAYERS,
        data_min=train.data_min,
        data_max=train.data_max,
        MStoic=chem_struct.MStoic,
        TD_coeffs=chem_struct.TD_coeffs,
        R=chem_struct.R,
        p_standard=chem_struct.p_standard
    )

    # Optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1,
        line_search_fn="strong_wolfe"
    )

    # Define loss function and evaluation metric
    criterion = RMS_RelativeError()
    error_metric = RelativeError()

    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    # Closure function for lbfgs optimizer
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        ypred = model(train.x).squeeze()
        loss = criterion(ypred, train.y)
        if loss.requires_grad:
            loss.backward()
        return loss

    best_val_loss = np.inf

    # Training loop
    for epoch in range(N_EPOCHS):
        optimizer.step(closure)

        if ((epoch + 1) % 10) == 0:
            trainLoss = check_accuracy(model, train, criterion, error_metric)
            train_loss_list.append(trainLoss['error_metric'])

            valLoss = check_accuracy(model, validation, criterion, error_metric)
            val_loss_list.append(valLoss['error_metric'])

            epoch_list.append(epoch)
            print(
                'Epoch: %d, Train loss: %.4f, Val loss: %.4f'
                % (epoch + 1, trainLoss['error_metric'], valLoss['error_metric'])
            )

            # Save best model on validation set to avoid overfitting
            if valLoss['error_metric'] < best_val_loss:
                best_val_loss = valLoss['error_metric']
                best_model = copy.deepcopy(model)
    
    # save the best model
    torch.save(best_model.state_dict(), 'data/model.pth')

    # save loss curve
    np.save('data/train_loss_list.npy', train_loss_list)
    np.save('data/val_loss_list.npy', val_loss_list)

