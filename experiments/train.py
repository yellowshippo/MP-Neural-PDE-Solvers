import argparse
import os
import pathlib
from distutils.util import strtobool
from typing import Tuple
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import common.datasets as datasets
from experiments.mpnpde import MP_PDE_Solver
from experiments.train_helper \
    import training_loop, test_timestep_losses, test_unrolled_losses, \
    save_prediction


def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists('experiments/log'):
        os.mkdir('experiments/log')
    if not os.path.exists('models'):
        os.mkdir('models')


def train(
        args: argparse,
        pde: str,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim,
        loader: DataLoader,
        graph_creator: datasets.GraphCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device = "cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random
    timestep.
    This is done for the number of timesteps in our training sample,
    which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()

    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the
    # max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one
    # trajectory.
    # Since the starting point is randomly drawn, this in expectation has every
    # possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is
    # covered.
    for i in range(graph_creator.t_res):
        losses = training_loop(
            model, unrolling, args.batch_size, optimizer, loader,
            graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(
                f"Training Loss (progress: {i / graph_creator.t_res:.2f}): "
                f"{torch.mean(losses)}")


def test(args: argparse,
         pde: str,
         model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: datasets.GraphCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device = "cpu") -> Tuple[torch.Tensor, dict]:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()

    # first we check the losses for different timesteps
    # (one forward prediction array!)
    steps = [
        t for t in range(
            graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(
        model=model,
        steps=steps,
        batch_size=args.batch_size,
        loader=loader,
        graph_creator=graph_creator,
        criterion=criterion,
        device=device)

    # next we test the unrolled losses
    losses, predictions = test_unrolled_losses(
        model=model,
        steps=steps,
        batch_size=args.batch_size,
        nr_gt_steps=args.nr_gt_steps,
        loader=loader,
        graph_creator=graph_creator,
        criterion=criterion,
        device=device)

    return torch.mean(losses), predictions


def main(args: argparse):

    device = args.device
    check_directory()

    if args.experiment == 'fluid':
        pde = 'ns'
        train_string = pathlib.Path('../data/fluid/preprocessed/train')
        valid_string = pathlib.Path('../data/fluid/preprocessed/validation')
        if args.transformed:
            test_string = pathlib.Path(
                '../data/fluid/transformed/preprocessed/test')
        else:
            test_string = pathlib.Path('../data/fluid/preprocessed/test')
    else:
        raise Exception("Wrong experiment")

    # Load datasets
    if args.mode == 'train':
        train_dataset = datasets.NpyDataset(
            train_string, pde=pde, mode='train',
            time_window=args.time_window)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)

        valid_dataset = datasets.NpyDataset(
            valid_string, pde=pde, mode='valid',
            time_window=args.time_window)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0)

    test_dataset = datasets.NpyDataset(
        test_string, pde=pde, mode='test',
        time_window=args.time_window)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    dateTimeObj = datetime.now()
    timestring = f"{dateTimeObj.date().month}{dateTimeObj.date().day}" \
        + f"{dateTimeObj.time().hour}{dateTimeObj.time().minute}" \
        + f"{dateTimeObj.time().second}"

    # if (args.log):
    #     logfile = f"experiments/log/{args.model}_{pde}_{args.experiment}_" \
    #         + f"n{args.neighbors}_tw{args.time_window}_" \
    #         + f"unrolling{args.unrolling}_time{timestring}.csv"
    #     print(f'Writing to log file {logfile}')
    #     sys.stdout = open(logfile, 'w')

    if args.save_directory is None:
        save_directory = pathlib.Path(
            f"models/MPPDE_{pde}_{args.experiment}_"
            + f"n{args.neighbors}_tw{args.time_window}_"
            + f"unrolling{args.unrolling}_time{timestring}")
    else:
        save_directory = args.save_directory
    save_directory.mkdir(parents=True)
    print(f'Training on dataset {train_string}')
    print(device)

    graph_creator = datasets.GraphCreator(
        pde=pde,
        neighbors=args.neighbors,
        time_window=args.time_window).to(device)

    if args.model == 'GNN':
        model = MP_PDE_Solver(
            pde=pde,
            time_window=graph_creator.tw,
            hidden_features=args.hidden_features,
            eq_variables={
                'tmax': graph_creator.tmax,
                'dt': graph_creator.dt,
            }
        ).to(device)
    else:
        raise Exception("Wrong model specified")
    if args.pretrained_model_file:
        model_state_dict = torch.load(
            args.pretrained_model_file, map_location=args.device)
        model.load_state_dict(model_state_dict)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')
    criterion = torch.nn.MSELoss(reduction="sum")

    # Optimizer
    if args.mode == 'train':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.unrolling, 5, 10, 15],
            gamma=args.lr_decay)

        # Training loop
        min_val_loss = 10e30
        test_loss = 10e30
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            train(
                args, pde, epoch, model, optimizer, train_loader,
                graph_creator, criterion, device=device)
            print("Evaluation on validation dataset:")
            val_loss, _ = test(
                args, pde, model, valid_loader, graph_creator, criterion,
                device=device)
            # Save model
            # save_path = save_directory / f"model_epoch{epoch}.pt"
            # torch.save(model.state_dict(), save_path)
            # print(f"Saved model at {save_path}")

            if (val_loss < min_val_loss):
                print("Evaluation on test dataset:")
                test_loss, test_prediction = test(
                    args, pde, model, test_loader, graph_creator, criterion,
                    device=device)
                # Save model
                save_path = save_directory / 'model.pt'
                torch.save(model.state_dict(), save_path)
                print(f"Saved model at {save_path}")

                # Save prediction
                save_prediction(pde, test_prediction, save_directory)
                min_val_loss = val_loss

            print('--')
            scheduler.step()

        print(f"Test loss: {test_loss}")

    elif args.mode == 'predict':
        steps = [
            t for t in range(
                graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
        test_losses, test_prediction = test_unrolled_losses(
            model=model,
            steps=steps,
            batch_size=args.batch_size,
            nr_gt_steps=args.nr_gt_steps,
            loader=test_loader,
            graph_creator=graph_creator,
            criterion=criterion,
            device=device)

        # Save prediction
        save_prediction(
            pde, test_prediction, save_directory, transformed=args.transformed)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument(
        '--device', type=str, default='cpu', help='Used device')
    parser.add_argument(
        '--experiment', type=str, default='fluid',
        help='Experiment for PDE solver should be trained: '
        '[fluid]')

    # Model
    parser.add_argument(
        '--model', type=str, default='GNN',
        help='Model used as PDE solver: [GNN]')

    # Model parameters
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Number of samples in each minibatch')
    parser.add_argument(
        '--num_epochs', type=int, default=20,
        help='Number of training epochs')
    parser.add_argument(
        '--hidden_features', type=int, default=128,
        help='Number of hidden features')
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate')
    parser.add_argument(
        '--lr_decay', type=float,
        default=0.4, help='multistep lr decay')

    parser.add_argument(
        '--neighbors', type=int,
        default=8, help="Neighbors to be considered in GNN solver")
    parser.add_argument(
        '--time_window', type=int,
        default=10, help="Time steps to be considered in GNN solver")
    parser.add_argument(
        '--unrolling', type=int,
        default=1, help="Unrolling which proceeds with each epoch")
    parser.add_argument(
        '--nr_gt_steps', type=int,
        default=1, help="Number of steps done by numerical solver")

    # Misc
    parser.add_argument(
        '--print_interval', type=int, default=1,
        help='Interval between print statements')
    parser.add_argument(
        '--log', type=eval, default=False,
        help='pip the output to log file')
    parser.add_argument(
        '--pretrained_model_file', type=pathlib.Path, default=None,
        help='Pretrained model file')
    parser.add_argument(
        '--mode', type=str, default='train',
        help='Mode of the script: [train, predict]')
    parser.add_argument(
        '--transformed',
        type=strtobool,
        default=0,
        help='If True, prediction on the transformed dataset [False]')
    parser.add_argument(
        '--save_directory',
        type=pathlib.Path,
        default=None,
        help='Path to save data')

    args = parser.parse_args()
    main(args)
