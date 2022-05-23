import time
import pathlib
from typing import Tuple

import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from common.datasets import GraphCreator


DEBUG = False


def training_loop(
        model: torch.nn.Module,
        unrolling: list,
        batch_size: int,
        optimizer: torch.optim,
        loader: DataLoader,
        graph_creator: GraphCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory

    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list):
            list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for (u_base, u_super, x, variables) in loader:
        optimizer.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [
            t for t in range(
                graph_creator.tw,
                graph_creator.t_res - graph_creator.tw
                - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)
        graph = graph_creator.create_graph(
            data, labels, x, variables, random_steps).to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                pred = model(graph)
                graph = graph_creator.create_next_graph(
                    graph, pred, labels, random_steps).to(device)

        pred = model(graph)
        loss = criterion(torch.reshape(pred, graph.y.shape), graph.y)
        if DEBUG:
            save_ns(pred.detach().numpy(), pathlib.Path('tmp/pred'))
            save_ns(
                pred.reshape(pred.shape).detach().numpy(),
                pathlib.Path('tmp/ans'))

        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()

    losses = torch.stack(losses)
    return losses


def test_timestep_losses(
        model: torch.nn.Module,
        steps: list,
        batch_size: int,
        loader: DataLoader,
        graph_creator: GraphCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device = "cpu") -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the
    validation/test datasets

    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)
                graph = graph_creator.create_graph(
                    data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                loss = criterion(torch.reshape(pred, graph.y.shape), graph.y)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print(f'Step {step}, mean loss {torch.mean(losses)}')


def test_unrolled_losses(
        model: torch.nn.Module,
        steps: list,
        batch_size: int,
        nr_gt_steps: int,
        loader: DataLoader,
        graph_creator: GraphCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device = "cpu") -> Tuple[torch.Tensor, dict]:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    predictions = {}
    for i_data, loaded_data in enumerate(loader):
        u_base, u_super, x, variables = loaded_data
        data_directory = loader.dataset.data_directories[i_data]
        losses_tmp = []
        ans_tmp = []
        predictions_tmp = []
        print(f"Start prediction for: {data_directory}")
        with torch.no_grad():
            step = graph_creator.tw * nr_gt_steps
            print(f"Test step: {step}")
            same_steps = [step] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)

            graph_start_time = time.time()
            graph = graph_creator.create_graph(
                data, labels, x, variables, same_steps).to(device)
            graph_finish_time = time.time()
            graph_time = graph_finish_time - graph_start_time

            start_time = time.time()
            pred = model(graph)
            pred0_time = time.time()
            elapsed_time = pred0_time - start_time

            loss = criterion(torch.reshape(pred, graph.y.shape), graph.y)

            losses_tmp.append(loss / batch_size)
            ans_tmp.append(graph.y.reshape(pred.shape))
            predictions_tmp.append(pred)

            # Unroll trajectory and add losses which are obtained for each
            # unrolling, using the previous prediction
            for step in range(
                    graph_creator.tw * (nr_gt_steps + 1),
                    graph_creator.t_res - graph_creator.tw + 1,
                    graph_creator.tw):
                # TODO: Investigate appropreate nr_gt_steps value
                print(f"Test step: {step}")
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                graph = graph_creator.create_next_graph(
                    graph, pred, labels, same_steps).to(device)
                before_pred_time = time.time()
                pred = model(graph)
                after_pred_time = time.time()
                elapsed_time += after_pred_time - before_pred_time
                loss = criterion(
                    torch.reshape(pred, graph.y.shape), graph.y)

                losses_tmp.append(loss / batch_size)
                ans_tmp.append(graph.y.reshape(pred.shape))
                predictions_tmp.append(pred)

        predictions.update({
            data_directory: {
                'ans':
                torch.cat(ans_tmp, -1).cpu().detach().numpy(),
                'pred':
                torch.cat(predictions_tmp, -1).cpu().detach().numpy(),
                'graph_creation_time': graph_time,
                'prediction_time': elapsed_time,
            }
        })
        losses.append(torch.sum(torch.stack(losses_tmp)))
        print(f"prediction time: {elapsed_time}")
        print('--')

    losses = torch.stack(losses)
    print(f'Unrolled forward losses {torch.mean(losses)}')

    return losses, predictions


def save_prediction(
        pde: str,
        dict_prediction: dict,
        save_directory: pathlib.Path,
        transformed=False) -> None:
    log_file = save_directory / 'prediction.csv'
    with open(log_file, 'w') as f:
        f.write(
            'data_directory, '
            'graph_creation_time, '
            'prediction_time, '
            'loss_u, '
            'loss_p, '
            'loss\n')
    for data_directory, dict_data in dict_prediction.items():
        if transformed:
            sample_save_directory = save_directory \
                / data_directory.parent.parent.parent.parent.name \
                / data_directory.parent.parent.parent.name \
                / data_directory.parent.parent.name \
                / data_directory.parent.name / data_directory.name
        else:
            sample_save_directory = save_directory \
                / data_directory.parent.parent.parent.name \
                / data_directory.parent.parent.name \
                / data_directory.parent.name / data_directory.name
        sample_save_directory.mkdir(parents=True, exist_ok=True)
        if pde == 'ns':
            dict_loss = save_ns(dict_data, sample_save_directory)
        else:
            raise ValueError(f"Invalid PDE type: {pde}")
        with open(log_file, 'a') as f:
            f.write(
                f"{data_directory}, "
                f"{dict_data['graph_creation_time']}, "
                f"{dict_data['prediction_time']}, "
                f"{dict_loss['loss_u']}, "
                f"{dict_loss['loss_p']}, "
                f"{dict_loss['loss']}\n")
    return


def save_ns(
        dict_data: dict,
        save_directory: pathlib.Path) -> None:
    prediction_data = dict_data['pred']
    if prediction_data.shape[-1] % 4 != 0:
        raise ValueError(
            f"Invalid prediction_data shape for NS: {prediction_data.shape}")
    n_step = prediction_data.shape[-1] // 4
    for i_step in range(n_step):
        pu = prediction_data[:, 4*i_step:4*i_step+3]
        pp = prediction_data[:, [4*i_step+3]]
        np.save(save_directory / f"predicted_nodal_U_step{i_step+1}.npy", pu)
        np.save(save_directory / f"predicted_nodal_p_step{i_step+1}.npy", pp)
        # Time starts from 1 since 0 is the initial state

        if 'ans' in dict_data:
            answer_data = dict_data['ans']
            au = answer_data[:, 4*i_step:4*i_step+3]
            ap = answer_data[:, [4*i_step+3]]
            np.save(save_directory / f"answer_nodal_U_step{i_step+1}.npy", au)
            np.save(save_directory / f"answer_nodal_p_step{i_step+1}.npy", ap)

            if i_step == n_step - 1:
                loss_u = np.mean((pu - au)**2)
                loss_p = np.mean((pp - pu)**2)
                loss = np.mean([loss_u, loss_p])
    print(f"Predicted data saved in: {save_directory}")
    return {'loss_u': loss_u, 'loss_p': loss_p, 'loss': loss}
