"""Leakage, architecture and experimental protocol regression tests."""
import unittest
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from test_gnn_lstm import synthetic_dataset
from data import YieldSATDataset, load_dataset
from preprocessing import SpectralTransform
from diagnostics import data_diagnostics
from model import GNN_LSTM_Model, DirectionalGNN, mean_neighbor_adjacency, directional_adjacencies
from graphs import build_grid_graph, DIRECTIONS, collate_graphs
from train import model_training, model_validation, parse_args, run_cross_validation, load_checkpoint
from prepareFold import split_dataset_by_group, split_manifest, splits_from_manifest, inner_field_split
from compare import paired_bootstrap, align_predictions
from eval import regression_metrics, save_evaluation


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_normalization_training_only_and_missing(self):
        x = np.array([[[1., np.nan], [3., 5.]], [[1000., 2000.], [np.inf, -1.]]])
        transform = SpectralTransform(normalize=True, append_mask=True, missing_value=-1).fit(x, [0])
        np.testing.assert_allclose(transform.mean, [2, 5])
        self.assertEqual(transform.count, [2, 1])
        normalized, mask = transform(x)
        np.testing.assert_allclose(normalized[0], [[-1, 0, 1, 0], [1, 0, 1, 1]])
        self.assertFalse(mask[1, 1])
        self.assertTrue(np.isfinite(normalized).all())
        restored = SpectralTransform(**transform.state_dict())
        np.testing.assert_array_equal(restored(x)[0], normalized)
        # Held-out values never alter the fitted transform.
        x[1] *= 100
        np.testing.assert_allclose(SpectralTransform(normalize=True).fit(x, [0]).mean, [2, 5])

    def test_shared_arrays_and_masks(self):
        ds = synthetic_dataset()
        ds['sample'].values[0, 0] = np.nan
        transform = SpectralTransform(normalize=True, append_mask=True).fit(ds['sample'].values, [0, 1, 2])
        selected = YieldSATDataset(ds, indices=['p0', 'p1', 'p2'], band=None, transform=transform)
        self.assertTrue(np.shares_memory(selected.samples, ds['sample'].values))
        self.assertEqual(selected.num_features, 4)
        self.assertFalse(selected[0]['time_mask'][0, 0])
        report = data_diagnostics(ds, transform)
        self.assertGreater(report['fully_unobserved_bin_percent'], 0)

    def test_plain_lstm_has_no_spatial_influence(self):
        model = GNN_LSTM_Model(2, hidden_dim=4, model_type='lstm', dropout=0).eval()
        self.assertEqual(model.lstm.input_size, 2)
        self.assertEqual(len(model.gnn), 0)
        x = torch.randn(3, 4, 2)
        edges = torch.tensor([[0, 1], [1, 0]])
        before = model(x, edges)
        x[1] += 100
        after = model(x, edges)
        torch.testing.assert_close(before[0], after[0])
        after.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_residual_dimensions_input_and_gradients(self):
        for layers in (1, 2):
            for connectivity in (4, 8):
                model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=3,
                                       gnn_layers=layers, residual=True, dropout=0)
                self.assertEqual(model.lstm.input_size, 5)
                for p in model.parameters():
                    torch.nn.init.constant_(p, 0.1)
                x = torch.ones(3, 4, 2, requires_grad=True)
                captured = []
                handle = model.lstm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
                model(x, build_grid_graph([0, 0, 1], [0, 1, 1], connectivity)).sum().backward()
                handle.remove()
                torch.testing.assert_close(captured[0][:, :, :2], x)
                self.assertGreater(x.grad.abs().sum().item(), 0)
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_directional_messages_and_permutation(self):
        positions = torch.tensor([[0, 0], [0, 1], [1, 0]])
        edges = build_grid_graph(positions[:, 0], positions[:, 1], 4)
        x = torch.tensor([[[1.]], [[2.]], [[3.]]])
        layer = DirectionalGNN(1, 1)
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            for linear in layer.neighbor_linears:
                linear.weight.zero_()
            layer.neighbor_linears[DIRECTIONS.index((0, 1))].weight.fill_(2)
            layer.neighbor_linears[DIRECTIONS.index((1, 0))].weight.fill_(4)
        a = mean_neighbor_adjacency(edges, 3, x.dtype)
        self.assertAlmostEqual(layer(x, directional_adjacencies(a, positions))[0].item(), 8)
        model = GNN_LSTM_Model(1, hidden_dim=3, gnn_hidden_dim=3, aggregation='directional', dropout=0).eval()
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(model(x, edges, positions)[permutation],
                                   model(x[permutation], inverse[edges], positions[permutation]))
        with self.assertRaisesRegex(ValueError, 'positions'):
            model(x, edges)

    def test_attention_masks_normalization_and_all_missing(self):
        for residual in (False, True):
            model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=3, residual=residual,
                                   temporal_pool='attention', attention_mask=True, dropout=0)
            x = torch.randn(3, 4, 2)
            mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
            edges = build_grid_graph([0, 0, 0], [0, 1, 2])
            prediction, weights = model(x, edges, time_mask=mask, return_attention=True)
            torch.testing.assert_close(weights.sum(1), torch.tensor([1., 0., 1.]))
            self.assertTrue((weights[~mask] == 0).all())
            self.assertTrue(torch.isfinite(prediction).all())
            torch.testing.assert_close(prediction[1], model.fc.bias[0])
            prediction.sum().backward()
            self.assertGreater(model.attention.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_pixel_accumulation_matches_full_pixel_gradient(self):
        ds = synthetic_dataset().isel(index=[0, 1, 2, 6])
        graphs = YieldSATDataset(ds, band=None)
        loader = DataLoader(graphs, batch_size=1, collate_fn=collate_graphs)
        model = GNN_LSTM_Model(2, hidden_dim=3, model_type='lstm', dropout=0)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        stats = {}
        model_training(model, loader, torch.nn.MSELoss(), optimizer, 'cpu',
                       objective='pixel', accumulation_steps=0, gradient_clip=0, stats=stats)
        batch = collate_graphs([graphs[i] for i in range(len(graphs))])
        loss = (reference(batch['sample'], batch['edge_index']) - batch['target']).square().mean()
        loss.backward()
        reference_optimizer.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q)
        self.assertEqual(stats['optimizer_steps'], 1)
        self.assertGreater(stats['gradient_norm_mean'], 0)

    def test_pixel_weighting_differs_from_equal_graph_updates(self):
        ds = synthetic_dataset().isel(index=[0, 1, 2, 6, 12])
        graphs = YieldSATDataset(ds, band=None)
        loader = DataLoader(graphs, batch_size=1, collate_fn=collate_graphs)
        model = GNN_LSTM_Model(2, hidden_dim=3, model_type='lstm', dropout=0)
        reference = copy.deepcopy(model)
        stats = {}
        model_training(model, loader, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.01),
                       'cpu', objective='pixel', accumulation_steps=2, gradient_clip=0, stats=stats)
        # Independent manual reference includes a partial final accumulation window.
        opt = torch.optim.SGD(reference.parameters(), lr=0.01)
        for window in ([graphs[0], graphs[1]], [graphs[2]]):
            opt.zero_grad()
            loss = sum((reference(g['sample'], g['edge_index'])-g['target']).square().sum() for g in window)
            (loss / (len(window) * 5/3)).backward()
            opt.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q)
        self.assertEqual(stats['optimizer_steps'], 2)

    def test_manifests_and_inner_groups(self):
        ds = synthetic_dataset()
        splits = split_dataset_by_group(ds, 2)
        manifest = split_manifest(ds, splits)
        self.assertEqual(splits, splits_from_manifest(ds, manifest, 2))
        train, val = inner_field_split(ds, splits[0][0], 0.5, 9)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(ds.sel(index=train).field_shared_name.values) & set(ds.sel(index=val).field_shared_name.values))
        self.assertEqual(set(train + val), set(splits[0][0]))
        with self.assertRaisesRegex(ValueError, 'fields'):
            splits_from_manifest(ds.isel(index=slice(0, 18)), manifest, 2)

    def test_official_encoded_metadata_and_string_crop(self):
        ds = synthetic_dataset()
        ds['crop'] = ('index', ['rapeseed'] * 24)
        ds['year'] = ('index', np.tile(np.repeat([0, 1], 3), 4))
        ds.year.attrs = {'0': 2020, '1': 2021}
        ds['field_shared_name'] = ('index', np.repeat(np.arange(8), 3))
        ds.field_shared_name.attrs = {str(i): f'Germany_farm1_field{i//2}_rapeseed_{2020+i%2}' for i in range(8)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'official_schema.nc'
            ds.to_netcdf(path)
            loaded = load_dataset(path, country='Germany', bands=['B02', 'B01'])
        self.assertEqual(loaded.band.values.tolist(), ['B02', 'B01'])
        self.assertEqual(set(loaded.year.values), {2020, 2021})
        self.assertEqual(len(np.unique(loaded.field_shared_name)), 4)
        self.assertEqual(len(YieldSATDataset(loaded, band=None)), 8)
        # Direct splitter callers get the same protection as the training loader.
        for train, val in split_dataset_by_group(ds, 2):
            self.assertTrue(set(loaded.sel(index=train).field_shared_name.values).isdisjoint(
                loaded.sel(index=val).field_shared_name.values))

    def test_all_ablations_checkpoint_reload_and_evaluation(self):
        configs = json.loads((Path(__file__).parent / 'ablations.json').read_text())
        configs['combined'] = {'args': ['--normalize', '--append-mask', '--residual', '--aggregation', 'directional',
            '--temporal-pool', 'attention', '--attention-mask', '--export-attention',
            '--export-training-predictions',
            '--objective', 'pixel', '--accumulation-steps', '2', '--scheduler', 'step', '--scheduler-patience', '1']}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / 'synthetic.nc'
            ds = synthetic_dataset()
            ds['crop'] = ('index', ['rapeseed'] * 24)
            ds['sample'].values[0, 0] = np.nan
            ds.to_netcdf(source)
            selected = load_dataset(source, bands=['B01', 'B02'])
            for name, entry in configs.items():
                with self.subTest(experiment=name):
                    out = root / name
                    args = parse_args(['--data', str(source), '--bands', 'B01', 'B02', '--folds', '2',
                        '--epochs', '2', '--hidden-dim', '3', '--gnn-hidden-dim', '3', '--dropout', '0',
                        '--device', 'cpu', '--output-dir', str(out), *entry['args']])
                    if name != 'A':
                        args.splits = root / 'A/splits.json'
                    results = run_cross_validation(args)
                    model, transform, checkpoint = load_checkpoint(out / 'fold_0.pt')
                    val = YieldSATDataset(selected, indices=checkpoint['val_indices'], band=None,
                                          connectivity=checkpoint['connectivity'], transform=transform)
                    frame, _ = model_validation(model, DataLoader(val, batch_size=1, collate_fn=collate_graphs),
                                                torch.nn.MSELoss(), 'cpu', export_attention=True)
                    np.testing.assert_allclose(frame.prediction, results[results.cv_fold == 0].prediction, rtol=1e-6)
                    for file in ('data_diagnostics.json', 'per_fold_metrics.csv', 'fold_summary.json',
                                 'train_mean_baseline_metrics.json', 'environment.json', 'splits.json'):
                        self.assertTrue((out / file).exists())
                    self.assertEqual(json.loads((out / 'splits.json').read_text()), json.loads((root / 'A/splits.json').read_text()))
                    if name == 'combined':
                        self.assertIn('attention_0', results)
                        history = pd.read_csv(out / 'history.csv')
                        np.testing.assert_allclose(history.train_rmse ** 2, history.train_mse)
                        self.assertTrue(np.isfinite(history.train_r2).all())
                        training_csv = pd.read_csv(out / 'fold_0_training_predictions.csv')
                        self.assertEqual(set(training_csv['index']), set(checkpoint['train_indices']))
                        self.assertFalse(set(training_csv['index']) & set(checkpoint['val_indices']))
                        self.assertEqual(set(training_csv['split']), {'train'})
                        training_data = YieldSATDataset(selected, indices=checkpoint['train_indices'], band=None,
                            connectivity=checkpoint['connectivity'], transform=transform)
                        reloaded, _ = model_validation(model, DataLoader(training_data, batch_size=1,
                            collate_fn=collate_graphs), torch.nn.MSELoss(), 'cpu')
                        np.testing.assert_allclose(training_csv.prediction, reloaded.prediction, rtol=1e-6)
                        np.testing.assert_allclose(training_csv.residual, training_csv.prediction-training_csv.target, atol=1e-6)
                        np.testing.assert_allclose(history[history.cv_fold == 0].learning_rate, [0.001, 0.0005])
                        save_evaluation(results, out, plots=True, history=history)
                        for file in ('pixel_evaluation.png', 'field_year_evaluation.png', 'training_curves.png'):
                            self.assertGreater((out / file).stat().st_size, 0)

    def test_inner_selection_does_not_depend_on_outer_targets(self):
        ds = synthetic_dataset()
        ds['crop'] = ('index', ['rapeseed'] * 24)
        splits = split_dataset_by_group(ds, 2)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            states = []
            for version in range(2):
                source = root / f'v{version}.nc'
                out = root / f'run{version}'
                altered = ds.copy(deep=True)
                if version:
                    altered['target'].loc[{'index': splits[0][1]}] += 100
                altered.to_netcdf(source)
                args = parse_args(['--data', str(source), '--bands', 'B01', 'B02', '--folds', '2',
                    '--epochs', '4', '--hidden-dim', '3', '--gnn-hidden-dim', '3', '--device', 'cpu',
                    '--output-dir', str(out), '--normalize', '--inner-val-fraction', '0.5',
                    '--export-training-predictions',
                    '--early-stopping-patience', '1', '--min-delta', '1000000', '--scheduler', 'plateau'])
                run_cross_validation(args)
                _, transform, state = load_checkpoint(out / 'fold_0.pt')
                states.append(state)
                self.assertEqual(state['epoch'], 1)
                self.assertEqual(state['selection'], 'inner_mse')
                training_csv = pd.read_csv(out / 'fold_0_training_predictions.csv')
                self.assertEqual(set(training_csv['epoch']), {state['epoch']})
                self.assertEqual(set(training_csv['index']), set(state['train_indices']))
                self.assertFalse(set(training_csv['index']) & set(state['inner_indices'] + state['val_indices']))
                history = pd.read_csv(out / 'history.csv')
                self.assertEqual(len(history[history.cv_fold == 0]), 2)
                self.assertFalse(set(state['train_indices']) & set(state['inner_indices']))
                positions = ds.index.to_index().get_indexer(state['train_indices'])
                np.testing.assert_allclose(transform.mean, ds['sample'].values[positions].mean((0, 1)), rtol=1e-6)
            self.assertEqual(states[0]['preprocessing'], states[1]['preprocessing'])
            for key in states[0]['model_state_dict']:
                torch.testing.assert_close(states[0]['model_state_dict'][key], states[1]['model_state_dict'][key])

    def test_paired_field_bootstrap_and_metric_edge_cases(self):
        ds = synthetic_dataset()
        a = pd.DataFrame(dict(index=ds.index.values, field_shared_name=ds.field_shared_name.values,
            year=ds.year.values, target=ds.target.values, prediction=ds.target.values.astype(float) + 1,
            cv_fold=np.repeat([0, 1, 0, 1], 6)))
        b = a.copy()
        b.prediction += 1
        report = paired_bootstrap(a, b, repetitions=100, seed=7)
        self.assertEqual(report, paired_bootstrap(a, b.sample(frac=1, random_state=3), repetitions=100, seed=7))
        for level in ('pixel', 'field_year'):
            self.assertAlmostEqual(report[level]['mse']['difference'], 3)
            np.testing.assert_allclose(report[level]['mse']['ci95'], [3, 3])
            np.testing.assert_allclose(report[level]['rmse']['ci95'], [1, 1])
        b.loc[0, 'target'] += 1
        with self.assertRaisesRegex(ValueError, 'targets'):
            align_predictions(a, b)
        metrics = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertIsNone(metrics['pearson'])
        self.assertEqual(metrics['prediction_target_variance_ratio'], 0)


if __name__ == '__main__':
    unittest.main()
