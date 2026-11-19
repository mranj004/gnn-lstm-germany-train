"""Spatial/temporal axis, field isolation, and checkpoint tests for 3D encoders."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import YieldSATDataset, load_dataset
from graphs import build_grid_graph, collate_graphs
from model import GNN_LSTM_Model, SpatialCNN3D
from test_gnn_lstm import synthetic_dataset
from train import forward_batch, load_checkpoint, model_validation, parse_args, run_cross_validation


class CNN3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_dense_encoder_matches_explicit_patches_and_gradients(self):
        # Unordered, translated coordinates with holes and unequal row/col extents.
        positions = torch.tensor([[12, 24], [10, 21], [12, 21], [11, 23]])
        x = torch.randn(4, 3, 2, requires_grad=True)
        for temporal in (1, 3):
            with self.subTest(temporal=temporal):
                encoder = SpatialCNN3D(2, 3, spatial_kernel=5, temporal_kernel=temporal).eval()
                actual = encoder(x, positions, torch.tensor([0, 4]))
                patches = []
                lookup = {tuple(p): i for i, p in enumerate(positions.tolist())}
                for row, col in positions.tolist():
                    patch = x.new_zeros(2, 3, 5, 5)
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            node = lookup.get((row + dr, col + dc))
                            if node is not None:
                                patch[:, :, dr + 2, dc + 2] = x[node].t()
                    patches.append(patch)
                reference = F.conv3d(torch.stack(patches), encoder.conv.weight, encoder.conv.bias,
                                     padding=(temporal // 2, 0, 0)).squeeze(-1).squeeze(-1).transpose(1, 2)
                reference = encoder.activation(encoder.norm(reference.reshape(-1, 3)).reshape(4, 3, 3))
                torch.testing.assert_close(actual, reference)
                inputs = (x, encoder.conv.weight, encoder.conv.bias)
                actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
                reference_grad = torch.autograd.grad(reference.square().sum(), inputs)
                for a, b in zip(actual_grad, reference_grad):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_spatial_orientation_temporal_extent_and_node_order(self):
        positions = torch.tensor([[10, 10], [12, 10], [10, 8], [40, 40]])
        encoder = SpatialCNN3D(1, 1).eval()
        with torch.no_grad():
            encoder.conv.weight.zero_()
            encoder.conv.bias.zero_()
            encoder.conv.weight[0, 0, 0, 4, 2] = 2  # two rows down
            encoder.conv.weight[0, 0, 0, 2, 0] = 3  # two columns left
        x = torch.zeros(4, 3, 1)
        x[1, 1, 0], x[2, 1, 0] = 5, 7
        output = encoder(x, positions, torch.tensor([0, 4]))
        self.assertAlmostEqual(output[0, 1, 0].item(), 31 / np.sqrt(1 + encoder.norm.eps), places=4)
        torch.testing.assert_close(output[0, [0, 2]], torch.zeros(2, 1))
        torch.testing.assert_close(output[3], torch.zeros(3, 1))
        permutation = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(output[permutation],
            encoder(x[permutation], positions[permutation] + 100, torch.tensor([0, 4])))
        temporal = SpatialCNN3D(1, 1, temporal_kernel=3).eval()
        with torch.no_grad():
            temporal.conv.weight.fill_(1)
            temporal.conv.bias.zero_()
        self.assertTrue((temporal(x, positions, torch.tensor([0, 4]))[0, :, 0] > 0).all())

    def test_batch_isolation_with_overlapping_coordinates_and_graph_edges(self):
        graphs = YieldSATDataset(synthetic_dataset(), band=None)
        batch = collate_graphs([graphs[0], graphs[1]])  # same coordinates, different years
        self.assertEqual(batch['graph_ptr'].tolist(), [0, 3, 6])
        for kind in ('3d_lstm', '3d_gnn_lstm'):
            with self.subTest(model=kind):
                model = GNN_LSTM_Model(2, model_type=kind, cnn_hidden_dim=4, hidden_dim=4,
                                       gnn_hidden_dim=3, dropout=0).eval()
                before = forward_batch(model, batch, 'cpu')
                singles = torch.cat([forward_batch(model, collate_graphs([graphs[i]]), 'cpu') for i in (0, 1)])
                torch.testing.assert_close(before, singles)
                altered = dict(batch, sample=batch['sample'].clone())
                altered['sample'][3:] += 50
                after = forward_batch(model, altered, 'cpu')
                torch.testing.assert_close(before[:3], after[:3])
                self.assertGreater((after[3:] - before[3:]).abs().max().item(), 1e-6)
                permutation = torch.tensor([2, 0, 1, 5, 3, 4])
                permuted = dict(batch, sample=batch['sample'][permutation], positions=batch['positions'][permutation],
                                time_mask=batch['time_mask'][permutation],
                                edge_index=torch.argsort(permutation)[batch['edge_index']])
                torch.testing.assert_close(before[permutation], forward_batch(model, permuted, 'cpu'))
                if kind == '3d_gnn_lstm':
                    invalid = dict(batch, edge_index=torch.tensor([[0], [3]]))
                    with self.assertRaisesRegex(ValueError, 'boundaries'):
                        forward_batch(model, invalid, 'cpu')

    def test_gradients_through_cnn_gnn_lstm_and_optional_paths(self):
        graphs = YieldSATDataset(synthetic_dataset(), band=None)
        batch = collate_graphs([graphs[0], graphs[1]])
        for kind, residual, aggregation in [('3d_lstm', False, 'mean'),
                                           ('3d_gnn_lstm', False, 'mean'),
                                           ('3d_gnn_lstm', True, 'directional')]:
            with self.subTest(model=kind, residual=residual):
                model = GNN_LSTM_Model(2, model_type=kind, cnn_hidden_dim=4, hidden_dim=4,
                    gnn_hidden_dim=3, dropout=0, residual=residual, aggregation=aggregation,
                    temporal_pool='attention', attention_mask=True)
                prediction, weights = forward_batch(model, batch, 'cpu', return_attention=True)
                (prediction - batch['target']).square().mean().backward()
                torch.testing.assert_close(weights.sum(1), torch.ones(6))
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for module in (model.cnn.conv, model.lstm, model.fc, *model.gnn):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_missing_bins_singleton_and_invalid_metadata(self):
        encoder = SpatialCNN3D(2, 3).eval()
        x = torch.randn(2, 3, 2)
        positions = torch.tensor([[1, 1], [1, 2]])
        mask = torch.tensor([[True, False, True], [True, True, True]])
        before = encoder(x, positions, torch.tensor([0, 2]), mask)
        x[0, 1] = 100000
        torch.testing.assert_close(before, encoder(x, positions, torch.tensor([0, 2]), mask))
        with self.assertRaisesRegex(ValueError, 'positions'):
            encoder(x, None, torch.tensor([0, 2]))
        for ptr in (None, torch.tensor([0, 1]), torch.tensor([0, 2, 2]), torch.tensor([0., 2.])):
            with self.assertRaisesRegex(ValueError, 'graph_ptr'):
                encoder(x, positions, ptr)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            encoder(x, positions[:1].expand(2, 2), torch.tensor([0, 2]))
        with self.assertRaisesRegex(ValueError, 'cnn_max_grid_cells'):
            encoder(x, torch.tensor([[0, 0], [1000, 1000]]), torch.tensor([0, 2]))
        with self.assertRaisesRegex(ValueError, 'time_mask'):
            encoder(x, positions, torch.tensor([0, 2]), mask[:, :1])
        model = GNN_LSTM_Model(2, model_type='3d_gnn_lstm', cnn_hidden_dim=3, hidden_dim=3)
        output = model(torch.ones(1, 1, 2), torch.empty(2, 0, dtype=torch.long),
                       positions=torch.tensor([[0, 0]]), graph_ptr=torch.tensor([0, 1]))
        output.sum().backward()
        self.assertTrue(torch.isfinite(output).all())

    def test_cli_validation_and_legacy_checkpoint_defaults(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (['--cnn-kernel-size', '2'], ['--cnn-temporal-kernel', '0'],
                          ['--cnn-hidden-dim', '0'], ['--cnn-max-grid-cells', '0'],
                          ['--model', '3d_lstm', '--residual'],
                          ['--model', '3d_lstm', '--aggregation', 'directional']):
                with self.assertRaises(SystemExit):
                    parse_args(flags)
        for kind in ('gnn_lstm', 'lstm'):
            legacy_config = dict(input_dim=2, hidden_dim=4, model_type=kind)
            original = GNN_LSTM_Model(**legacy_config).eval()
            restored = GNN_LSTM_Model(**legacy_config).eval()
            restored.load_state_dict(original.state_dict(), strict=True)
            self.assertIsNone(restored.cnn)
            x = torch.randn(2, 3, 2)
            edges = build_grid_graph([0, 0], [0, 1])
            torch.testing.assert_close(original(x, edges), restored(x, edges))

    def test_two_fold_training_exports_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / 'synthetic.nc'
            ds = synthetic_dataset()
            ds['crop'] = ('index', ['rapeseed'] * 24)
            ds['sample'].values[0, 0] = np.nan
            ds.to_netcdf(source)
            selected = load_dataset(source, bands=['B01', 'B02'])
            for kind in ('3d_gnn_lstm', '3d_lstm'):
                out = root / kind
                args = parse_args(['--data', str(source), '--bands', 'B01', 'B02', '--folds', '2',
                    '--epochs', '2', '--batch-size', '2', '--hidden-dim', '3', '--gnn-hidden-dim', '3',
                    '--cnn-hidden-dim', '4', '--cnn-temporal-kernel', '3', '--dropout', '0',
                    '--model', kind, '--normalize', '--append-mask', '--inner-val-fraction', '0.5',
                    '--early-stopping-patience', '1', '--export-training-predictions',
                    '--device', 'cpu', '--output-dir', str(out)])
                results = run_cross_validation(args)
                self.assertEqual(len(results), 24)
                self.assertFalse(results['index'].duplicated().any())
                self.assertTrue(np.isfinite(results.prediction).all())
                model, transform, checkpoint = load_checkpoint(out / 'fold_0.pt')
                self.assertEqual(checkpoint['model_config']['cnn_temporal_kernel'], 3)
                self.assertEqual(checkpoint['selection'], 'inner_mse')
                history = pd.read_csv(out / 'history.csv')
                self.assertTrue(np.isfinite(history.inner_val_mse).all())
                self.assertEqual(checkpoint['epoch'], int(history[history.cv_fold == 0]
                                 .sort_values('inner_val_mse').iloc[0].epoch))
                val = YieldSATDataset(selected, indices=checkpoint['val_indices'], band=None, transform=transform)
                frame, _ = model_validation(model, DataLoader(val, batch_size=2, collate_fn=collate_graphs),
                                             torch.nn.MSELoss(), 'cpu')
                np.testing.assert_allclose(frame.prediction, results[results.cv_fold == 0].prediction, rtol=1e-6)
                training = pd.read_csv(out / 'fold_0_training_predictions.csv')
                self.assertEqual(set(training['index']), set(checkpoint['train_indices']))
                self.assertFalse(set(training['index']) & set(checkpoint['inner_indices'] + checkpoint['val_indices']))


if __name__ == '__main__':
    unittest.main()
