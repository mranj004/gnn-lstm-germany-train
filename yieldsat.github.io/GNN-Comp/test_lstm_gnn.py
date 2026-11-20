"""Temporal-first message passing, spatial corrections, and checkpoint integration."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import YieldSATDataset, load_dataset
from graphs import collate_graphs
from model import GNN_LSTM_Model
from test_gnn_lstm import synthetic_dataset
from train import forward_batch, load_checkpoint, model_validation, parse_args, run_cross_validation


class LSTMGNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_initial_prediction_matches_plain_lstm_for_both_pooling_modes(self):
        x = torch.randn(4, 5, 2)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([[True, False, True, True, False]] * 3 + [[False] * 5])
        for pooling in ('last', 'attention'):
            options = dict(hidden_dim=4, dropout=0, temporal_pool=pooling,
                           attention_mask=pooling == 'attention')
            plain = GNN_LSTM_Model(2, model_type='lstm', **options).eval()
            spatial = GNN_LSTM_Model(2, model_type='lstm_gnn', **options).eval()
            spatial.lstm.load_state_dict(plain.lstm.state_dict())
            spatial.fc.load_state_dict(plain.fc.state_dict())
            if plain.attention is not None:
                spatial.attention.load_state_dict(plain.attention.state_dict())
            expected, weights = plain(x, edges, time_mask=mask, return_attention=True)
            actual, actual_weights = spatial(x, edges, time_mask=mask, return_attention=True)
            torch.testing.assert_close(actual, expected)
            if weights is not None:
                torch.testing.assert_close(actual_weights, weights)
                torch.testing.assert_close(weights[-1], torch.zeros(5))

    def test_neighbor_effect_is_after_lstm_and_limited_to_one_hop(self):
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=3,
                               model_type='lstm_gnn', dropout=0).eval()
        # Activate the correction with positive, deterministic graph weights.
        with torch.no_grad():
            model.gnn[0].self_linear.weight.fill_(0.1)
            model.gnn[0].self_linear.bias.fill_(2)
            model.gnn[0].neighbor_linear.weight.fill_(0.3)
            model.spatial_fc.weight.fill_(1)
        x = torch.randn(4, 5, 2)
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        ptr = torch.tensor([0, 3, 4])
        lstm_outputs = []
        hook = model.lstm.register_forward_hook(lambda module, args, output: lstm_outputs.append(output[0].detach()))
        before = model(x, edges, graph_ptr=ptr)
        changed = x.clone()
        changed[0, 0] += 5  # Change early-season data at one pixel only.
        after = model(changed, edges, graph_ptr=ptr)
        hook.remove()
        torch.testing.assert_close(lstm_outputs[0][1:], lstm_outputs[1][1:])
        self.assertGreater(abs((after[1] - before[1]).item()), 1e-7)
        torch.testing.assert_close(after[2:], before[2:])  # two hops away and separate field
        permutation = torch.tensor([2, 0, 1, 3])
        torch.testing.assert_close(before[permutation], model(x[permutation],
            torch.argsort(permutation)[edges], graph_ptr=ptr))
        singles = torch.cat([model(x[:3], edges), model(x[3:], torch.empty(2, 0, dtype=torch.long))])
        torch.testing.assert_close(before, singles)
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            model(x, torch.tensor([[0], [3]]), graph_ptr=ptr)
        for invalid in (torch.tensor([0, 3]), torch.tensor([0, 4, 4]), torch.tensor([0., 4.])):
            with self.assertRaisesRegex(ValueError, 'graph_ptr'):
                model(x, edges, graph_ptr=invalid)

    def test_correction_learns_and_gradients_reach_temporal_and_graph_layers(self):
        graphs = YieldSATDataset(synthetic_dataset(), band=None)
        batch = collate_graphs([graphs[0], graphs[1]])
        for aggregation in ('mean', 'directional'):
            model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=3, gnn_layers=2,
                model_type='lstm_gnn', dropout=0, aggregation=aggregation,
                temporal_pool='attention', attention_mask=True)
            # Keep ReLUs active so this test exercises all layers deterministically.
            with torch.no_grad():
                for layer in model.gnn:
                    layer.self_linear.bias.fill_(2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            (forward_batch(model, batch, 'cpu') - batch['target']).square().mean().backward()
            self.assertGreater(model.spatial_fc.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            optimizer.zero_grad()
            (forward_batch(model, batch, 'cpu') - batch['target']).square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for module in (model.lstm, model.fc, model.attention, model.spatial_fc, *model.gnn):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_cli_defaults_and_residual_validation(self):
        self.assertEqual(parse_args(['--model', 'lstm_gnn']).gnn_layers, 1)
        self.assertEqual(parse_args([]).gnn_layers, 2)
        self.assertEqual(parse_args(['--model', 'lstm_gnn', '--gnn-layers', '3']).gnn_layers, 3)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (['--model', 'lstm_gnn', '--residual'], ['--gnn-layers', '0']):
                with self.assertRaises(SystemExit):
                    parse_args(flags)
        with self.assertRaisesRegex(ValueError, 'additive'):
            GNN_LSTM_Model(2, model_type='lstm_gnn', residual=True)

    def test_two_fold_training_checkpoint_and_exports(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, out = root / 'synthetic.nc', root / 'run'
            ds = synthetic_dataset()
            ds['crop'] = ('index', ['rapeseed'] * 24)
            ds['sample'].values[0, 0] = np.nan
            ds.to_netcdf(source, engine='scipy')
            args = parse_args(['--data', str(source), '--bands', 'B01', 'B02', '--folds', '2',
                '--epochs', '3', '--batch-size', '2', '--hidden-dim', '4', '--gnn-hidden-dim', '3',
                '--dropout', '0', '--model', 'lstm_gnn', '--normalize', '--append-mask',
                '--inner-val-fraction', '0.5', '--early-stopping-patience', '1',
                '--temporal-pool', 'attention', '--attention-mask', '--export-attention',
                '--export-training-predictions', '--device', 'cpu', '--output-dir', str(out)])
            results = run_cross_validation(args)
            self.assertEqual(len(results), 24)
            self.assertFalse(results['index'].duplicated().any())
            self.assertTrue(np.isfinite(results.prediction).all())
            selected = load_dataset(source, bands=['B01', 'B02'])
            model, transform, checkpoint = load_checkpoint(out / 'fold_0.pt')
            self.assertEqual(checkpoint['model_config']['model_type'], 'lstm_gnn')
            self.assertEqual(checkpoint['model_config']['gnn_layers'], 1)
            self.assertEqual(checkpoint['selection'], 'inner_mse')
            self.assertGreater(sum(p.abs().sum().item() for p in model.spatial_fc.parameters()), 0)
            history = pd.read_csv(out / 'history.csv')
            self.assertEqual(checkpoint['epoch'], int(history[history.cv_fold == 0]
                             .sort_values('inner_val_mse').iloc[0].epoch))
            val = YieldSATDataset(selected, indices=checkpoint['val_indices'], band=None, transform=transform)
            frame, _ = model_validation(model, DataLoader(val, batch_size=2, collate_fn=collate_graphs),
                                         torch.nn.MSELoss(), 'cpu', export_attention=True)
            np.testing.assert_allclose(frame.prediction, results[results.cv_fold == 0].prediction, rtol=1e-6)
            np.testing.assert_allclose(frame.filter(like='attention_').sum(axis=1), 1, atol=1e-6)
            training = pd.read_csv(out / 'fold_0_training_predictions.csv')
            self.assertEqual(set(training['index']), set(checkpoint['train_indices']))
            self.assertFalse(set(training['index']) & set(checkpoint['inner_indices'] + checkpoint['val_indices']))


if __name__ == '__main__':
    unittest.main()
