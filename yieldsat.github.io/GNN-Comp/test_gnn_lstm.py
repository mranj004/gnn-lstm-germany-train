"""Run with: python -m unittest discover -s yieldsat.github.io/GNN-Comp -v."""

from pathlib import Path
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xarray as xr

from data import YieldSATDataset, filter_ds_by_attribute, load_dataset
from eval import evaluate_predictions
from graphs import build_grid_graph, collate_graphs
from model import GNN_LSTM_Model, MessagePassingGNN, mean_neighbor_adjacency
from prepareFold import split_dataset_by_group
from train import model_training, model_validation, parse_args, run_cross_validation


def synthetic_dataset():
    rng = np.random.default_rng(7)
    n = 24
    return xr.Dataset(
        {
            "sample": (("index", "time_step", "band"), rng.normal(size=(n, 4, 2)).astype("float32")),
            "target": ("index", np.linspace(1, 4, n, dtype="float32")),
            "field_shared_name": ("index", np.repeat(np.arange(4), 6)),
            "year": ("index", np.tile(np.repeat([2020, 2021], 3), 4)),
            "row": ("index", np.zeros(n, dtype="uint8")),
            "col": ("index", np.tile([0, 1, 2], 8).astype("uint8")),
            "crop": ("index", np.repeat([1, 1, 1, 2], 6)),
            "country": ("index", np.repeat("Germany", n)),
        },
        coords={"index": [f"p{i}" for i in range(n)], "time_step": np.arange(4), "band": ["B01", "B02"]},
    ).assign_attrs(description="Synthetic test data")


class GNNLSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5)

    def test_relative_paths_are_independent_of_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = parse_args([])
                self.assertEqual(args.data, Path(__file__).resolve().parent / "data")
                self.assertEqual(args.output_dir, Path(__file__).resolve().parent / "runs/germany_rapeseed_s2")
                self.assertEqual((args.country, args.crop), ("Germany", "rapeseed"))
            finally:
                os.chdir(original)

    def test_raw_s2_filtering_temporal_bins_and_training(self):
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for country, crop, field, year in [
                ("Germany", "rapeseed", "field1", 2020),
                ("Germany", "rapeseed", "field1", 2021),
                ("Germany", "rapeseed", "field2", 2020),
                ("Germany", "wheat", "field3", 2020),
                ("France", "rapeseed", "field4", 2020),
            ]:
                name = f"{country}_farm1_{field}_{crop}_{year}"
                folder = root / name
                folder.mkdir()
                (folder / f"metadata-{name}.json").write_text(json.dumps({
                    "adm_units": {"country": country}, "crop": crop, "year": year,
                    "field_shared_name": name,
                    "seeding_date": f"01.01.{year}", "harvesting_date": f"09.01.{year}",
                }))
                # Excluded crops/countries intentionally have no raster files.
                if country != "Germany" or crop != "rapeseed":
                    continue
                (folder / "s2_images").mkdir()
                (folder / "scl_masks").mkdir()
                (folder / "yield_masks").mkdir()
                profile = dict(driver="GTiff", height=2, width=2, count=1,
                               dtype="float32", crs="EPSG:32633",
                               transform=from_origin(400000, 5900000, 10, 10), nodata=-1)
                with rasterio.open(folder / "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif", "w", **profile) as dst:
                    dst.write(np.array([[1, 2], [3, -1]], dtype="float32"), 1)
                for date, value in [(f"{year}0101", 2), (f"{year}0102", 4),
                                    (f"{year}0109", 8), (f"{year}0110", 100)]:
                    with rasterio.open(folder / f"s2_images/S2_L2A_{date}.tif", "w", **profile) as dst:
                        values = np.full((2, 2), value, dtype="float32")
                        values[1, 0] = -1
                        dst.write(values, 1)
                        dst.set_band_description(1, "B02 - Blue")
                    with rasterio.open(folder / f"scl_masks/S2_L2A_SCL_{date}.tif", "w", **profile) as dst:
                        scl = np.full((2, 2), 4, dtype="float32")
                        scl[0, 0] = 9
                        dst.write(scl, 1)
            ds = load_dataset(root, country="Germany", bands=["B02"], time_steps=2)
            self.assertEqual(dict(ds.sizes), {"index": 9, "time_step": 2, "band": 1})
            np.testing.assert_allclose(ds["sample"].values[0, :, 0], [3, 8])
            self.assertTrue(np.isnan(ds["sample"].values[2]).all())
            masked = load_dataset(root, country="Germany", bands=["B02"], time_steps=2, cloud_mask=True)
            self.assertTrue(np.isnan(masked['sample'].values[0]).all())
            np.testing.assert_allclose(masked['sample'].values[1], ds['sample'].values[1])
            np.testing.assert_array_equal(masked.target, ds.target)
            self.assertEqual(len(np.unique(ds.field_shared_name)), 2)
            self.assertEqual(len(YieldSATDataset(ds, band=None)), 3)
            for training, validation in split_dataset_by_group(ds, 2):
                self.assertTrue(set(ds.sel(index=training).field_shared_name.values).isdisjoint(
                    ds.sel(index=validation).field_shared_name.values))
            args = parse_args([
                "--data", str(root), "--bands", "B02", "--time-steps", "2",
                "--folds", "2", "--epochs", "1", "--hidden-dim", "4",
                "--gnn-hidden-dim", "4", "--device", "cpu", "--output-dir", str(root / "run"),
            ])
            predictions = run_cross_validation(args)
            self.assertEqual(len(predictions), 9)
            self.assertTrue(np.isfinite(predictions.prediction).all())

    def test_grid_topology_and_unsigned_coordinates(self):
        edges = build_grid_graph(np.array([0, 0, 1], dtype="uint8"), [0, 1, 1], 4)
        self.assertEqual(set(map(tuple, edges.t().tolist())), {(0, 1), (1, 0), (1, 2), (2, 1)})
        self.assertEqual(build_grid_graph([0, 0, 1], [0, 1, 1], 8).shape, (2, 6))
        self.assertEqual(build_grid_graph([0], [0]).shape, (2, 0))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_grid_graph([0, 0], [0, 0])
        with self.assertRaisesRegex(ValueError, "finite integer"):
            build_grid_graph([0.5], [0])

    def test_message_passing_uses_separate_self_and_mean_neighbor_weights(self):
        # Directed edges, unequal degrees, a duplicate, a self-loop, and an isolate.
        edges = torch.tensor([[0, 1, 2, 0, 1], [1, 0, 1, 1, 1]])
        adjacency = mean_neighbor_adjacency(edges, 4, torch.float32)
        layer = MessagePassingGNN(1, 1)
        with torch.no_grad():
            layer.self_linear.weight.fill_(2)
            layer.self_linear.bias.fill_(1)
            layer.neighbor_linear.weight.fill_(3)
        samples = torch.tensor([[[2.], [4.]], [[6.], [8.]], [[10.], [12.]], [[14.], [16.]]])
        output = layer(samples, adjacency)
        expected = torch.tensor([[[23.], [33.]], [[31.], [41.]], [[21.], [25.]], [[29.], [33.]]])
        torch.testing.assert_close(output, expected)

    def test_empty_neighbors_preserve_self_update(self):
        layer = MessagePassingGNN(2, 3)
        samples = torch.randn(2, 4, 2)
        for edges in (torch.empty(2, 0, dtype=torch.long), torch.tensor([[0, 1], [0, 1]])):
            with self.subTest(edges=edges):
                adjacency = mean_neighbor_adjacency(edges, 2, samples.dtype)
                torch.testing.assert_close(layer(samples, adjacency), layer.self_linear(samples))

    def test_batching_isolates_graphs_and_neighbors_affect_predictions(self):
        graphs = YieldSATDataset(synthetic_dataset(), band=None)
        batch = collate_graphs([graphs[0], graphs[1]])
        src, dst = batch["edge_index"]
        self.assertTrue(torch.equal(src < 3, dst < 3))
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=4, dropout=0).eval()
        # Positive parameters make influence through ReLU deterministic.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.1)
            samples = torch.ones(6, 4, 2)
            before = model(samples, batch["edge_index"])
            samples[1] += 3
            after = model(samples, batch["edge_index"])
            self.assertGreater(abs((after[0] - before[0]).item()), 1e-6)
            torch.testing.assert_close(before[3:], after[3:])
            single = model(samples[:3], graphs[0]["edge_index"])
            torch.testing.assert_close(single, after[:3])

    def test_node_order_equivariance(self):
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=4, dropout=0).eval()
        x = torch.randn(3, 4, 2)
        edges = build_grid_graph([0, 0, 0], [0, 1, 2])
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(model(x, edges)[permutation], model(x[permutation], inverse[edges]))

    def test_gradients_reach_gnn_lstm_and_readout(self):
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=4, dropout=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.1)
        x = torch.ones(3, 4, 2)
        loss = (model(x, build_grid_graph([0, 0, 0], [0, 1, 2])) - 2).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_isolated_pixel(self):
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=4)
        out = model(torch.randn(1, 4, 2), torch.empty(2, 0, dtype=torch.long))
        self.assertEqual(out.shape, (1,))
        self.assertTrue(torch.isfinite(out).all())

    def test_group_splits_and_graph_metadata(self):
        ds = synthetic_dataset()
        seen = []
        for train, val in split_dataset_by_group(ds, 2):
            self.assertTrue(set(ds.sel(index=train).field_shared_name.values).isdisjoint(
                ds.sel(index=val).field_shared_name.values))
            seen.extend(val)
            graphs = YieldSATDataset(ds, indices=train, band=None)
            for graph in graphs:
                self.assertEqual(len(set(graph["field_shared_name"])), 1)
                self.assertEqual(len(set(graph["year"])), 1)
        self.assertCountEqual(seen, ds.index.values)
        self.assertEqual(len(YieldSATDataset(ds, band=None)), 8)

    def test_dimension_order_missing_values_and_crop_filter(self):
        ds = synthetic_dataset()
        ds["crop"].attrs = {"1": "rapeseed", "2": "wheat"}
        filtered = filter_ds_by_attribute(ds, condition="rapeseed")
        self.assertEqual(filtered.sizes["index"], 18)
        self.assertEqual(ds["crop"].attrs, {"1": "rapeseed", "2": "wheat"})
        ds["sample"].values[0, 0, 0] = np.nan
        ds["sample"].values[1, 0, 0] = np.inf
        ds["sample"] = ds["sample"].transpose("band", "index", "time_step")
        graph = YieldSATDataset(ds, band=["B01"], fill_value=-7)[0]
        self.assertEqual(graph["sample"].shape, (3, 4, 1))
        self.assertTrue(torch.isfinite(graph["sample"]).all())
        self.assertEqual(graph["sample"][0, 0, 0].item(), -7)

    def test_training_validation_modes_and_pixel_weighted_loss(self):
        ds = synthetic_dataset().isel(index=list(range(3)) + [6])
        graphs = YieldSATDataset(ds, band=None)
        loader = DataLoader(graphs, batch_size=1, collate_fn=collate_graphs)
        model = GNN_LSTM_Model(2, hidden_dim=4, gnn_hidden_dim=4, dropout=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        expected_mse = float(np.mean(ds.target.values ** 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        stats = {}
        actual = model_training(model, loader, torch.nn.MSELoss(), optimizer, "cpu", stats=stats)
        self.assertTrue(model.training)
        self.assertAlmostEqual(actual, expected_mse, places=5)
        self.assertAlmostEqual(stats['train_rmse'], np.sqrt(expected_mse), places=5)
        targets64 = ds.target.values.astype(np.float64)
        expected_r2 = 1 - np.sum(targets64 ** 2) / np.sum((targets64 - targets64.mean()) ** 2)
        self.assertAlmostEqual(stats['train_r2'], expected_r2, places=4)
        self.assertEqual(stats['train_pixels'], 4)
        frame, _ = model_validation(model, loader, torch.nn.MSELoss(), "cpu")
        self.assertFalse(model.training)
        self.assertAlmostEqual(frame.attrs["mse"], expected_mse, places=5)

    def test_evaluation_keeps_years_separate_and_rejects_epoch_duplicates(self):
        frame = pd.DataFrame({
            "index": ["a", "b", "c"], "field_shared_name": [1, 1, 1],
            "year": [2020, 2020, 2021], "target": [1., 3., 5.], "prediction": [2., 2., 5.],
        })
        metrics, fields = evaluate_predictions(frame)
        self.assertEqual(len(fields), 2)
        self.assertEqual(metrics["field_year"]["rmse"], 0)
        with self.assertRaisesRegex(ValueError, "one out-of-fold"):
            evaluate_predictions(pd.concat([frame, frame]))

    def test_end_to_end_netcdf_training_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "synthetic.nc"
            ds = synthetic_dataset()
            ds["crop"].attrs = {"1": "rapeseed", "2": "wheat"}
            ds.to_netcdf(source)
            selected = load_dataset(source, bands=["B01", "B02"])
            args = parse_args([
                "--data", str(source), "--bands", "B01", "B02", "--folds", "2",
                "--epochs", "2", "--hidden-dim", "4", "--gnn-hidden-dim", "4",
                "--batch-size", "2", "--dropout", "0", "--device", "cpu",
                "--output-dir", str(path / "run"),
            ])
            results = run_cross_validation(args)
            self.assertEqual(len(results), 18)
            self.assertFalse(results["index"].duplicated().any())
            for name in ("metrics.json", "history.csv", "cv_predictions.csv", "fold_0.pt", "fold_1.pt"):
                self.assertTrue((path / "run" / name).exists(), name)
            checkpoint = torch.load(path / "run/fold_0.pt", weights_only=True)
            model = GNN_LSTM_Model(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            val = YieldSATDataset(selected, indices=checkpoint["val_indices"], band=None)
            loader = DataLoader(val, batch_size=2, collate_fn=collate_graphs)
            frame, _ = model_validation(model, loader, torch.nn.MSELoss(), "cpu")
            saved = results[results.cv_fold == 0]
            np.testing.assert_allclose(frame.prediction, saved.prediction, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
