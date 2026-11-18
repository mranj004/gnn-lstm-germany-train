"""Fold-local spectral transforms; missingness never enters fitted statistics."""

import numpy as np


class SpectralTransform:
    def __init__(self, normalize=False, append_mask=False, fill_value=-1.0,
                 missing_value=None, mean=None, scale=None, count=None):
        self.normalize = normalize
        self.append_mask = append_mask
        self.fill_value = fill_value
        self.missing_value = missing_value
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.count = count

    def observed(self, x):
        valid = np.isfinite(x)
        if self.missing_value is not None:
            valid &= x != self.missing_value
        return valid

    def fit(self, samples, positions, chunk_size=4096):
        """Population moments over training pixels/time only, in bounded chunks."""
        n = np.zeros(samples.shape[-1], dtype=np.int64)
        mean = np.zeros_like(n, dtype=np.float64)
        m2 = np.zeros_like(mean)
        for start in range(0, len(positions), chunk_size):
            x = samples[positions[start:start + chunk_size]].astype(np.float64)
            valid = self.observed(x)
            k = valid.sum(axis=(0, 1))
            batch_mean = np.where(valid, x, 0).sum(axis=(0, 1)) / np.maximum(k, 1)
            batch_m2 = np.square(np.where(valid, x - batch_mean, 0)).sum(axis=(0, 1))
            delta = batch_mean - mean
            total = n + k
            m2 += batch_m2 + delta ** 2 * n * k / np.maximum(total, 1)
            mean += delta * k / np.maximum(total, 1)
            n = total
        scale = np.sqrt(m2 / np.maximum(n, 1))
        self.mean = mean
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        self.count = n.tolist()
        return self

    def __call__(self, samples):
        valid = self.observed(samples)
        if self.normalize:
            if self.mean is None:
                raise ValueError("Fit normalization on training indices before transforming")
            x = (np.where(valid, samples, self.mean) - self.mean) / self.scale
            x = np.where(valid, x, 0).astype(np.float32)
        else:
            x = np.where(valid, samples, self.fill_value).astype(np.float32)
        if self.append_mask:
            x = np.concatenate([x, valid.astype(np.float32)], axis=-1)
        return x, valid.any(axis=-1)

    def state_dict(self):
        return dict(normalize=self.normalize, append_mask=self.append_mask,
                    fill_value=self.fill_value, missing_value=self.missing_value,
                    mean=None if self.mean is None else self.mean.tolist(),
                    scale=None if self.scale is None else self.scale.tolist(), count=self.count)
