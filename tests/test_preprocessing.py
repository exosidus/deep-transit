import unittest
from unittest.mock import patch

import numpy as np

from preprocessing.pipeline import (
    PreprocessingError,
    median_bin,
    phase_fold,
    preprocess_lightcurve,
)


class FakeLightCurve:
    def __init__(self, time, flux):
        self.time = np.asarray(time)
        self.flux = np.asarray(flux)
        self.quality = np.zeros(len(self.time), dtype=np.int32)


class PreprocessingTests(unittest.TestCase):
    def test_phase_fold_centers_catalog_epoch(self):
        phase, _ = phase_fold(np.array([12.5, 13.0, 13.5]), np.ones(3), 2.0, 13.0)
        self.assertAlmostEqual(phase[1], 0.0)
        self.assertTrue(np.all(phase >= -0.5))
        self.assertTrue(np.all(phase < 0.5))

    def test_median_bins_fill_gaps_and_standardize(self):
        phase = np.array([-0.45, -0.25, 0.1, 0.3])
        flux = np.array([1.0, 0.8, 0.9, 1.2])
        centers, standardized, occupancy = median_bin(phase, flux, bins=20)
        self.assertEqual(centers.shape, (20,))
        self.assertEqual(standardized.shape, (20,))
        self.assertEqual(occupancy.sum(), len(phase))
        self.assertTrue(np.isfinite(standardized).all())
        self.assertAlmostEqual(float(standardized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(standardized.std()), 1.0, places=6)

    def test_preprocessing_preserves_stages_and_model_shape(self):
        time = np.arange(0, 30, 0.005)
        period, epoch, duration_hours = 2.0, 0.5, 2.4
        phase = ((time - epoch + period / 2) % period) / period - 0.5
        slow_variability = 1.0 + 0.001 * np.sin(2 * np.pi * time / 20)
        transit = np.where(np.abs(phase) < 0.025, -0.01, 0.0)
        flux = 1000.0 * slow_variability * (1 + transit)
        lightcurve = FakeLightCurve(time, flux)
        with patch("preprocessing.pipeline.quality_mask_lightcurve", return_value=np.ones(len(time), bool)):
            result = preprocess_lightcurve(
                lightcurve,
                period_days=period,
                epoch_bkjd=epoch,
                duration_hours=duration_hours,
                minimum_valid_bins=50,
            )
        self.assertEqual(result.model_input.shape, (1, 1000))
        self.assertEqual(result.time.shape, result.raw_flux.shape)
        self.assertEqual(result.quality_mask.shape, result.time.shape)
        self.assertTrue(np.isfinite(result.model_input).all())
        self.assertEqual(result.detrending_window_samples % 2, 1)
        self.assertGreater(result.bin_occupancy.sum(), 0)

    def test_missing_ephemeris_is_rejected(self):
        lightcurve = FakeLightCurve(np.arange(10.0), np.ones(10))
        with patch("preprocessing.pipeline.quality_mask_lightcurve", return_value=np.ones(10, bool)):
            with self.assertRaisesRegex(PreprocessingError, "Known orbital period"):
                preprocess_lightcurve(
                    lightcurve,
                    period_days=None,
                    epoch_bkjd=None,
                    duration_hours=2.0,
                )


if __name__ == "__main__":
    unittest.main()
