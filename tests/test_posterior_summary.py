"""Distance priors and posterior summaries must preserve their stated coordinates."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad

from Speculate_addons.distance_scale import (
    distance_prior_to_log_scale_prior, distance_to_log_scale, log_scale_to_distance_pc,
)
from Speculate_addons.gp_covariance import bounds_for_frozen_prior
from Speculate_addons.Spec_functions import (
    posterior_arviz_summary, posterior_corner, posterior_summary_rows, summarize_posterior,
)
from Speculate_addons import speculate_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app._maybe_initialize()
    return module


def defining(module, name):
    return next(cell for cell in module.app._cell_manager.cells() if name in cell.defs)


class Model:
    """Small model with Starfish's frozen-key assignment behaviour."""
    def __init__(self):
        self.params = {"param1": 2., "log_scale": 0., "Av": .3}
        self.frozen = ["Av"]
        self.emulator = SimpleNamespace(param_names=["param1"], min_params=np.array([0.]), max_params=np.array([20.]))
        self.data = SimpleNamespace(flux=np.array([2., 4., 6.]), sigma=np.ones(3), wave=np.array([1., 2., 3.]))
        self.data_name = "skewed spectrum"
        self.flux_scale = "linear"

    @property
    def labels(self):
        return tuple(key for key in self.params if key not in self.frozen)

    @property
    def grid_params(self):
        return np.array([self.params["param1"]])

    def get_param_dict(self, flat=True):
        return {key: self.params[key] for key in self.labels}

    def set_param_dict(self, values):
        for key, value in values.items():
            if key not in self.frozen:
                self.params[key] = float(value)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = float(value)

    def __call__(self):
        return np.asarray(spectrum(self)["model_flux"]), np.eye(3)

    def thaw(self, key):
        if key in self.frozen:
            self.frozen.remove(key)

    def freeze(self, key):
        if key not in self.frozen:
            self.frozen.append(key)


def spectrum(model):
    return {
        "wavelength": [1., 2., 3.], "data_flux": model.data.flux.tolist(),
        "model_flux": (np.arange(1., 4.) * model.params["param1"] * np.exp(model.params["log_scale"])).tolist(),
        "model_cov_diag": [1., 1., 1.],
        "covariance_components": {"observation_variance": [1., 1., 1.]},
    }


class DistancePriorTests(unittest.TestCase):
    def test_exact_density_and_quantiles_recover_positive_normal_distance(self):
        # Cover the observed CV prior, a broad prior, and appreciable truncation.
        for mean, sigma in ((90.36, .15), (100., 45.), (10., 20.)):
            with self.subTest(mean=mean, sigma=sigma):
                reference = 250.
                prior = distance_prior_to_log_scale_prior(mean, sigma, reference)
                distance_prior = stats.truncnorm(-mean / sigma, np.inf, loc=mean, scale=sigma)
                q = np.array([.01, .16, .5, .84, .99])
                distances = distance_prior.ppf(q)
                s = distance_to_log_scale(distances, reference)
                # Omitting the Jacobian fails this density-in-parsecs check.
                np.testing.assert_allclose(
                    prior.logpdf(s) - np.log(distances / 2.),
                    distance_prior.logpdf(distances), atol=1e-10,
                )
                np.testing.assert_allclose(prior.cdf(s), 1. - q, atol=1e-10)
                np.testing.assert_allclose(prior.ppf(1. - q), s, atol=1e-10)
                lo, hi = prior.ppf([1e-8, 1. - 1e-8])
                self.assertAlmostEqual(quad(prior.pdf, lo, hi)[0], 1. - 2e-8, places=8)
                self.assertEqual(prior.interval(1.), (-np.inf, np.inf))
                self.assertFalse(np.isnan(prior.logpdf([-1e6, 1e6])).any())

    def test_tier3_bounds_metadata_and_initial_value_use_exact_prior(self):
        for mean, sigma in ((90.36, .15), (100., 45.), (10., 20.)):
            metadata, priors, start = benchmark._tier3_distance_prior_to_log_scale(
                {"mean_pc": mean, "sigma_pc": sigma},
            )
            prior = priors["log_scale"]
            bounds = bounds_for_frozen_prior("log_scale", prior)
            self.assertTrue(np.isfinite(bounds).all())
            np.testing.assert_allclose(prior.cdf(bounds), stats.norm.cdf([-4., 4.]), atol=1e-10)
            self.assertAlmostEqual(log_scale_to_distance_pc(start["log_scale"]), mean)
            np.testing.assert_allclose(
                [metadata["plot_min_pc"], metadata["plot_max_pc"]],
                log_scale_to_distance_pc(np.array(bounds)[::-1]),
            )
            self.assertEqual(metadata["transform"], "exact_positive_distance")
            self.assertNotIn("log_scale_sigma", metadata)
            json.dumps(metadata)
        # Existing priors retain their previous optimizer bounds.
        self.assertEqual(bounds_for_frozen_prior("param1", stats.uniform(2., 3.)), (2., 5.))
        self.assertEqual(bounds_for_frozen_prior("param1", stats.norm(2., 3.)), (-10., 14.))

    def test_distance_prior_rejects_invalid_parameters(self):
        for invalid in (0., -1., np.inf, np.nan):
            for args in ((invalid, 1., 100.), (100., invalid, 100.), (100., 1., invalid)):
                with self.assertRaises(ValueError):
                    distance_prior_to_log_scale_prior(*args)


class PosteriorTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.array([[1., -2.], [1., 0.], [10., 2.]])
        self.labels = ["Physical axis", "log_scale"]
        self.internal = ["param1", "log_scale"]
        self.model = Model()
        self.addCleanup(plt.close, "all")

    def cached(self):
        with patch.object(benchmark, "_model_bestfit_spectrum", side_effect=spectrum), patch.object(
            benchmark, "_model_likelihood_diagnostics", return_value={"nll": 3.}
        ), patch.object(benchmark, "_extract_spectrum_nuisance_transforms", side_effect=lambda model: {
            "log_scale": model.params["log_scale"], "Av": model.params["Av"],
        }):
            return benchmark.evaluate_posterior_summaries(self.model, {}, self.samples, self.labels, self.internal)

    def test_skewed_summary_distance_and_frozen_point_share_coordinates(self):
        original = self.samples.copy()
        result = summarize_posterior(self.samples, self.labels, self.internal, {"Av": .3})
        expected_distance = np.mean([100 * np.e, 100., 100 / np.e])
        self.assertAlmostEqual(result["summary"]["Distance (pc)"]["mean"], expected_distance)
        for method, physical in (("mean", 4.), ("median", 1.)):
            point = result["point_estimates"][method]
            self.assertEqual(point["param1"], physical)
            self.assertEqual(point["Av"], .3)
            self.assertAlmostEqual(log_scale_to_distance_pc(point["log_scale"]), result["summary"]["Distance (pc)"][method])
            rows = posterior_summary_rows(result["summary"], method, {"Physical axis": 2.})
            self.assertEqual(rows[0][method.title()], physical)
            self.assertAlmostEqual(rows[0][f"Δ/SD ({method})"], (physical - 2.) / np.std(original[:, 0]))
            self.assertLess(rows[0]["16th percentile"], rows[0]["84th percentile"])
        np.testing.assert_array_equal(self.samples, original)
        self.assertEqual(posterior_summary_rows(result["summary"])[0]["Median"], 1.)
        # A displayed distance column has no backend log_scale identity; it
        # must not be converted a second time on export/load.
        again = summarize_posterior(result["samples"], result["labels"])
        np.testing.assert_array_equal(again["samples"], result["samples"])

    def test_invalid_and_zero_width_samples(self):
        for bad in (np.array([]), np.array([[np.nan, 1.]])):
            with self.assertRaises(ValueError):
                summarize_posterior(bad, self.labels)
        constant = summarize_posterior(np.ones((5, 1)), ["Inclination (degrees)"])
        row = posterior_summary_rows(constant["summary"], "median", {"Inclination": 2.})[0]
        self.assertTrue(np.isnan(row["Δ/SD (median)"]))
        failed = benchmark.select_posterior_summary({"samples": [[np.nan]], "labels": ["Axis"]})
        self.assertEqual(failed["display_samples"].size, 0)
        self.assertIn("must be finite", failed["summary_notice"])

    def test_arviz_intervals_and_sd_do_not_change_with_selected_centre(self):
        import arviz as az
        data = az.from_dict(posterior={"Axis": np.random.default_rng(4).lognormal(size=(4, 100))})
        mean = posterior_arviz_summary(data, "mean")
        median = posterior_arviz_summary(data)
        self.assertNotEqual(mean.loc["Axis", "mean"], median.loc["Axis", "median"])
        self.assertIn("mcse_median", median.columns)
        for column in ("sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"):
            self.assertEqual(mean.loc["Axis", column], median.loc["Axis", column])

    def test_postprocessing_failure_keeps_chain_and_restores_model(self):
        chain = np.random.default_rng(4).normal(size=(20, 8, 2))
        sampler = SimpleNamespace(
            get_chain=lambda discard=0, thin=1: chain[discard::thin],
            get_autocorr_time=lambda **kwargs: np.full(2, np.nan),
            run_mcmc=lambda *a, **kw: self.model.__setitem__("log_scale", 7.),
            acceptance_fraction=np.full(8, .3),
        )
        with patch("emcee.EnsembleSampler", return_value=sampler), patch.object(
            benchmark, "evaluate_posterior_summaries", side_effect=OverflowError("display transform failed"),
        ):
            result = benchmark.run_mcmc_single(
                self.model, {}, nsteps=20, nwalkers=8, burnin=1, freeze_params={"Av": True},
            )
        np.testing.assert_array_equal(result["samples"], chain[1:].reshape((-1, 2)))
        self.assertEqual(result["posterior_models"], {})
        self.assertIn("display transform failed", result["summary_error"])
        self.assertEqual(self.model["log_scale"], 0.)
        self.assertEqual(self.model.frozen, ["Av"])

    def test_cached_models_use_selected_full_point_and_restore_model(self):
        original = dict(self.model.params)
        result = self.cached()
        self.assertEqual(self.model.params, original)
        self.assertEqual(self.model.frozen, ["Av"])
        for method in ("mean", "median"):
            point = result["posterior_models"][method]
            self.assertEqual(point["parameters"]["Av"], .3)
            self.assertEqual(point["grid_params"], [result["summary"]["Physical axis"][method]])
            self.assertAlmostEqual(point["spectrum"]["model_flux"][0], point["parameters"]["param1"] * np.exp(point["parameters"]["log_scale"]))
        self.assertNotEqual(result["posterior_models"]["mean"]["reduced_chi2"], result["posterior_models"]["median"]["reduced_chi2"])

    def test_sampler_defaults_to_median_and_keeps_explicit_mean_fields(self):
        chain = np.tile(self.samples, (7, 1))[:20, None, :].repeat(8, axis=1)
        initial_positions = []
        sampler = SimpleNamespace(
            get_chain=lambda discard=0, thin=1: chain[discard::thin],
            get_autocorr_time=lambda **kwargs: np.full(2, np.nan),
            run_mcmc=lambda ball, *a, **kw: initial_positions.append(np.array(ball)),
            acceptance_fraction=np.full(8, .3),
        )
        with patch("emcee.EnsembleSampler", return_value=sampler), patch.object(
            benchmark, "_model_bestfit_spectrum", side_effect=spectrum,
        ), patch.object(benchmark, "_model_likelihood_diagnostics", return_value={}), patch.object(
            benchmark, "_extract_spectrum_nuisance_transforms", return_value={},
        ):
            result = benchmark.run_mcmc_single(
                self.model, {"log_scale": distance_prior_to_log_scale_prior(100., 45.)},
                nsteps=20, nwalkers=8, burnin=1, freeze_params={"Av": True},
            )
        self.assertTrue(np.isfinite(initial_positions[0]).all())
        self.assertGreater(np.ptp(initial_positions[0][:, 1]), 0.)
        self.assertEqual(result["summary_statistic"], "median")
        self.assertEqual(result["bestfit_spec"], result["posterior_models"]["median"]["spectrum"])
        self.assertEqual(result["posterior_mean_bestfit_spec"], result["posterior_models"]["mean"]["spectrum"])
        self.assertNotEqual(result["bestfit_spec"], result["posterior_mean_bestfit_spec"])

    def test_one_failed_model_never_substitutes_the_other_method(self):
        def fail_mean(model):
            if model.params["param1"] > 2:
                raise ValueError("invalid mean point")
            return spectrum(model)
        with patch.object(benchmark, "_model_bestfit_spectrum", side_effect=fail_mean), patch.object(
            benchmark, "_model_likelihood_diagnostics", return_value={}
        ), patch.object(benchmark, "_extract_spectrum_nuisance_transforms", return_value={}):
            result = benchmark.evaluate_posterior_summaries(self.model, {}, self.samples, self.labels, self.internal)
        self.assertIn("error", result["posterior_models"]["mean"])
        self.assertNotIn("spectrum", result["posterior_models"]["mean"])
        self.assertIn("spectrum", result["posterior_models"]["median"])
        self.assertEqual(self.model.params["param1"], 2.)

    def test_frozen_grid_coordinate_is_in_both_export_vectors(self):
        self.model.freeze("param1")
        self.samples = self.samples[:, 1:]
        self.labels, self.internal = ["log_scale"], ["log_scale"]
        result = self.cached()
        for payload in result["posterior_models"].values():
            self.assertEqual(payload["grid_params"], [2.])
            self.assertEqual(payload["parameters"]["param1"], 2.)
        self.assertIn("param1", self.model.frozen)

    def test_corner_titles_and_markers_follow_choice_but_intervals_do_not(self):
        draws = np.random.default_rng(7).lognormal(size=(300, 1))
        for method, reducer in (("mean", np.mean), ("median", np.median)):
            fig = posterior_corner(draws, ["Axis"], method)
            axis = fig.axes[0]
            self.assertIn(f"{method}={reducer(draws):.4f}", axis.get_title())
            self.assertTrue(any(np.allclose(line.get_xdata(), reducer(draws)) for line in axis.lines))
            lo, hi = np.percentile(draws, [16, 84])
            self.assertIn(f"[{lo:.4f}, {hi:.4f}]", axis.get_title())

    def test_metrics_and_report_json_round_trip_keep_both_centres(self):
        draws = [np.array([1., 1., 10.]), np.array([2., 2., 8.])]
        truths = [2., 3.]
        records = [{"mcmc_converged": True, **{f"Physical axis_{k}": v for k, v in benchmark.posterior_parameter_metrics(d, t).items()}} for d, t in zip(draws, truths)]
        aggregate = benchmark.aggregate_tier2_results(
            records, {"Physical axis": draws}, {"Physical axis": truths},
            ["Physical axis"], 1, np.array([0.]), np.array([20.]), 2, 0, [], 8, 100, 10, 1.,
        )
        self.assertEqual(aggregate["aggregate"]["Physical axis"]["rmse"], 1.)
        self.assertEqual(aggregate["aggregate"]["Physical axis"]["bias"], -1.)
        for row in records:
            self.assertEqual(row["Physical axis_delta_sigma"], row["Physical axis_delta_sigma_median"])
        cached = self.cached()
        post = {
            "run": 0, "samples": self.samples, "labels": self.labels,
            "internal_labels": self.internal, "summary": {}, "truths": {},
            "converged": True, "posterior_models": cached["posterior_models"],
            "display_summary": cached["summary"], "summary_statistic": "mean",
            "summary_version": 1, "summary_space": "display",
        }
        report = benchmark.build_report_card(tier2=aggregate, tier2_posteriors=[post])
        loaded = json.loads(json.dumps(report))  # same strict serialization as checkpoints
        self.assertEqual(loaded["summary_statistic"], "median")
        self.assertEqual(benchmark.select_report_summary(loaded)["summary_statistic"], "median")
        for method, reducer in (("mean", np.mean), ("median", np.median)):
            selected = benchmark.select_report_summary(loaded, method)
            expected = np.sqrt(np.mean([(reducer(d) - t) ** 2 for d, t in zip(draws, truths)]))
            self.assertAlmostEqual(selected["tier2"]["aggregate"]["Physical axis"]["rmse"], expected)
            self.assertEqual(selected["tier2"]["aggregate"]["Physical axis"]["coverage_levels"], aggregate["aggregate"]["Physical axis"]["coverage_levels"])
        chosen = benchmark.select_posterior_summary(loaded["tier2"]["posteriors"][0])
        self.assertEqual(chosen["selected_model"]["grid_params"], [1.])
        self.assertEqual(chosen["display_summary"]["Distance (pc)"]["median"], 100.)

    def test_legacy_result_never_relabels_old_mean_curve(self):
        old = {"samples": self.samples, "labels": self.labels, "internal_labels": self.internal,
               "bestfit_spec": spectrum(self.model), "reduced_chi2": 5., "sirocco_reduced_chi2": 7.}
        for method in ("mean", "median"):
            selected = benchmark.select_posterior_summary(old, method)
            self.assertEqual(selected["selected_model"], {})
            self.assertIsNone(selected["reduced_chi2"])
            self.assertIsNone(selected["sirocco_reduced_chi2"])
            self.assertIn("legacy", selected["summary_notice"])

    def test_tier3_artifact_and_export_match_selected_model(self):
        cached = self.cached()
        for payload in cached["posterior_models"].values():
            payload["observer_angle"] = 55.
        mcmc = {"samples": self.samples, "labels": self.labels, "internal_labels": self.internal,
                "posterior_models": cached["posterior_models"], "summary": {}, "full_chain": self.samples[:, None, :]}
        with tempfile.TemporaryDirectory() as directory:
            artifacts = benchmark._save_tier3_artifacts(directory, "dummy", mcmc, cached["posterior_models"]["mean"]["spectrum"], {}, {}, {})
            with np.load(artifacts["posterior_npz"]) as data:
                np.testing.assert_array_equal(data["samples"], self.samples)
            with np.load(artifacts["plot_data_npz"]) as data:
                self.assertNotEqual(data["mean_model_flux"][0], data["median_model_flux"][0])
            record = {"obs_file": "dummy.csv", "grid_name": "speculate_cv_bl_grid_v87f", "param_names": ["param1"],
                      "posterior_models": cached["posterior_models"], "display_summary": cached["summary"], "artifacts": artifacts,
                      "reduced_chi2": 5., "sirocco_reduced_chi2": 7., "summary_version": 1, "sirocco_statistic": "mean",
                      "wl_range": [1., 3.]}
            median = benchmark.select_posterior_summary(record, "median")
            self.assertIsNone(median["sirocco_reduced_chi2"])
            mean_again = benchmark.select_posterior_summary(json.loads(json.dumps(median)), "mean")
            self.assertEqual(mean_again["sirocco_reduced_chi2"], 7.)
            with patch.object(benchmark, "export_pf_template") as export, patch.object(benchmark, "run_sirocco_model_spectrum") as run:
                benchmark.export_tier3_summary(record)
                run.assert_not_called()
                np.testing.assert_array_equal(export.call_args.args[1], [1.])
                self.assertEqual(export.call_args.kwargs["summary_statistic"], "median")
                self.assertEqual(export.call_args.kwargs["global_params"]["log_scale"], 0.)
            with patch.object(benchmark, "export_pf_template"), patch.object(
                benchmark, "run_sirocco_model_spectrum", return_value={
                    "wavelength": np.array([1., 2., 3.]), "flux": np.array([1., 2., 3.]),
                    "label": "Sirocco Model",
                },
            ) as run:
                exported = benchmark.export_tier3_summary(record, "median", run_sirocco=True, cpus=2)
                self.assertEqual(run.call_args.kwargs["transforms"]["log_scale"], 0.)
                self.assertEqual(exported["sirocco_reduced_chi2"], 14.)
                self.assertEqual(exported["emulator_sirocco_frac_rmse"], 0.)
                json.dumps(exported)

    def test_tier3_uses_mcmc_point_after_mle_freeze_flags_are_restored(self):
        """PPC and Sirocco inputs must not silently retain an MLE-frozen value."""
        cached = self.cached()
        self.model.freeze("log_scale")
        mcmc = {
            "samples": self.samples, "labels": self.labels, "internal_labels": self.internal,
            "summary": {}, "converged": True, "posterior_models": cached["posterior_models"],
            "display_summary": cached["summary"], "full_chain": self.samples[:, None, :],
        }
        self.model.emulator.wl = self.model.data.wave
        self.model.emulator.name = "test_linear_smooth.npz"
        mle = {"model": self.model, "priors": {}, "grid_params": [2.]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dummy.csv"
            path.write_text("Wavelength,Flux,Error\n1,2,1\n2,4,1\n3,6,1\n")
            with patch.object(benchmark, "run_mle_single", return_value=mle), patch.object(
                benchmark, "run_mcmc_single", return_value=mcmc,
            ), patch.object(benchmark, "export_pf_template") as export, patch.object(
                benchmark, "check_sirocco_runtime", return_value={"ok": True},
            ), patch.object(benchmark, "run_sirocco_pf", return_value={"spec_files": [str(path)]}), patch.object(
                benchmark, "_reduce_single_observer_sirocco_spectrum", return_value=path,
            ), patch.object(benchmark, "_load_single_observer_sirocco_spectrum",
                            return_value=(np.array([1., 2., 3.]), np.ones(3))), patch.object(
                benchmark, "_prepare_sirocco_flux_for_fit", return_value=np.ones(3),
            ) as transform:
                result = benchmark.run_tier3_single(
                    self.model.emulator, str(path), output_dir=directory, n_ppc_draws=3,
                    grid_name="speculate_cv_bl_grid_v87f",
                )
            chosen = cached["posterior_models"]["median"]
            self.assertEqual(result["summary_statistic"], "median")
            self.assertEqual(result["sirocco_statistic"], "median")
            self.assertEqual(export.call_args.kwargs["summary_statistic"], "median")
            np.testing.assert_array_equal(export.call_args.args[1], chosen["grid_params"])
            self.assertAlmostEqual(self.model["log_scale"], chosen["parameters"]["log_scale"])
            self.assertEqual(result["sirocco_transform_params"], chosen["transforms"])
            self.assertEqual(export.call_args.kwargs["global_params"]["log_scale"], chosen["parameters"]["log_scale"])
            self.assertTrue(transform.call_args.kwargs["smoothing"])
            self.assertTrue(result["sirocco_smoothing"])
            with np.load(result["artifacts"]["plot_data_npz"]) as data:
                predictions = np.array([
                    np.arange(1., 4.) * draw[0] * np.exp(draw[1]) for draw in self.samples
                ])
                np.testing.assert_allclose(data["ppc_low"], np.percentile(predictions, 2.5, axis=0))


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inference = notebook("speculate_inference")
        cls.viewer = notebook("speculate_benchmark_viewer")

    def test_switch_has_no_graph_path_to_sampling(self):
        for module, switch, trigger in ((self.inference, "mcmc_summary_switch", "run_mcmc_btn"), (self.viewer, "benchmark_summary_switch", "run_btn")):
            # Include the runner as well as any cells that only display its button.
            triggered_cells = {
                cell._cell.cell_id for cell in module.app._cell_manager.cells()
                if trigger in cell.refs
            }
            pending = [defining(module, switch)._cell.cell_id]
            seen = set()
            while pending:
                current = pending.pop()
                if current not in seen:
                    seen.add(current)
                    pending.extend(module.app._graph.children[current])
            self.assertTrue(triggered_cells)
            self.assertEqual(triggered_cells & seen, set())

    def test_tier3_selection_survives_record_updates(self):
        getter, setter = mo.state(1)
        records = [{"obs_file": "first.csv"}, {"obs_file": "second.csv"}]
        cell = defining(self.viewer, "t3_observation_slider")
        _, widgets = cell.run(get_raw_tier3_posteriors=lambda: records,
                              get_tier3_index=getter, set_tier3_index=setter, mo=mo)
        self.assertEqual(widgets["t3_observation_slider"].value, 1)
        updated = [records[0], dict(records[1], summary_exports={"median": {"pf_path": "median.pf"}})]
        _, widgets = cell.run(get_raw_tier3_posteriors=lambda: updated,
                              get_tier3_index=getter, set_tier3_index=setter, mo=mo)
        self.assertEqual(widgets["t3_observation_slider"].value, 1)

    def test_corner_export_skips_a_failed_empty_summary(self):
        cell = next(cell for cell in self.viewer.app._cell_manager.cells()
                    if "t2_corner_export_btn" in cell.refs)
        records = [
            {"summary_statistic": "mean", "display_samples": np.empty((0, 0)),
             "display_labels": []},
            {"summary_statistic": "mean", "display_samples": np.array([[1.], [2.]]),
             "display_labels": ["Axis"]},
        ]
        with patch.object(benchmark, "export_cornerplot_data", return_value={
            "record_count": 1, "bundle_dir": "/tmp/example-bundle",
        }) as export:
            cell.run(get_tier2_posteriors=lambda: records, mo=mo, np=np,
                     t2_corner_export_btn=SimpleNamespace(value=True), time=time)
        self.assertEqual(len(export.call_args.args[0]), 1)
        self.assertEqual(len(export.call_args.kwargs["manifest_metadata"]["skipped"]), 1)

    def test_inference_export_cell_rebuild_has_no_side_effects(self):
        cell = defining(self.inference, "export_pf_btn")
        for method, centre in (("mean", 4.), ("median", 1.)):
            payload = {"summary_statistic": method, "grid_params": [centre], "parameters": {"param1": centre, "Av": .3}, "wavelength": [1., 2., 3.]}
            with patch.object(benchmark, "export_pf_template") as export, patch.object(benchmark, "run_sirocco_model_spectrum") as simulate:
                _, widgets = cell.run(
                    emu=SimpleNamespace(param_names=["param1"]), export_dir_input=SimpleNamespace(value="/tmp/speculate-test-export"),
                    get_mcmc_bestfit_plot=lambda: payload, get_mcmc_corner_meta=lambda: {"summary_statistic": method},
                    get_mcmc_labels=lambda: ["Physical axis"], get_mcmc_samples=lambda: np.array([[1.], [1.], [10.]]),
                    get_mcmc_summary_df=lambda: None, get_mle_model=Model,
                    grid_selector=SimpleNamespace(value="speculate_cv_bl_grid_v87f"), inference_sirocco_cpu_slider=None,
                    mo=mo, np=np, obs_flux_scale=SimpleNamespace(value="linear"), os=os,
                    set_mcmc_sirocco_spectrum=lambda value: None, set_mcmc_export_status=lambda value: None,
                )
                export.assert_not_called()
                simulate.assert_not_called()
                widgets["export_pf_btn"]._on_click(None)
                export.assert_called_once()
                np.testing.assert_array_equal(export.call_args.args[1], [centre])
                self.assertEqual(export.call_args.kwargs["summary_statistic"], method)

    def test_inference_toggle_reuses_samples_models_and_playground_point(self):
        """Exercise the real sampling cell/renderer with a deterministic chain."""
        model = Model()
        rng = np.random.default_rng(6)
        chain = np.stack((rng.lognormal(size=(20, 8)), rng.normal(size=(20, 8))), axis=-1)

        class Sampler:
            iteration = 0

            def run_mcmc(self, *args, **kwargs):
                if self.iteration == 0:
                    self.initial_positions = np.array(args[0])
                self.iteration += 1

            def get_chain(self, discard=0, thin=1):
                return chain[discard::thin]

            def get_autocorr_time(self, **kwargs):
                raise ValueError("short test chain")

        sampler = Sampler()
        get_renderer, set_renderer = mo.state(None)
        state = {}
        cell = next(cell for cell in self.inference.app._cell_manager.cells()
                    if "run_mcmc_btn" in cell.refs)
        args = {
            "get_mle_model": lambda: model,
            "get_mle_priors": lambda: {
                "param1": stats.uniform(0, 20),
                "log_scale": distance_prior_to_log_scale_prior(100., 45.),
            },
            "ground_truth_params": {}, "grid_selector": SimpleNamespace(value=None),
            "mcmc_burnin": SimpleNamespace(value=1),
            "mcmc_freeze": SimpleNamespace(value={}),
            "mcmc_label_map": {"param1": "Physical axis", "log_scale": "Distance (pc)"},
            "mcmc_nsteps": SimpleNamespace(value=20), "mcmc_nwalkers": SimpleNamespace(value=8),
            "mo": mo, "np": np, "stats": stats, "run_mcmc_btn": SimpleNamespace(value=True),
            "set_mcmc_renderer": set_renderer,
        }
        for name in cell.refs:
            if name.startswith("set_") and name not in args:
                args[name] = lambda value, key=name: state.update({key: value})
        with patch("emcee.EnsembleSampler", return_value=sampler) as construct, patch.object(
            benchmark, "_model_bestfit_spectrum", side_effect=spectrum
        ) as evaluate, patch.object(
            benchmark, "_model_likelihood_diagnostics", return_value={}
        ), patch.object(benchmark, "_extract_spectrum_nuisance_transforms", return_value={}):
            cell.run(**args)
            self.assertIsNotNone(get_renderer())
            _, switch_state = defining(self.inference, "get_mcmc_report_means").run(mo=mo)
            switch_cell = defining(self.inference, "mcmc_summary_switch")
            render_cell = defining(self.inference, "mcmc_results")
            _, widgets = switch_cell.run(mo=mo, **switch_state)
            self.assertFalse(widgets["mcmc_summary_switch"].value)
            for method in ("median", "mean", "median"):
                widgets["mcmc_summary_switch"]._update(method == "mean")
                _, widgets = switch_cell.run(mo=mo, **switch_state)
                switch = widgets["mcmc_summary_switch"]
                self.assertEqual(switch._args.label, f"Report posterior {method}s")
                _, rendered = render_cell.run(
                    get_mcmc_renderer=get_renderer, mcmc_summary_switch=switch, mo=mo,
                )
                button = rendered["mcmc_playground_export_button"]
                point = state["set_mcmc_bestfit_plot"]
                self.assertEqual(point["summary_statistic"], method)
                self.assertAlmostEqual(
                    log_scale_to_distance_pc(point["parameters"]["log_scale"]),
                    (np.mean if method == "mean" else np.median)(
                        state["set_mcmc_samples"][:, 1]
                    ),
                )
                with patch.object(mo.status, "toast"):
                    button._on_click(None)
                self.assertEqual(state["set_inf_playground_target"]["phys"], point["grid_params"])
            self.assertEqual(evaluate.call_count, 2)
            construct.assert_called_once()
            self.assertEqual(sampler.iteration, 20)
            self.assertTrue(np.isfinite(sampler.initial_positions).all())
            self.assertGreater(np.ptp(sampler.initial_positions[:, 1]), 0.)
            self.assertEqual(plt.get_fignums(), [])

    def test_benchmark_switch_restores_report_choice_and_updates_selected_model(self):
        cached = PosteriorTests("test_cached_models_use_selected_full_point_and_restore_model")
        cached.setUp()
        result = cached.cached()
        record = {
            "samples": cached.samples, "labels": cached.labels,
            "internal_labels": cached.internal, "posterior_models": result["posterior_models"],
        }
        state_cell = defining(self.viewer, "get_benchmark_report_means")
        switch_cell = defining(self.viewer, "benchmark_summary_switch")
        result_cell = defining(self.viewer, "benchmark_summary_statistic")
        legacy_report = {"tier2": {"aggregate": {"shape": {"rmse": 0.5, "bias": 0.1}}}}
        for report, use_means in (
            (None, False), (legacy_report, True),
            ({"summary_statistic": "mean"}, True), ({"summary_statistic": "median"}, False),
        ):
            get_raw_report = lambda: report
            _, switch_state = state_cell.run(get_raw_report=get_raw_report, mo=mo)
            _, widgets = switch_cell.run(mo=mo, **switch_state)
            self.assertEqual(widgets["benchmark_summary_switch"].value, use_means)
            for method, centre in (("median", 1.), ("mean", 4.)):
                widgets["benchmark_summary_switch"]._update(method == "mean")
                _, widgets = switch_cell.run(mo=mo, **switch_state)
                switch = widgets["benchmark_summary_switch"]
                self.assertEqual(switch._args.label, f"Report posterior {method}s")
                _, selected = result_cell.run(
                    benchmark_summary_switch=switch, get_raw_report=get_raw_report,
                    get_raw_tier2_posteriors=lambda: [record],
                    get_raw_tier3_posteriors=lambda: [record],
                )
                self.assertEqual(selected["benchmark_summary_statistic"], method)
                if report is legacy_report:
                    rmse = selected["get_report"]()["tier2"]["aggregate"]["shape"]["rmse"]
                    self.assertTrue(np.isnan(rmse) if method == "median" else rmse == 0.5)
                for getter in ("get_tier2_posteriors", "get_tier3_posteriors"):
                    point = selected[getter]()[0]["selected_model"]
                    self.assertEqual(point["grid_params"], [centre])

    def test_report_loader_keeps_model_caches_and_coordinate_metadata(self):
        cached = PosteriorTests("test_cached_models_use_selected_full_point_and_restore_model")
        cached.setUp()
        result = cached.cached()
        post = {"run": 0, "samples": cached.samples, "labels": cached.labels, "internal_labels": cached.internal,
                "posterior_models": result["posterior_models"], "display_summary": result["summary"],
                "summary_version": 1, "summary_statistic": "mean", "summary_space": "display"}
        report = benchmark.build_report_card(tier2={
            "aggregate": {}, "n_spectra": 1, "n_processed": 1, "n_failures": 0,
            "tier2_time_s": 1., "mcmc_config": {},
        }, tier2_posteriors=[post])
        loader = next(cell for cell in self.viewer.app._cell_manager.cells() if "load_btn" in cell.refs and "set_tier2_posteriors" in cell.refs)
        loaded = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            loader.run(compare_btn=SimpleNamespace(value=False), compare_picker=SimpleNamespace(value=[]),
                       json=json, load_btn=SimpleNamespace(value=True), np=np, os=os, report_picker=SimpleNamespace(value=str(path)),
                       set_comparison_reports=lambda value: None, set_report=lambda value: None,
                       set_tier1_arrays=lambda value: None, set_tier2_posteriors=loaded.append,
                       set_tier3_posteriors=lambda value: None, upload_btn=SimpleNamespace(value=[]))
        selected = benchmark.select_posterior_summary(loaded[0][0], "median")
        self.assertEqual(selected["selected_model"]["parameters"]["param1"], 1.)
        self.assertEqual(selected["display_summary"]["Distance (pc)"]["median"], 100.)

    def test_live_and_checkpoint_payloads_retain_both_model_caches(self):
        """Execute the actual payload expressions without the expensive runner."""
        helper = PosteriorTests("test_cached_models_use_selected_full_point_and_restore_model")
        helper.setUp()
        cached = helper.cached()
        mcmc = {
            "samples": helper.samples, "full_chain": helper.samples[:, None, :],
            "labels": helper.labels, "internal_labels": helper.internal,
            "summary": cached["summary"], "converged": True,
            "posterior_models": cached["posterior_models"], "display_summary": cached["summary"],
            "summary_statistic": "median", "summary_space": "display", "summary_version": 1,
            "bestfit_spec": cached["posterior_models"]["median"]["spectrum"],
        }
        namespace = {
            "_mcmc": mcmc, "_run_idx": 0, "_sf": Path("run0.spec"), "_inc": 55.,
            "_mcmc_burnin_val": 1, "_mcmc_labels": helper.labels, "_mle": {},
            "_gt": {}, "_prior_ranges_dict": {}, "_spec_res": {},
        }
        tree = ast.parse((ROOT / "speculate_benchmark_viewer.py").read_text())
        payloads = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
            if "full_samples" in keys or ("mle_bestfit_spec" in keys and "samples" in keys):
                payloads.append(eval(compile(ast.Expression(node), "<benchmark payload>", "eval"), namespace))
        self.assertEqual(len(payloads), 2)
        for payload in payloads:
            self.assertEqual(payload["posterior_mean_bestfit_spec"], {})
            if "full_samples" in payload:
                restored = json.loads(json.dumps(payload))
            else:
                report = benchmark.build_report_card(tier2={
                    "aggregate": {}, "n_spectra": 1, "n_processed": 1, "n_failures": 0,
                    "tier2_time_s": 1., "mcmc_config": {},
                }, tier2_posteriors=[payload])
                restored = json.loads(json.dumps(report))["tier2"]["posteriors"][0]
            self.assertEqual(restored["internal_labels"], helper.internal)
            self.assertEqual(restored["posterior_models"]["median"]["grid_params"], [1.])
            self.assertEqual(restored["summary_version"], 1)


if __name__ == "__main__":
    unittest.main()
