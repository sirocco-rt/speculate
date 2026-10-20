"""Regression coverage for observation formats, file management and notebook controls."""

import base64
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import altair as alt
import marimo as mo
import numpy as np
import pandas as pd

from Speculate_addons.observation_files import (
    delete_uploaded_observation,
    list_observation_files,
    read_observation_file,
    save_uploaded_observation,
)
from Speculate_addons.observation_priors import OBSERVATION_PRIORS


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_notebook(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Validate the complete reactive graph without running its expensive cells.
    module.app._maybe_initialize()
    return module


def cell_defining(module, name):
    return next(cell for cell in module.app._cell_manager.cells() if name in cell.defs)


class ObservationFormatTests(unittest.TestCase):
    def test_named_and_headerless_tables_preserve_all_samples(self):
        """TXT delimiters, comments and headers must not consume a data row."""
        examples = (
            b'WAVELENGTH,FLUX,ERROR\n2000,3,0.3\n1000,2,0.2\n',
            b'" Wavelength ","Flux","Error"\n2000,3,0.3\n1000,2,0.2\n',
            b'Wavelength\tFlux\tError\n2000\t3\t0.3\n1000\t2\t0.2\n',
            b'\xef\xbb\xbf# Instrument metadata\n\n2000  3  0.3\n1000 2 0.2\n',
            b'2000,3,0.3\n1000,2,0.2\n',
            b'# Flux Wavelength Error\n3 2000 0.3\n2 1000 0.2\n',
            b'# "Flux" "Wavelength" "Error"\n3 2000 0.3\n2 1000 0.2\n',
            b'# observation, "unclosed quote in a comment\n'
            b'Wavelength Flux Error\n2000 3 0.3\n1000 2 0.2\n',
        )
        for contents in examples:
            with self.subTest(contents=contents):
                data = read_observation_file(contents)
                self.assertEqual(data["wavelength"].tolist(), [1000, 2000])
                self.assertEqual(data["flux"].tolist(), [2, 3])
                np.testing.assert_allclose(data["error"], [0.2, 0.3])

    def test_missing_errors_and_optional_metadata_are_preserved(self):
        data = read_observation_file(
            b"Wavelength,Flux,Continuum,Sigma_Source\n"
            b"1000,2,1.5,synthetic_sirocco_continuum\n"
        )
        self.assertNotIn("error", data)
        self.assertEqual(data["continuum"].iloc[0], 1.5)
        self.assertEqual(data["sigma_source"].iloc[0], "synthetic_sirocco_continuum")
        # An unfamiliar named column must not be guessed to mean uncertainty.
        named_extra = read_observation_file(b"# wavelength flux ivar\n1000 2 25\n")
        self.assertNotIn("error", named_extra)
        self.assertIn("ivar", named_extra)
        self.assertEqual(
            read_observation_file(b"1000 2\n").columns.tolist(),
            ["wavelength", "flux"],
        )
        # These values are handled by Quick Fit's existing uncertainty fallback.
        data = read_observation_file(
            b"Wavelength,Flux,Error\n1000,2,\n2000,3,NaN\n3000,4,0\n"
        )
        self.assertTrue(data["error"].iloc[:2].isna().all())
        self.assertEqual(data["error"].iloc[2], 0)

    def test_ambiguous_or_invalid_tables_raise_useful_errors(self):
        examples = (
            (b"", "No observation data"),
            (b"# only a comment\n", "No observation data"),
            (b"\xff", "UTF-8"),
            (b"Wavelength Flux\n", "no data rows"),
            (b"Wave Intensity\n1000 2\n", "WAVELENGTH and FLUX"),
            (b"1000 2 0.1 999\n", "two or three"),
            (b"Wavelength Flux\n1000 2 3\n", "Data row 1"),
            (b"1000 2\n2000\n", "Data row 2"),
            (b"Wavelength Flux FLUX\n1000 2 3\n", "unique"),
            (b"Wavelength Flux\n1000 bad\n", "FLUX must contain numeric"),
            (b"Wavelength,Flux\n1000,\n", "FLUX must contain finite"),
            (b"Wavelength Flux\ninf 2\n", "WAVELENGTH must contain finite"),
        )
        for contents, message in examples:
            with self.subTest(contents=contents):
                with self.assertRaisesRegex(ValueError, message):
                    read_observation_file(contents)

    def test_all_shipped_csvs_match_the_existing_reader(self):
        """Preserve wavelengths, fluxes and errors for every shipped spectrum."""
        for path in sorted((REPO_ROOT / "observation_files").glob("*.csv")):
            with self.subTest(filename=path.name):
                expected = pd.read_csv(path)
                expected.columns = expected.columns.str.lower()
                expected = expected.sort_values("wavelength", kind="stable").reset_index(drop=True)
                pd.testing.assert_frame_equal(read_observation_file(path), expected)


class ObservationFileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "observation_files"

    def test_upload_list_and_delete_an_older_malformed_file(self):
        save_uploaded_observation("old.TXT", b"1000 2\n", self.directory)
        # Represents an invalid file saved before upload validation existed.
        (self.directory / "old.TXT").write_text("wrong format")
        save_uploaded_observation("new.csv", b"Wavelength,Flux\n2000,3\n", self.directory)
        self.assertEqual(list_observation_files(self.directory), ["new.csv", "old.TXT"])
        delete_uploaded_observation("old.TXT", self.directory)
        self.assertEqual(list_observation_files(self.directory), ["new.csv"])
        delete_uploaded_observation("new.csv", self.directory)
        self.assertEqual(list_observation_files(self.directory), [])

    def test_bundled_demos_cannot_be_deleted_or_overwritten(self):
        self.directory.mkdir()
        for name in OBSERVATION_PRIORS:
            with self.subTest(filename=name):
                filename = name.upper()
                path = self.directory / filename
                path.write_bytes(b"original demo")
                with self.assertRaisesRegex(ValueError, "Bundled demo"):
                    delete_uploaded_observation(filename, self.directory)
                with self.assertRaisesRegex(ValueError, "Bundled demo"):
                    save_uploaded_observation(filename, b"1000 2\n", self.directory)
                self.assertEqual(path.read_bytes(), b"original demo")

    def test_invalid_upload_does_not_replace_existing_file(self):
        save_uploaded_observation("user.txt", b"1000 2\n", self.directory)
        with self.assertRaises(ValueError):
            save_uploaded_observation("user.txt", b"invalid", self.directory)
        self.assertEqual((self.directory / "user.txt").read_bytes(), b"1000 2\n")
        with self.assertRaises(ValueError):
            save_uploaded_observation("new.txt", b"invalid", self.directory)
        self.assertFalse((self.directory / "new.txt").exists())

    def test_only_csv_txt_plain_files_are_listed_and_uploaded(self):
        self.assertEqual(list_observation_files(self.directory), [])
        self.directory.mkdir()
        for filename in ("a.CSV", "b.TXT", "c.dat", "d.fits"):
            (self.directory / filename).write_bytes(b"1000 2\n")
        (self.directory / "folder.txt").mkdir()
        (self.directory / "link.txt").symlink_to(self.directory / "a.CSV")
        self.assertEqual(list_observation_files(self.directory), ["a.CSV", "b.TXT"])
        for filename in ("c.dat", "d.fits", "../outside.txt", "/outside.csv", "dir\\file.txt", "link.txt"):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    save_uploaded_observation(filename, b"1000 2\n", self.directory)
        self.assertEqual((self.directory / "a.CSV").read_bytes(), b"1000 2\n")


class ObservationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {
            "quickfit": load_notebook("speculate_quick_fit.py"),
            "inspector": load_notebook("speculate_grid_inspector.py"),
            "inference": load_notebook("speculate_inference.py"),
        }
        cls.benchmark = load_notebook("speculate_benchmark_viewer.py")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(temporary.name)
        Path("observation_files").mkdir()
        self.toasts = self.enterContext(patch.object(mo.status, "toast"))

    def selector(self, name):
        """Run each notebook's disk discovery and dropdown cells."""
        module = self.notebooks[name]
        if name == "quickfit":
            _, definitions = cell_defining(module, "qf_obs_selector").run(
                mo=mo,
                get_qf_obs_refresh=lambda: 1,
                qf_inf_emu_data=None,
                qf_inf_model_selector=None,
            )
            return definitions["qf_obs_selector"]
        if name == "inference":
            grid = SimpleNamespace(value=None)
            _, definitions = cell_defining(module, "obs_files").run(
                get_obs_refresh=lambda: 1, grid_selector=grid, os=os
            )
            _, definitions = cell_defining(module, "obs_file_selector").run(
                grid_selector=grid,
                mo=mo,
                obs_files=definitions["obs_files"],
                test_grid_files=definitions["test_grid_files"],
            )
            return definitions["obs_file_selector"]
        _, definitions = cell_defining(module, "obs_files").run(
            get_obs_refresh=lambda: 1, selected_grid=None
        )
        _, definitions = cell_defining(module, "obs_file_selector").run(
            mo=mo, obs_files=definitions["obs_files"]
        )
        return definitions["obs_file_selector"]

    def delete_button(self, name, selected, refresh):
        module = self.notebooks[name]
        button_name = "qf_delete_obs_btn" if name == "quickfit" else "delete_obs_btn"
        selector_name = "qf_obs_selector" if name == "quickfit" else "obs_file_selector"
        setter_name = "set_qf_obs_refresh" if name == "quickfit" else "set_obs_refresh"
        _, definitions = cell_defining(module, button_name).run(
            **{"mo": mo, selector_name: selected, setter_name: refresh}
        )
        return definitions[button_name]

    def upload_controls(self, name, refresh):
        """Build the real widget so tests exercise its frontend-change callback."""
        uploader_name = "qf_obs_uploader" if name == "quickfit" else "obs_file_uploader"
        setter_name = "set_qf_obs_refresh" if name == "quickfit" else "set_obs_refresh"
        _, definitions = cell_defining(self.notebooks[name], uploader_name).run(
            **{"mo": mo, setter_name: refresh}
        )
        return definitions[uploader_name], definitions

    def upload(self, uploader, filename, contents=b"Wavelength,Flux\n1000,2\n"):
        uploader._update([(filename, base64.b64encode(contents).decode("ascii"))])

    def test_upload_then_delete_any_selection_and_refresh(self):
        """The selected older file is deleted while the latest upload remains."""
        for name in self.notebooks:
            with self.subTest(notebook=name):
                old = Path("observation_files/older.txt")
                old.write_text("malformed old upload")
                refreshes = []

                def refresh(update):
                    refreshes.append(update(len(refreshes)))

                uploader, _ = self.upload_controls(name, refresh)
                self.upload(uploader, "newest.csv")
                self.assertEqual(refreshes, [1])
                self.assertEqual(set(self.selector(name).options), {"older.txt", "newest.csv"})
                self.toasts.assert_called_with("Uploaded newest.csv")
                selected = mo.ui.dropdown(options=["older.txt", "newest.csv"], value="older.txt")
                button = self.delete_button(name, selected, refresh)
                self.assertFalse(button._component_args["disabled"])
                button._on_click(None)
                self.assertFalse(old.exists())
                self.assertTrue(Path("observation_files/newest.csv").exists())
                self.assertEqual(refreshes, [1, 2])
                selected = self.selector(name)
                self.assertEqual(list(selected.options), ["newest.csv"])
                self.delete_button(name, selected, refresh)._on_click(None)
                self.assertEqual(refreshes, [1, 2, 3])
                self.assertFalse(list(Path("observation_files").iterdir()))
                self.assertTrue(
                    self.delete_button(name, self.selector(name), refresh)._component_args["disabled"]
                )
                # Clearing after Delete Selected is harmless and refreshes a stale list.
                uploader._update([])
                self.assertEqual(refreshes, [1, 2, 3, 4])
                self.assertIn("already removed", self.toasts.call_args.args[0])

    def test_clear_deletes_its_upload_and_keeps_other_observations(self):
        """Clear targets its widget's saved upload, independently of the dropdown."""
        for name in self.notebooks:
            with self.subTest(notebook=name):
                other = Path("observation_files/another.txt")
                other.write_text("1000 4\n")
                refreshes = []
                uploader, _ = self.upload_controls(name, refreshes.append)
                uploader._update([])
                self.assertEqual(refreshes, [])
                self.upload(uploader, "newest.csv")
                self.assertEqual(self.selector(name).value, "another.txt")
                uploader._update([])
                self.assertEqual(len(refreshes), 2)
                self.toasts.assert_called_with("Deleted newest.csv")
                self.assertFalse(Path("observation_files/newest.csv").exists())
                self.assertEqual(list(self.selector(name).options), ["another.txt"])
                uploader._update([])
                self.assertEqual(len(refreshes), 2)
                self.upload(uploader, "newest.csv")
                self.assertTrue(Path("observation_files/newest.csv").exists())
                uploader._update([])
                self.assertFalse(Path("observation_files/newest.csv").exists())
                other.unlink()

    def test_clear_after_rejected_upload_preserves_existing_files_and_demos(self):
        for name in self.notebooks:
            with self.subTest(notebook=name):
                refreshes = []
                uploader, _ = self.upload_controls(name, refreshes.append)
                self.upload(uploader, "existing.csv")
                saved = Path("observation_files/existing.csv").read_bytes()
                for filename in ("existing.csv", "invalid.txt"):
                    self.upload(uploader, filename, b"not a spectrum")
                    self.assertIn("Upload failed:", self.toasts.call_args.args[0])
                    uploader._update([])
                    self.assertEqual(len(refreshes), 1)
                    self.assertEqual(Path("observation_files/existing.csv").read_bytes(), saved)
                    self.assertFalse(Path("observation_files/invalid.txt").exists())
                for filename in ("CV:ixvel_all.csv", "AGN:nonbal_qso_composite.csv"):
                    demo = Path("observation_files") / filename
                    demo.write_bytes(saved)
                    self.upload(uploader, filename)
                    uploader._update([])
                    self.assertEqual(demo.read_bytes(), saved)
                    self.assertEqual(len(refreshes), 1)
                    demo.unlink()
                Path("observation_files/existing.csv").unlink()

    def test_failed_clear_reports_error_and_keeps_saved_file(self):
        for name in self.notebooks:
            with self.subTest(notebook=name):
                refreshes = []
                uploader, _ = self.upload_controls(name, refreshes.append)
                self.upload(uploader, "saved.csv")
                with patch(
                    "Speculate_addons.observation_files.delete_uploaded_observation",
                    side_effect=PermissionError("read-only directory"),
                ):
                    uploader._update([])
                self.assertTrue(Path("observation_files/saved.csv").exists())
                self.assertEqual(len(refreshes), 1)
                self.assertIn("Could not delete saved.csv", self.toasts.call_args.args[0])
                self.delete_button(name, self.selector(name), refreshes.append)._on_click(None)
                self.assertFalse(Path("observation_files/saved.csv").exists())

    def test_demo_buttons_are_disabled_and_callbacks_also_protect_files(self):
        for name in self.notebooks:
            for filename in ("CV:ixvel_all.csv", "AGN:nonbal_qso_composite.csv"):
                with self.subTest(notebook=name, filename=filename):
                    path = Path("observation_files") / filename
                    path.write_text("original demo")
                    selected = mo.ui.dropdown(options=[filename], value=filename)
                    refreshes = []
                    button = self.delete_button(name, selected, refreshes.append)
                    self.assertTrue(button._component_args["disabled"])
                    button._on_click(None)
                    self.assertEqual(path.read_text(), "original demo")
                    self.assertEqual(refreshes, [])
                    path.unlink()

    def test_missing_file_refreshes_and_delete_failures_are_visible(self):
        for name in self.notebooks:
            with self.subTest(notebook=name):
                selected = mo.ui.dropdown(options=["missing.txt"], value="missing.txt")
                refreshes = []
                self.delete_button(name, selected, refreshes.append)._on_click(None)
                self.assertEqual(len(refreshes), 1)
                self.assertIn("already removed", self.toasts.call_args.args[0])
                with patch(
                    "Speculate_addons.observation_files.delete_uploaded_observation",
                    side_effect=PermissionError("read-only directory"),
                ):
                    self.delete_button(name, selected, refreshes.append)._on_click(None)
                self.assertEqual(len(refreshes), 1)
                self.assertIn("Could not delete", self.toasts.call_args.args[0])

    def test_refreshing_the_list_cannot_retrigger_an_upload(self):
        """Check the graph for a path from disk-list refresh to file saving."""
        for name, module in self.notebooks.items():
            with self.subTest(notebook=name):
                list_name = "qf_obs_selector" if name == "quickfit" else "obs_files"
                uploader = "qf_obs_uploader" if name == "quickfit" else "obs_file_uploader"
                upload_cell = cell_defining(module, uploader)
                visited = set()
                pending = [cell_defining(module, list_name)._cell.cell_id]
                while pending:
                    cell_id = pending.pop()
                    if cell_id in visited:
                        continue
                    visited.add(cell_id)
                    pending.extend(module.app._graph.children[cell_id])
                self.assertNotIn(upload_cell._cell.cell_id, visited)

    def test_quickfit_reads_txt_and_keeps_existing_uncertainty_metadata(self):
        Path("observation_files/user.txt").write_text("2000 3\n1000 2\n")
        _, definitions = cell_defining(self.notebooks["quickfit"], "qf_obs_data").run(
            mo=mo, np=np, os=os, pd=pd,
            qf_data_source_selector=SimpleNamespace(value="Observation File"),
            qf_obs_selector=SimpleNamespace(value="user.txt"),
            qf_test_grid_info={},
            qf_test_grid_params_df=None,
            qf_test_inclination_selector=SimpleNamespace(value=None),
            qf_test_run_selector=SimpleNamespace(value=None),
        )
        data = definitions["qf_obs_data"]
        self.assertEqual(data["wavelength"].tolist(), [1000, 2000])
        self.assertEqual(data["sigma_source"].unique().tolist(), ["uploaded_default_continuum"])
        self.assertEqual(data.attrs["source_label"], "user.txt")
        self.assertFalse(data.attrs["is_test_grid"])

    def test_quickfit_shows_delete_with_an_invalid_file_and_no_model(self):
        """A parsing error must leave the Stage 3 file controls usable."""
        Path("observation_files/bad.txt").write_text("wrong format")
        module = self.notebooks["quickfit"]
        selected = self.selector("quickfit")
        with patch.object(mo.output, "replace"):
            _, definitions = cell_defining(module, "qf_obs_data").run(
                mo=mo, np=np, os=os, pd=pd,
                qf_data_source_selector=SimpleNamespace(value="Observation File"),
                qf_obs_selector=selected,
                qf_test_grid_info={},
                qf_test_grid_params_df=None,
                qf_test_inclination_selector=SimpleNamespace(value=None),
                qf_test_run_selector=SimpleNamespace(value=None),
            )
        self.assertIsNone(definitions["qf_obs_data"])
        _, upload_defs = self.upload_controls("quickfit", lambda update: None)
        layout_cell = next(
            cell for cell in module.app._cell_manager.cells()
            if {"qf_file_upload_ui", "qf_delete_obs_btn"}.issubset(cell.refs)
        )
        button = self.delete_button("quickfit", selected, lambda update: None)
        output, _ = layout_cell.run(
            json=json, mo=mo, np=np, os=os,
            qf_data_source_info="",
            qf_data_source_selector=mo.ui.dropdown(
                options=["Observation File", "Test Grid"], value="Observation File"
            ),
            qf_delete_obs_btn=button,
            qf_file_upload_ui=upload_defs["qf_file_upload_ui"],
            qf_inf_emu_data=None,
            qf_inf_is_grid_interp=False,
            qf_inf_model_picker=None,
            qf_inf_model_selector=None,
            qf_inf_models=None,
            qf_inf_wl_slider=mo.ui.range_slider(start=1000, stop=2000),
            qf_obs_data=None,
            qf_obs_scale_selector=mo.ui.dropdown(options=["linear"], value="linear"),
            qf_obs_selector=selected,
            qf_param_map_db_for_grid=None,
            set_qf_playground_target=None,
            qf_test_inclination_selector=None,
            qf_test_run_selector=None,
        )
        self.assertIn("Delete Selected", output._text)
        self.assertIn("CSV or TXT", output._text)
        self.assertFalse(button._component_args["disabled"])

    def test_inspector_plots_txt_and_displays_read_errors(self):
        module = self.notebooks["inspector"]
        cell = cell_defining(module, "spectrum_chart")
        path = Path("observation_files/user.txt")
        path.write_text("1000 2\n2000 3\n")
        arguments = dict(
            alt=alt, mo=mo, np=np, os=os, pd=pd,
            current_flux=None,
            current_run_index=0,
            fit_power_law_continuum=None,
            flux_scale_selector=SimpleNamespace(value="linear"),
            inclination_selector=SimpleNamespace(value=60),
            load_spectrum=None,
            obs_file_selector=SimpleNamespace(value="user.txt"),
            pinned_spectra_indices=[],
            selected_grid="test",
            show_current=SimpleNamespace(value=False),
            show_fixed=SimpleNamespace(value=False),
            show_grid=SimpleNamespace(value=False),
            show_obs=SimpleNamespace(value=True),
            use_dimensionless=SimpleNamespace(value=False),
            use_smoothing=SimpleNamespace(value=False),
            wavelength_range=SimpleNamespace(value=(1000, 2000)),
            wavelengths=np.array([1000, 2000]),
        )
        _, definitions = cell.run(**arguments)
        plotted = definitions["plot_data_list"]
        self.assertEqual(len(plotted), 1)
        self.assertEqual(plotted[0]["Flux"].tolist(), [2, 3])
        path.write_text("wrong format")
        output, definitions = cell.run(**arguments)
        self.assertFalse(definitions["plot_data_list"])
        self.assertIn("Could not load observation", output._text)

    def test_inference_reads_txt_and_reports_malformed_data(self):
        path = Path("observation_files/user.txt")
        path.write_text("2000 3 0.3\n1000 2 0.2\n")
        arguments = dict(
            build_synthetic_sirocco_sigma=None,
            data_source_selector=SimpleNamespace(value="Observation File"),
            emu=None,
            grid_selector=SimpleNamespace(value=None),
            mo=mo, np=np, os=os, pd=pd,
            obs_file_selector=SimpleNamespace(value="user.txt"),
            test_grid_params_df=None,
            test_grid_path=None,
            test_inclination_selector=SimpleNamespace(value=None),
            test_run_selector=SimpleNamespace(value=None),
        )
        cell = cell_defining(self.notebooks["inference"], "obs_data")
        _, definitions = cell.run(**arguments)
        self.assertEqual(definitions["obs_data"]["wavelength"].tolist(), [1000, 2000])
        np.testing.assert_allclose(definitions["obs_data"]["error"], [0.2, 0.3])
        self.assertEqual(definitions["obs_flux_scale"].value, "linear")
        path.write_text("wrong format")
        with patch.object(mo.output, "replace") as show_error:
            _, definitions = cell.run(**arguments)
        self.assertIsNone(definitions["obs_data"])
        self.assertIn("Error reading file", show_error.call_args.args[0]._text)
        button = self.delete_button("inference", self.selector("inference"), lambda update: None)
        self.assertFalse(button._component_args["disabled"])

    def test_observation_uploaders_share_the_same_formats_and_help(self):
        from Speculate_addons.observation_files import OBSERVATION_UPLOAD_HELP

        rendered_help = mo.md(OBSERVATION_UPLOAD_HELP)._text
        for name in self.notebooks:
            uploader, definitions = self.upload_controls(name, lambda update: None)
            self.assertEqual(tuple(uploader._component_args["filetypes"]), (".csv", ".txt"))
            help_name = "qf_file_upload_ui" if name == "quickfit" else "file_upload_ui"
            rendered_ui = definitions[help_name]._text
            self.assertIn("marimo-accordion", rendered_ui)
            self.assertIn("marimo-tabs", rendered_ui)
            self.assertIn("Observation file format", rendered_ui)
            self.assertIn(rendered_help, rendered_ui)
            for label in ("Format guide", "CSV", "TXT (spaces)", "TXT (tabs)", "No headers"):
                self.assertIn(label, rendered_ui)
            self.assertIn("Download example", rendered_ui)

    def test_benchmark_discovers_txt_from_the_shared_directory(self):
        for filename in ("user.txt", "user.CSV", "skip.dat", "skip.fits"):
            Path("observation_files", filename).write_text("1000 2\n2000 3\n")
        _, definitions = cell_defining(self.benchmark, "obs_picker").run(
            emu_picker=SimpleNamespace(value=None), mo=mo, os=os
        )
        self.assertEqual(
            definitions["obs_picker"].options,
            {
                "user.CSV": os.path.join("observation_files", "user.CSV"),
                "user.txt": os.path.join("observation_files", "user.txt"),
            },
        )

    def test_benchmark_loads_txt_before_starting_the_fit(self):
        """Exercise Tier 3's real input path while stopping before MLE/MCMC."""
        from Speculate_addons import speculate_benchmark

        path = Path("observation_files/user.txt")
        path.write_text("2000 3 0.3\n1000 2 0.2\n")

        class FitReached(Exception):
            pass

        with patch.object(speculate_benchmark, "run_mle_single", side_effect=FitReached) as fit:
            with self.assertRaises(FitReached):
                speculate_benchmark.run_tier3_single(
                    SimpleNamespace(wl=np.array([1000, 2000])),
                    str(path),
                    require_sirocco=False,
                    run_sirocco=False,
                )
        np.testing.assert_allclose(fit.call_args.args[1], [1000, 2000])
        np.testing.assert_allclose(fit.call_args.args[2], [2, 3])
        np.testing.assert_allclose(fit.call_args.args[3], [0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
