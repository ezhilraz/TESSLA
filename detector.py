# detector.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from astropy.io import fits


class BLS:
    def __init__(
        self,
        TIC_ID,
        lc_flat,
        df,
        Plot=True,
        save_fits=True,
        base_dir=".",
        min_period=0.5,
        max_period=200,
        n_periods=20000,
        min_duration=0.03,
        max_duration=0.30,
        n_durations=30,
        frequency_factor=500,
        window_length=401,
        polyorder=2,
        n_bins=2001,
        smooth_window=51,
        smooth_polyorder=2,
        verbose=True
    ):
        self.tic_id = int(TIC_ID)
        self.lc_flat_initial = lc_flat
        self.df = df.copy()

        self.Plot = Plot
        self.save_fits = save_fits
        self.base_dir = base_dir
        self.plot_dir = os.path.join(base_dir, "plots", "lc_plots")
        self.processed_dir = os.path.join(base_dir, "processed_fits")

        self.min_period = min_period
        self.max_period = max_period
        self.n_periods = n_periods
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_durations = n_durations
        self.frequency_factor = frequency_factor

        self.window_length = window_length
        self.polyorder = polyorder

        self.n_bins = n_bins
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder

        self.verbose = verbose

        self.bls = None
        self.period = None
        self.t0 = None
        self.duration = None
        self.transit_mask = None

        self.lc_flat = None
        self.trend = None
        self.folded_lc = None
        self.binned_lc = None
        self.smooth_lc = None

        self.phase = None
        self.flux = None
        self.phase_bins = None
        self.flux_binned = None
        self.flux_smooth = None

        self.estimated_params = {}
        self.accuracy = {}

        self._run()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @staticmethod
    def _value(x):
        return x.value if hasattr(x, "value") else x

    @staticmethod
    def _safe_float(x):
        try:
            x = BLS._value(x)
            return float(x)
        except Exception:
            return np.nan

    @staticmethod
    def _accuracy_percent(estimated, catalog):
        estimated = BLS._safe_float(estimated)
        catalog = BLS._safe_float(catalog)

        if not np.isfinite(estimated) or not np.isfinite(catalog) or catalog == 0:
            return np.nan

        rel_error = abs(estimated - catalog) / abs(catalog)
        return float(np.clip(100 * (1 - rel_error), 0, 100))

    def _valid_odd_window(self, n):
        if n < 5:
            raise ValueError(f"Too few points to flatten TIC {self.tic_id}")

        window = min(self.window_length, n - 1)

        if window % 2 == 0:
            window -= 1

        if window < 5:
            window = 5

        if window >= n:
            window = n - 1 if (n - 1) % 2 == 1 else n - 2

        return max(window, 5)

    def _valid_savgol_window(self, n):
        window = min(self.smooth_window, n - 1)

        if window % 2 == 0:
            window -= 1

        min_window = self.smooth_polyorder + 3

        if min_window % 2 == 0:
            min_window += 1

        if window < min_window:
            window = min_window

        if window >= n:
            window = n - 1 if (n - 1) % 2 == 1 else n - 2

        return max(window, min_window)

    def _save_plot(self, filename):
        os.makedirs(self.plot_dir, exist_ok=True)
        path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print("Saved plot:", path)

    # ------------------------------------------------------------
    # BLS search
    # ------------------------------------------------------------

    def _make_bls_grid(self):
        time = np.asarray(self.lc_flat_initial.time.value, dtype=float)
        time = time[np.isfinite(time)]

        if len(time) < 10:
            raise ValueError(f"Not enough valid time points for TIC {self.tic_id}")

        time_span = np.nanmax(time) - np.nanmin(time)
        max_period = min(self.max_period, time_span / 2)

        if max_period <= self.min_period:
            raise ValueError(
                f"Observation baseline is too short for this period range. "
                f"time_span={time_span:.3f} days, "
                f"min_period={self.min_period}, "
                f"max_period={max_period:.3f}. "
                "Try lowering min_period or use more sectors."
            )

        period_grid = np.linspace(
            self.min_period,
            max_period,
            self.n_periods
        )

        max_duration = min(
            self.max_duration,
            0.5 * self.min_period
        )

        if max_duration <= self.min_duration:
            raise ValueError(
                f"Invalid duration grid for TIC {self.tic_id}: "
                f"min_duration={self.min_duration}, "
                f"max_duration={max_duration}"
            )

        duration_grid = np.linspace(
            self.min_duration,
            max_duration,
            self.n_durations
        )

        if self.verbose:
            print("Period range:", period_grid.min(), "to", period_grid.max(), "days")
            print("Duration range:", duration_grid.min(), "to", duration_grid.max(), "days")
            print("Max duration < min period:", duration_grid.max() < period_grid.min())

        return period_grid, duration_grid

    def _run_bls(self):
        period_grid, duration_grid = self._make_bls_grid()

        self.bls = self.lc_flat_initial.to_periodogram(
            method="bls",
            period=period_grid,
            duration=duration_grid,
            frequency_factor=self.frequency_factor
        )

        self.period = self.bls.period_at_max_power
        self.t0 = self.bls.transit_time_at_max_power
        self.duration = self.bls.duration_at_max_power

        if hasattr(self.bls, "depth_at_max_power"):
            bls_depth = self.bls.depth_at_max_power
        else:
            best_idx = np.nanargmax(np.asarray(self.bls.power, dtype=float))
            bls_depth = np.asarray(self.bls.depth, dtype=float)[best_idx]

        bls_depth = self._safe_float(bls_depth)

        self.estimated_params.update({
            "period_days": self._safe_float(self.period),
            "t0": self._safe_float(self.t0),
            "duration_days": self._safe_float(self.duration),
            "duration_hours": self._safe_float(self.duration) * 24,
            "bls_depth_fraction": bls_depth,
            "bls_depth_ppm": bls_depth * 1e6,
            "bls_depth_percent": bls_depth * 100,
            "bls_max_power": self._safe_float(self.bls.max_power),
            "bls_sde": self._compute_bls_sde()
        })

        self.transit_mask = self.bls.get_transit_mask(
            period=self.period,
            transit_time=self.t0,
            duration=self.duration
        )

        self.transit_mask = np.asarray(self.transit_mask, dtype=bool)

        if self.verbose:
            print("Best BLS period:", self.period)
            print("Best transit time:", self.t0)
            print("Best duration:", self.duration)
            print("BLS depth ppm:", self.estimated_params["bls_depth_ppm"])
            print("BLS SDE:", self.estimated_params["bls_sde"])

    def _compute_bls_sde(self):
        power = np.asarray(self.bls.power, dtype=float)

        if np.nanstd(power) == 0:
            return np.nan

        return float(
            (np.nanmax(power) - np.nanmedian(power)) / np.nanstd(power)
        )

    # ------------------------------------------------------------
    # Re-flatten using BLS transit mask
    # ------------------------------------------------------------

    def _refine_flattening_with_mask(self):
        window = self._valid_odd_window(len(self.lc_flat_initial))

        self.lc_flat, self.trend = self.lc_flat_initial.flatten(
            window_length=window,
            polyorder=self.polyorder,
            mask=self.transit_mask,
            return_trend=True
        )

        self.lc_flat = (
            self.lc_flat
            .remove_nans()
            .normalize(unit="unscaled")
        )

        self.estimated_params["final_flatten_window"] = window

    # ------------------------------------------------------------
    # Fold, bin, smooth
    # ------------------------------------------------------------

    def _fold_lightcurve(self):
        self.folded_lc = self.lc_flat.fold(
            period=self.period,
            epoch_time=self.t0,
            normalize_phase=True
        ).remove_nans()

        self.phase = np.asarray(self.folded_lc.time.value, dtype=float)
        self.flux = np.asarray(self.folded_lc.flux.value, dtype=float)

        valid = np.isfinite(self.phase) & np.isfinite(self.flux)
        self.phase = self.phase[valid]
        self.flux = self.flux[valid]

        order = np.argsort(self.phase)
        self.phase = self.phase[order]
        self.flux = self.flux[order]

    def _bin_folded_lightcurve(self):
        bin_edges = np.linspace(-0.5, 0.5, self.n_bins + 1)
        self.phase_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        flux_binned = np.full(self.n_bins, np.nan)
        bin_idx = np.digitize(self.phase, bin_edges) - 1

        for i in range(self.n_bins):
            mask = bin_idx == i
            if np.any(mask):
                flux_binned[i] = np.nanmedian(self.flux[mask])

        finite = np.isfinite(flux_binned)

        if finite.sum() < 2:
            raise ValueError(f"Not enough valid folded bins for TIC {self.tic_id}")

        self.flux_binned = np.interp(
            self.phase_bins,
            self.phase_bins[finite],
            flux_binned[finite]
        )

        self.binned_lc = lk.LightCurve(
            time=self.phase_bins,
            flux=self.flux_binned
        )

    def _smooth_binned_lightcurve(self):
        window = self._valid_savgol_window(len(self.flux_binned))

        self.flux_smooth = savgol_filter(
            self.flux_binned,
            window_length=window,
            polyorder=self.smooth_polyorder,
            mode="interp"
        )

        self.smooth_lc = lk.LightCurve(
            time=self.phase_bins,
            flux=self.flux_smooth
        )

        self.smooth_lc.meta["TICID"] = self.tic_id
        self.smooth_lc.meta["PERIOD"] = self.estimated_params["period_days"]
        self.smooth_lc.meta["T0"] = self.estimated_params["t0"]
        self.smooth_lc.meta["DURDAYS"] = self.estimated_params["duration_days"]

        self.estimated_params["smooth_window"] = window
        self.estimated_params["smooth_polyorder"] = self.smooth_polyorder

    # ------------------------------------------------------------
    # Trapezoid fit
    # ------------------------------------------------------------

    @staticmethod
    def trapezoid_transit_model(t, baseline, depth, t_center, t_tot, ingress_fraction):
        t_in = ingress_fraction * t_tot
        half_total = 0.5 * t_tot
        half_flat = half_total - t_in

        x = np.abs(t - t_center)
        model = np.full_like(t, baseline, dtype=float)

        flat = x <= half_flat
        model[flat] = baseline - depth

        ramp = (x > half_flat) & (x < half_total)

        if t_in > 0 and np.any(ramp):
            ramp_depth = depth * (half_total - x[ramp]) / t_in
            model[ramp] = baseline - ramp_depth

        return model

    def _fit_trapezoid_to_smooth_curve(
        self,
        window_multiplier=5,
        min_points=30
    ):
        phase = np.asarray(self.phase_bins, dtype=float)
        flux = np.asarray(self.flux_smooth, dtype=float)

        valid = np.isfinite(phase) & np.isfinite(flux)
        phase = phase[valid]
        flux = flux[valid]

        period_days = self.estimated_params["period_days"]
        duration_days = self.estimated_params["duration_days"]

        t_days = phase * period_days

        fit_window_days = min(
            0.45 * period_days,
            max(window_multiplier * duration_days, 0.25)
        )

        fit_mask = np.abs(t_days) < fit_window_days

        t_fit = t_days[fit_mask]
        f_fit = flux[fit_mask]

        if len(t_fit) < min_points:
            raise ValueError(
                f"Not enough points for trapezoid fit. Found {len(t_fit)}."
            )

        order = np.argsort(t_fit)
        t_fit = t_fit[order]
        f_fit = f_fit[order]

        oot_mask = np.abs(t_fit) > 1.5 * duration_days

        if oot_mask.sum() >= 10:
            baseline0 = np.nanmedian(f_fit[oot_mask])
        else:
            baseline0 = np.nanmedian(f_fit)

        central_mask = np.abs(t_fit) < 0.5 * duration_days

        if central_mask.sum() >= 5:
            bottom0 = np.nanpercentile(f_fit[central_mask], 10)
        else:
            bottom0 = np.nanpercentile(f_fit, 5)

        depth0 = baseline0 - bottom0

        if not np.isfinite(depth0) or depth0 <= 0:
            depth0 = max(np.nanstd(f_fit), 1e-5)

        scatter = 1.4826 * np.nanmedian(
            np.abs(f_fit - np.nanmedian(f_fit))
        )

        if not np.isfinite(scatter) or scatter <= 0:
            scatter = np.nanstd(f_fit)

        if not np.isfinite(scatter) or scatter <= 0:
            scatter = 1e-4

        cadence_days = np.nanmedian(np.diff(np.sort(t_days)))

        if not np.isfinite(cadence_days) or cadence_days <= 0:
            cadence_days = duration_days / 20

        min_t_tot = max(2 * cadence_days, 0.1 * duration_days)
        max_t_tot = min(0.8 * period_days, 5.0 * duration_days)

        lower = np.array([
            baseline0 - 10 * scatter,
            0.0,
            -0.5 * duration_days,
            min_t_tot,
            0.01
        ])

        upper = np.array([
            baseline0 + 10 * scatter,
            max(0.5, 20 * depth0),
            0.5 * duration_days,
            max_t_tot,
            0.49
        ])

        p0 = np.array([
            baseline0,
            depth0,
            0.0,
            np.clip(duration_days, min_t_tot * 1.01, max_t_tot * 0.99),
            0.15
        ])

        p0 = np.maximum(p0, lower + 1e-12)
        p0 = np.minimum(p0, upper - 1e-12)

        def residuals(p):
            model = self.trapezoid_transit_model(t_fit, *p)
            return (f_fit - model) / scatter

        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=20000
        )

        baseline, depth, t_center, t_tot, ingress_fraction = result.x

        t_ingress = ingress_fraction * t_tot
        t_flat = t_tot - 2 * t_ingress
        flat_bottom_flux = baseline - depth

        depth_fraction = depth / baseline if baseline != 0 else np.nan
        depth_ppm = depth_fraction * 1e6

        self.estimated_params.update({
            "period_days": float(period_days),
            "t0": self.estimated_params["t0"],
            "duration_days": float(t_tot),
            "duration_hours": float(t_tot * 24),
            "transit_depth_flux": float(depth),
            "transit_depth_fraction": float(depth_fraction),
            "transit_depth_ppm": float(depth_ppm),
            "baseline_flux": float(baseline),
            "flat_bottom_flux": float(flat_bottom_flux),
            "T_ingress_days": float(t_ingress),
            "T_ingress_hours": float(t_ingress * 24),
            "T_flat_days": float(t_flat),
            "T_flat_hours": float(t_flat * 24),
            "transit_center_offset_days": float(t_center),
            "transit_center_offset_hours": float(t_center * 24),
            "fit_success": bool(result.success),
            "fit_cost": float(result.cost),
            "fit_message": str(result.message),
            "n_fit_points": int(len(t_fit))
        })

        self.fit_result = result
        self.fit_t = t_fit
        self.fit_flux = f_fit

    # ------------------------------------------------------------
    # Accuracy against catalog df
    # ------------------------------------------------------------

    def _compute_accuracy(self):
        required = ["TIC ID", "Period (days)", "Duration (hours)", "Depth (ppm)"]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            self.accuracy = {
                "matched": False,
                "reason": f"Missing catalog columns: {missing}"
            }
            return

        df = self.df.copy()
        df["TIC ID"] = pd.to_numeric(df["TIC ID"], errors="coerce")

        rows = (
            df.loc[df["TIC ID"] == self.tic_id]
            .sort_values(["TIC ID", "Planet Num"] if "Planet Num" in df.columns else ["TIC ID"])
            .copy()
        )

        if len(rows) == 0:
            self.accuracy = {
                "matched": False,
                "reason": f"No catalog row found for TIC {self.tic_id}"
            }
            return

        est_period = self.estimated_params["period_days"]

        rows["_period_error"] = np.abs(
            pd.to_numeric(rows["Period (days)"], errors="coerce") - est_period
        )

        matched = rows.sort_values("_period_error").iloc[0]

        catalog_period = self._safe_float(matched["Period (days)"])
        catalog_duration = self._safe_float(matched["Duration (hours)"])
        catalog_depth = self._safe_float(matched["Depth (ppm)"])

        estimated_period = self.estimated_params["period_days"]
        estimated_duration = self.estimated_params["duration_hours"]
        estimated_depth = self.estimated_params["transit_depth_ppm"]

        self.accuracy = {
            "matched": True,
            "tic_id": self.tic_id,
            "matched_planet_num": matched.get("Planet Num", np.nan),
            "catalog_period_days": catalog_period,
            "estimated_period_days": estimated_period,
            "period_accuracy_percent": self._accuracy_percent(
                estimated_period,
                catalog_period
            ),
            "catalog_duration_hours": catalog_duration,
            "estimated_duration_hours": estimated_duration,
            "duration_accuracy_percent": self._accuracy_percent(
                estimated_duration,
                catalog_duration
            ),
            "catalog_depth_ppm": catalog_depth,
            "estimated_depth_ppm": estimated_depth,
            "depth_accuracy_percent": self._accuracy_percent(
                estimated_depth,
                catalog_depth
            )
        }

    # ------------------------------------------------------------
    # FITS save
    # ------------------------------------------------------------

    def _save_processed_fits(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        path = os.path.join(
            self.processed_dir,
            f"TIC_{self.tic_id}_smooth_lc.fits"
        )

        header = fits.Header()
        header["TICID"] = (self.tic_id, "TIC ID")
        header["PERIOD"] = (self.estimated_params["period_days"], "Period [days]")
        header["T0"] = (self.estimated_params["t0"], "Transit epoch")
        header["DURDAYS"] = (self.estimated_params["duration_days"], "Duration [days]")
        header["DURHRS"] = (self.estimated_params["duration_hours"], "Duration [hours]")
        header["DEPTH"] = (self.estimated_params["transit_depth_fraction"], "Transit depth fraction")
        header["DPPM"] = (self.estimated_params["transit_depth_ppm"], "Transit depth ppm")
        header["BASE"] = (self.estimated_params["baseline_flux"], "Baseline flux")
        header["FLATFLX"] = (self.estimated_params["flat_bottom_flux"], "Flat-bottom flux")
        header["TINGD"] = (self.estimated_params["T_ingress_days"], "Ingress duration [days]")
        header["TINGHR"] = (self.estimated_params["T_ingress_hours"], "Ingress duration [hours]")
        header["TFLATD"] = (self.estimated_params["T_flat_days"], "Flat-bottom duration [days]")
        header["TFLATH"] = (self.estimated_params["T_flat_hours"], "Flat-bottom duration [hours]")
        header["BLSSDE"] = (self.estimated_params["bls_sde"], "BLS SDE")
        header["BLSPOW"] = (self.estimated_params["bls_max_power"], "BLS max power")

        primary_hdu = fits.PrimaryHDU(header=header)

        table_hdu = fits.BinTableHDU.from_columns([
            fits.Column(
                name="PHASE",
                format="E",
                array=np.asarray(self.phase_bins, dtype="float32")
            ),
            fits.Column(
                name="FLUX_BIN",
                format="E",
                array=np.asarray(self.flux_binned, dtype="float32")
            ),
            fits.Column(
                name="FLUX_SMOOTH",
                format="E",
                array=np.asarray(self.flux_smooth, dtype="float32")
            )
        ])

        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(path, overwrite=True)

        self.estimated_params["processed_fits_path"] = path

        if self.verbose:
            print("Saved processed FITS:", path)

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    def _plot_all(self):
        period_days = self.estimated_params["period_days"]
        duration_days = self.estimated_params["duration_days"]
        duration_phase = duration_days / period_days
        zoom_width = min(0.5, max(0.05, 3 * duration_phase))

        # 1. BLS periodogram
        plt.figure(figsize=(10, 4))
        plt.plot(self.bls.period.value, self.bls.power)
        plt.axvline(
            period_days,
            linestyle="--", color="red",
            label=f"Best period = {period_days:.5f} d"
        )
        plt.xlabel("Period [days]")
        plt.ylabel("BLS power")
        plt.title(f"TIC {self.tic_id} BLS periodogram")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_bls_periodogram.png")

        # 2. Final flattened light curve with transit mask
        plt.figure(figsize=(12, 4))
        plt.scatter(
            self.lc_flat.time.value,
            self.lc_flat.flux.value,
            s=2,
            alpha=0.8,
            label="Final flattened LC"
        )

        if len(self.transit_mask) == len(self.lc_flat.time.value):
            plt.scatter(
                self.lc_flat.time.value[self.transit_mask],
                self.lc_flat.flux.value[self.transit_mask],
                s=8,
                label="Detected transit mask"
            )

        plt.xlabel("Time")
        plt.ylabel("Flattened flux")
        plt.title(f"TIC {self.tic_id} flattened LC with detected transit regions")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_flattened_with_transit_mask.png")

        # 3. Folded light curve
        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.folded_lc.time.value,
            self.folded_lc.flux.value,
            s=4,
            alpha=0.4
        )
        plt.xlabel("Phase")
        plt.ylabel("Normalized flux")
        plt.title(f"TIC {self.tic_id} folded LC | Period = {period_days:.5f} days")
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_folded_lc.png")

        # 4. Zoomed transit
        zoom_mask_raw = np.abs(self.phase) < zoom_width

        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.phase[zoom_mask_raw],
            self.flux[zoom_mask_raw],
            s=6,
            alpha=0.5
        )
        plt.xlabel("Phase")
        plt.ylabel("Normalized flux")
        plt.title(f"TIC {self.tic_id} zoomed transit candidate")
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_zoomed_transit.png")

        # 5. Binned and smoothed full phase curve
        plt.figure(figsize=(9, 5))
        plt.scatter(
            self.phase,
            self.flux,
            s=3,
            alpha=0.15,
            label="Folded raw points"
        )
        plt.plot(
            self.phase_bins,
            self.flux_binned,
            linewidth=1,color="red",
            label="Median-binned"
        )
        plt.plot(
            self.phase_bins,
            self.flux_smooth,
            linewidth=2,color="orange",
            label="Smoothed"
        )
        plt.xlabel("Phase")
        plt.ylabel("Normalized flux")
        plt.title(f"TIC {self.tic_id} phase-folded LC: binned and smoothed")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_binned_smoothed_phase_curve.png")

        # 6. Zoomed denoised transit region
        zoom_mask_bins = np.abs(self.phase_bins) < zoom_width

        plt.figure(figsize=(8, 5))
        plt.plot(
            self.phase_bins[zoom_mask_bins],
            self.flux_binned[zoom_mask_bins],
            linewidth=1,color="red",
            label="Median-binned"
        )
        plt.plot(
            self.phase_bins[zoom_mask_bins],
            self.flux_smooth[zoom_mask_bins],
            linewidth=2,color="orange",
            label="Smoothed"
        )
        plt.xlabel("Phase")
        plt.ylabel("Normalized flux")
        plt.title(f"TIC {self.tic_id} denoised transit region")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_denoised_transit_region.png")

        # 7. Trapezoid fit
        if hasattr(self, "fit_result"):
            dense_t = np.linspace(self.fit_t.min(), self.fit_t.max(), 2000)
            dense_model = self.trapezoid_transit_model(
                dense_t,
                *self.fit_result.x
            )

            plt.figure(figsize=(10, 5))
            plt.scatter(
                self.fit_t * 24,
                self.fit_flux,
                s=10,
                alpha=0.5,
                label="Smoothed fit data"
            )
            plt.plot(
                dense_t * 24,
                dense_model,
                linewidth=2,color="green",
                label="Trapezoid fit"
            )
            plt.axhline(
                self.estimated_params["baseline_flux"],
                linestyle="--",
                linewidth=1,
                color="black",
                label="Baseline"
            )
            plt.axhline(
                self.estimated_params["flat_bottom_flux"],
                linestyle="--",
                linewidth=1,
                color="black",
                label="Flat bottom"
            )
            plt.xlabel("Time from transit center [hours]")
            plt.ylabel("Normalized flux")
            plt.title(f"TIC {self.tic_id} trapezoid fit")
            plt.legend()
            plt.grid(alpha=0.3)
            self._save_plot(f"TIC_{self.tic_id}_trapezoid_fit.png")

    # ------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------

    def _run(self):
        self._run_bls()
        self._refine_flattening_with_mask()
        self._fold_lightcurve()
        self._bin_folded_lightcurve()
        self._smooth_binned_lightcurve()
        self._fit_trapezoid_to_smooth_curve()
        self._compute_accuracy()

        if self.save_fits:
            self._save_processed_fits()

        if self.Plot:
            self._plot_all()

    def get_outputs(self):
        return self.smooth_lc, self.accuracy