import os
import re
import shutil
import tempfile
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt


class label_loader:
    COLUMNS = [
        'TIC ID', 'RA', 'Dec', 'TOI', 'TESS Mag', 'TESS Mag err',
        'Planet Num', 'TFOPWG Disposition',
        'Period (days)', 'Period (days) err',
        'Duration (hours)', 'Duration (hours) err',
        'Depth (ppm)', 'Depth (ppm) err',
        'Planet Radius (R_Earth)', 'Planet Radius (R_Earth) err',
        'Planet SNR',
        'Stellar Radius (R_Sun)', 'Stellar Radius (R_Sun) err',
        'Sectors', 'Comments', 'Detection'
    ]

    DISTRIBUTION_COLS = [
        'TESS Mag',
        'Period (days)',
        'Duration (hours)',
        'Depth (ppm)'
    ]

    def __init__(self, df, plot=True, save_labels=True):
        self.plot = plot
        self.save_labels = save_labels

        self.df_clean = self._clean_dataframe(df)
        self.download_targets = self._make_download_targets(self.df_clean)
        self.transit_labels = self._make_transit_labels(self.df_clean)

        if self.save_labels:
            self._save_labels()

        if self.plot:
            self._plot_label_statistics(self.df_clean)

    def _clean_dataframe(self, df):
        missing_cols = [col for col in self.COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df[self.COLUMNS].copy()

        mask = (
            df['TFOPWG Disposition'].isna() |
            df['Period (days)'].isna() |
            df['Detection'].isna() |
            (df['Duration (hours)'] == 0)
        )

        df_clean = df.loc[~mask].reset_index(drop=True)

        print("Clean dataframe shape:", df_clean.shape)

        return df_clean

    def _make_download_targets(self, df):
        download_cols = ["TIC ID", "Detection", "Sectors"]

        if "Authors" in df.columns:
            download_cols.append("Authors")

        download_targets = (
            df[download_cols]
            .drop_duplicates(subset=["TIC ID"])
            .reset_index(drop=True)
        )

        print("Total number of lightcurves:", len(download_targets))

        return download_targets

    def _make_transit_labels(self, df):
        transit_labels = (
            df[self.COLUMNS]
            .dropna(subset=["TIC ID", "Planet Num", "Period (days)"])
            .reset_index(drop=True)
        )

        transit_labels["sample_id"] = (
            "TIC_" + transit_labels["TIC ID"].astype(str) +
            "_PN_" + transit_labels["Planet Num"].astype(str)
        )

        transit_labels = transit_labels[
            ["sample_id"] + [col for col in transit_labels.columns if col != "sample_id"]
        ]

        return transit_labels

    def _save_labels(self):
        self.download_targets.to_csv("download_targets.csv", index=False)
        self.transit_labels.to_csv("transit_labels.csv", index=False)

        print("Saved: download_targets.csv")
        print("Saved: transit_labels.csv")

    def _save_current_plot(self, filename):
        plot_dir = os.path.join("plots", "labels statistics")
        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_label_statistics(self, plot_df):
        for col in self.DISTRIBUTION_COLS:
            plt.figure(figsize=(7, 5))
            plt.hist(plot_df[col].dropna(), bins=40)
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"Distribution of {col}")

            if col in ["Period (days)", "Depth (ppm)"]:
                plt.xscale("log")

            plt.grid(True, alpha=0.3)
            self._save_current_plot(f"{col.replace(' ', '_').replace('/', '_')}_distribution.png")

        plt.figure(figsize=(7, 5))
        plot_df["Planet Num"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("Planet Num")
        plt.ylabel("Count")
        plt.title("Distribution of Planet Numbers")
        plt.grid(True, axis="y", alpha=0.3)
        self._save_current_plot("planet_num_distribution.png")

        plt.figure(figsize=(8, 5))
        plot_df["TFOPWG Disposition"].value_counts(dropna=False).plot(kind="bar")
        plt.xlabel("TFOPWG Disposition")
        plt.ylabel("Count")
        plt.title("Counts by TFOPWG Disposition")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        self._save_current_plot("tfopwg_disposition_counts.png")

        plt.figure(figsize=(8, 6))
        for disposition, group in plot_df.groupby("TFOPWG Disposition"):
            plt.scatter(
                group["Period (days)"],
                group["Depth (ppm)"],
                alpha=0.6,
                label=disposition
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Period (days)")
        plt.ylabel("Depth (ppm)")
        plt.title("Period vs Transit Depth by Disposition")
        plt.legend(title="Disposition", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        self._save_current_plot("period_vs_depth_by_disposition.png")

        plt.figure(figsize=(8, 6))
        for disposition, group in plot_df.groupby("TFOPWG Disposition"):
            plt.scatter(
                group["TESS Mag"],
                group["Depth (ppm)"],
                alpha=0.6,
                label=disposition
            )

        plt.yscale("log")
        plt.xlabel("TESS Mag")
        plt.ylabel("Depth (ppm)")
        plt.title("TESS Magnitude vs Transit Depth by Disposition")
        plt.gca().invert_xaxis()
        plt.legend(title="Disposition", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        self._save_current_plot("tess_mag_vs_depth_by_disposition.png")

        print("Saved plots to: plots/labels statistics/")

    def get_outputs(self):
        return self.df_clean, self.download_targets, self.transit_labels
    

class lc_downloader:
    def __init__(
        self,
        TIC_ID="231663901",
        df=None,
        Plot=True,
        Save_labels=True,
        stitch=True,
        single_index=0,
        download_root="raw_fits",
        plot_root=os.path.join("plots", "lc_plots"),
        exptimes=(20, 120),
        window_length=401,
        polyorder=2,
        verbose=True
    ):
        """
        Parameters
        ----------
        TIC_ID : str or int
            TIC ID to search and download.

        df : pandas.DataFrame
            Usually download_targets dataframe with columns:
            TIC ID, Detection, Sectors.

        Plot : bool
            If True, saves raw SAP, PDCSAP-cleaned, and flattened plots.

        Save_labels : bool
            Kept to match your requested interface.
            Here it controls whether final SAP and PDCSAP FITS files are saved.

        stitch : bool
            If True, stitches all available products.
            If False, uses only one product from the LightCurveCollection.

        single_index : int
            Product index to use when stitch=False.

        Returns
        -------
        Use:
            flat_lc = loader.get_outputs()
        """

        self.tic_id = int(TIC_ID)
        self.df = df
        self.Plot = Plot
        self.Save_labels = Save_labels
        self.stitch = stitch
        self.single_index = single_index
        self.download_root = download_root
        self.plot_root = plot_root
        self.exptimes = exptimes
        self.window_length = window_length
        self.polyorder = polyorder
        self.verbose = verbose

        os.makedirs(self.download_root, exist_ok=True)

        self.raw_lc = None
        self.pdc_lc = None
        self.clean_lc = None
        self.flat_lc = None
        self.trend_lc = None
        self.info = {}

        self._run()

    # ------------------------------------------------------------
    # Sector / author helpers
    # ------------------------------------------------------------

    @staticmethod
    def parse_sectors(value):
        if pd.isna(value):
            return None

        sectors = [int(x) for x in re.findall(r"\d+", str(value))]
        return sectors if sectors else None

    @staticmethod
    def authors_from_detection(detection):
        if pd.isna(detection):
            return [None]

        detection = str(detection).strip().upper()

        if detection == "SPOC":
            return ["SPOC"]
        if detection == "QLP":
            return ["QLP"]
        if detection == "SPOC/QLP":
            return ["SPOC", "QLP"]
        if detection == "SPOC/FAINT":
            return ["SPOC", None]
        if detection in ["FAINT", "CTOI", "UNKNOWN"]:
            return [None]

        return [None]

    def _get_target_metadata_from_df(self):
        detection = None
        sectors = None

        if self.df is None:
            return detection, sectors

        if "TIC ID" not in self.df.columns:
            return detection, sectors

        tic_numeric = pd.to_numeric(self.df["TIC ID"], errors="coerce")
        rows = self.df.loc[tic_numeric == self.tic_id]

        if len(rows) == 0:
            return detection, sectors

        row = rows.iloc[0]

        if "Detection" in self.df.columns:
            detection = row["Detection"]

        if "Sectors" in self.df.columns:
            sectors = self.parse_sectors(row["Sectors"])

        return detection, sectors

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------

    def _search_lightcurves(self, detection=None, sectors=None):
        target = f"TIC {self.tic_id}"
        authors_to_try = self.authors_from_detection(detection)

        for exptime in self.exptimes:
            if self.verbose:
                print(f"\nSearching {target} | exptime={exptime}")

            for author in authors_to_try:
                kwargs = {
                    "target": target,
                    "mission": "TESS",
                    "exptime": exptime,
                }

                if author is not None:
                    kwargs["author"] = author

                if sectors is not None:
                    kwargs["sector"] = sectors

                sr = lk.search_lightcurve(**kwargs)

                if self.verbose:
                    print(f"author={author} | products found={len(sr)}")

                if len(sr) > 0:
                    return sr, exptime, author

            kwargs = {
                "target": target,
                "mission": "TESS",
                "exptime": exptime,
            }

            if sectors is not None:
                kwargs["sector"] = sectors

            sr = lk.search_lightcurve(**kwargs)

            if self.verbose:
                print(f"all authors | products found={len(sr)}")

            if len(sr) > 0:
                return sr, exptime, None

        return None, None, None

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------

    def _download_collection(self, search_result, flux_column, temp_dir):
        lcc = search_result.download_all(
            flux_column=flux_column,
            download_dir=temp_dir
        )

        if lcc is None or len(lcc) == 0:
            raise ValueError(
                f"Download failed for TIC {self.tic_id} using {flux_column}"
            )

        if self.stitch:
            lc = lcc.stitch(
                corrector_func=lambda x: (
                    x.remove_nans()
                     .normalize(unit="unscaled")
                )
            )
            mode = "stitched"
        else:
            idx = min(self.single_index, len(lcc) - 1)
            lc = lcc[idx]
            mode = f"single_{idx}"

        return lc, mode, len(lcc)

    def _download_lightcurves(self, search_result, used_exptime, used_author):
        temp_dir = tempfile.mkdtemp(prefix=f"TIC_{self.tic_id}_temp_")

        try:
            if self.verbose:
                print("\nDownloading temporarily to:", temp_dir)

            # ------------------------------------------------------------
            # Download SAP flux: this is the only one saved permanently
            # ------------------------------------------------------------
            self.raw_lc, mode, n_products = self._download_collection(
                search_result=search_result,
                flux_column="sap_flux",
                temp_dir=temp_dir
            )

            # ------------------------------------------------------------
            # Download PDCSAP flux temporarily for preprocessing only
            # ------------------------------------------------------------
            try:
                self.pdc_lc, _, _ = self._download_collection(
                    search_result=search_result,
                    flux_column="pdcsap_flux",
                    temp_dir=temp_dir
                )
            except Exception as err:
                print(
                    f"Warning: PDCSAP download failed for TIC {self.tic_id}. "
                    f"Falling back to SAP flux. Reason: {err}"
                )
                self.pdc_lc = self.raw_lc.copy()

            self.info = {
                "tic_id": self.tic_id,
                "used_exptime": used_exptime,
                "used_author": used_author,
                "n_products": n_products,
                "mode": mode,
                "stitch": self.stitch,
            }

            # ------------------------------------------------------------
            # Save only SAP raw FITS
            # ------------------------------------------------------------
            if self.Save_labels:
                sap_path = os.path.join(
                    self.download_root,
                    f"TIC_{self.tic_id}_{mode}_{used_exptime}s_sap.fits"
                )

                self.raw_lc.to_fits(path=sap_path, overwrite=True)
                self.info["sap_path"] = sap_path

                if self.verbose:
                    print("Saved SAP raw light curve to:", sap_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

            if self.verbose:
                print("Deleted temporary downloaded files.")

    # ------------------------------------------------------------
    # Preprocessor
    # ------------------------------------------------------------

    def _make_valid_window_length(self, lc):
        n = len(lc)

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

    def _preprocess_lightcurve(self):
        lc = (
            self.pdc_lc
            .remove_nans()
            .normalize(unit="unscaled")
            .remove_outliers(
                sigma_lower=float("inf"),
                sigma_upper=5
            )
        )

        self.clean_lc = lc

        window = self._make_valid_window_length(lc)

        self.flat_lc, self.trend_lc = lc.flatten(
            window_length=window,
            polyorder=self.polyorder,
            return_trend=True
        )

        self.flat_lc = (
            self.flat_lc
            .remove_nans()
            .normalize(unit="unscaled")
        )

        self.info["flatten_window_length"] = window
        self.info["flatten_polyorder"] = self.polyorder

        if self.verbose:
            print("Flatten window length:", window)

    # ------------------------------------------------------------
    # Plotter
    # ------------------------------------------------------------

    def _save_plot(self, filename):
        os.makedirs(self.plot_root, exist_ok=True)

        path = os.path.join(self.plot_root, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print("Saved plot:", path)

    def _plot_lightcurves(self):
        plt.figure(figsize=(12, 4))
        plt.scatter(
            self.raw_lc.time.value,
            self.raw_lc.flux.value,
            s=2
        )
        plt.xlabel("Time")
        plt.ylabel("SAP Flux")
        plt.title(f"TIC {self.tic_id} Raw SAP Light Curve")
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_raw_sap.png")

        plt.figure(figsize=(12, 4))
        plt.scatter(
            self.clean_lc.time.value,
            self.clean_lc.flux.value,
            s=2
        )
        plt.xlabel("Time")
        plt.ylabel("Normalized PDCSAP Flux")
        plt.title(f"TIC {self.tic_id} Cleaned PDCSAP Light Curve")
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_cleaned_pdcsap.png")

        plt.figure(figsize=(12, 4))
        plt.scatter(
            self.flat_lc.time.value,
            self.flat_lc.flux.value,
            s=2,
            label="Flattened LC"
        )
        plt.plot(
            self.trend_lc.time.value,
            self.trend_lc.flux.value,
            linewidth=2, color="orange",
            label="Trend"
        )
        plt.xlabel("Time")
        plt.ylabel("Flattened Flux")
        plt.title(f"TIC {self.tic_id} Flattened Light Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_plot(f"TIC_{self.tic_id}_flattened_with_trend.png")

    # ------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------

    def _run(self):
        detection, sectors = self._get_target_metadata_from_df()

        search_result, used_exptime, used_author = self._search_lightcurves(
            detection=detection,
            sectors=sectors
        )

        if search_result is None or len(search_result) == 0:
            raise ValueError(f"No light curves found for TIC {self.tic_id}")

        self._download_lightcurves(
            search_result=search_result,
            used_exptime=used_exptime,
            used_author=used_author
        )

        self._preprocess_lightcurve()

        if self.Plot:
            self._plot_lightcurves()

    def get_outputs(self):
        return self.flat_lc