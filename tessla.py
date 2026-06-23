import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from data_loader import label_loader, lc_downloader
from detector import BLS

warnings.filterwarnings("ignore")

base_dir = os.getcwd()

TIC_ID = "231663901" #Try various TIC IDs to test the pipeline including for different types of trnsits in the TFOPWG Disposition 

df = pd.read_csv(os.path.join(base_dir, "tois.csv"))

loader = label_loader(df, plot=True, save_labels=True)
df, download_targets, transit_labels = loader.get_outputs()

print("Cleaned DataFrame shape:", df.shape)

loader = lc_downloader(
    TIC_ID=TIC_ID,
    df=download_targets,
    Plot=True,
    Save_labels=True,
    stitch=False,           # whether to stitch multiple sectors together
    single_index=0,         # product index from Lightkurve search results, not df index
    download_root="raw_fits"
)

lc_flat = loader.get_outputs()


# raw_lc = loader.raw_lc
# pdc_lc = loader.pdc_lc
# clean_lc = loader.clean_lc
# trend_lc = loader.trend_lc
# info = loader.info

detector = BLS(
    TIC_ID=TIC_ID,
    lc_flat=lc_flat,
    df=transit_labels,
    Plot=True,
    save_fits=True,
    base_dir="."
)

smooth_lc, accuracy = detector.get_outputs()
params = detector.estimated_params

print(params)
print(accuracy)