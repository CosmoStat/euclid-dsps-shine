import os
import urllib.request
import numpy as np

from dsps import load_ssp_templates, load_transmission_curve
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_mag

# 1. Download Data
SSP_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5"
FILTER_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/filters/lsst_g_transmission.h5"

ssp_file = "ssp_data_fsps_v3.2_lgmet_age.h5"
filter_file = "lsst_g_transmission.h5"

if not os.path.exists(ssp_file):
    print(f"Downloading {ssp_file}...")
    urllib.request.urlretrieve(SSP_URL, ssp_file)

if not os.path.exists(filter_file):
    print(f"Downloading {filter_file}...")
    urllib.request.urlretrieve(FILTER_URL, filter_file)

# 2. Load Data
print("Loading SSP data and filter transmission...")
ssp_data = load_ssp_templates(fn=ssp_file)
lsst_g = load_transmission_curve(fn=filter_file)

# 3. Setup Galaxy Parameters
gal_t_table = np.linspace(0.05, 13.8, 100) # age of the universe in Gyr
gal_sfr_table = np.random.uniform(0, 10, gal_t_table.size) # SFR in Msun/yr
gal_lgmet = -2.0 # log10(Z)
gal_lgmet_scatter = 0.2 # lognormal scatter

z_obs = 0.5
t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)[0] # age of the universe in Gyr at z_obs

# 4. Calculate SED
print("Calculating galaxy rest-frame SED...")
sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
    gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,
    ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)

# 5. Calculate Photometry
print("Calculating rest-frame magnitude in LSST g-band...")
rest_mag = calc_rest_mag(
    ssp_data.ssp_wave, sed_info.rest_sed,
    lsst_g.wave, lsst_g.transmission)

print(f"Success! Calculated rest-frame magnitude: {rest_mag:.2f}")
