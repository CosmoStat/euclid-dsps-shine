"""Column metadata for the CosmoHub catalog used by this project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogColumn:
    """Human-readable metadata for one catalog column."""

    name: str
    source_name: str
    group: str
    dtype: str
    unit: str
    description: str
    notes: str = ""


CATALOG_353_URL = "https://cosmohub.pic.es/catalogs/353"
CATALOG_353_TABLE = "euclid_fs2_mock_dr_v1_1_phz"


CATALOG_COLUMNS: tuple[CatalogColumn, ...] = (
    CatalogColumn(
        "ra_gal",
        "ra_gal",
        "coordinates",
        "float",
        "deg",
        "Galaxy right ascension.",
        "ICRS-like sky coordinate used for spatial selection.",
    ),
    CatalogColumn(
        "dec_gal",
        "dec_gal",
        "coordinates",
        "float",
        "deg",
        "Galaxy declination.",
        "ICRS-like sky coordinate used for spatial selection.",
    ),
    CatalogColumn(
        "ra_mag_gal",
        "ra_mag_gal",
        "coordinates",
        "float",
        "deg",
        "Right ascension associated with the photometry/magnitude position.",
    ),
    CatalogColumn(
        "dec_mag_gal",
        "dec_mag_gal",
        "coordinates",
        "float",
        "deg",
        "Declination associated with the photometry/magnitude position.",
    ),
    CatalogColumn(
        "z_true",
        "true_redshift_halo",
        "redshift",
        "float",
        "dimensionless",
        "True halo redshift, aliased by the download query.",
        "Used only for diagnostics in the default workflow.",
    ),
    CatalogColumn(
        "z_phz",
        "phz_mode_1",
        "redshift",
        "float",
        "dimensionless",
        "First mode of the photometric-redshift probability distribution.",
        "Used as fixed DSPS z_obs by the default config.",
    ),
    CatalogColumn(
        "euclid_vis",
        "euclid_vis",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated Euclid VIS flux density.",
        "Project config interprets this as Fnu in cgs units.",
    ),
    CatalogColumn(
        "euclid_nisp_y",
        "euclid_nisp_y",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated Euclid NISP Y-band flux density.",
    ),
    CatalogColumn(
        "euclid_nisp_j",
        "euclid_nisp_j",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated Euclid NISP J-band flux density.",
    ),
    CatalogColumn(
        "euclid_nisp_h",
        "euclid_nisp_h",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated Euclid NISP H-band flux density.",
    ),
    CatalogColumn(
        "lsst_u",
        "lsst_u",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST u-band flux density.",
    ),
    CatalogColumn(
        "lsst_g",
        "lsst_g",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST g-band flux density.",
    ),
    CatalogColumn(
        "lsst_r",
        "lsst_r",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST r-band flux density.",
    ),
    CatalogColumn(
        "lsst_i",
        "lsst_i",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST i-band flux density.",
    ),
    CatalogColumn(
        "lsst_z",
        "lsst_z",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST z-band flux density.",
    ),
    CatalogColumn(
        "lsst_y",
        "lsst_y",
        "photometry",
        "float",
        "erg s^-1 cm^-2 Hz^-1",
        "Simulated LSST y-band flux density.",
    ),
    CatalogColumn(
        "metallicity_true",
        "metallicity",
        "physical truth",
        "float",
        "12 + log10(O/H)",
        "Gas-phase oxygen abundance, aliased by the download query.",
        "Reports convert to log10 metallicity proxy with offset -10.61.",
    ),
    CatalogColumn(
        "log_sfr_true",
        "log_sfr",
        "physical truth",
        "float",
        "log10(Msun yr^-1)",
        "Base-10 logarithm of catalog star-formation rate.",
    ),
    CatalogColumn(
        "sfr_true",
        "POW(10, log_sfr)",
        "physical truth",
        "double",
        "Msun yr^-1",
        "Linear star-formation rate computed by the download query.",
    ),
    CatalogColumn(
        "ebv_cosmos_1",
        "ebv_cosmos_1",
        "dust",
        "float",
        "mag",
        "COSMOS E(B-V) component 1.",
    ),
    CatalogColumn(
        "ebv_cosmos_2",
        "ebv_cosmos_2",
        "dust",
        "float",
        "mag",
        "COSMOS E(B-V) component 2.",
    ),
    CatalogColumn(
        "ext_curve_cosmos_1",
        "ext_curve_cosmos_1",
        "dust",
        "int8",
        "code",
        "Extinction-curve identifier for dust component 1.",
    ),
    CatalogColumn(
        "ext_curve_cosmos_2",
        "ext_curve_cosmos_2",
        "dust",
        "int8",
        "code",
        "Extinction-curve identifier for dust component 2.",
    ),
    CatalogColumn(
        "mw_extinction",
        "mw_extinction",
        "dust",
        "float",
        "mag",
        "Milky Way foreground extinction scalar.",
    ),
    CatalogColumn(
        "dust_ebv_true",
        "SQL CASE expression",
        "dust",
        "float",
        "mag",
        "Single intrinsic E(B-V) target computed by the download query.",
        "Bulge-fraction weighted when both COSMOS dust components are present.",
    ),
    CatalogColumn(
        "bulge_fraction",
        "bulge_fraction",
        "morphology",
        "float",
        "fraction",
        "Fraction of galaxy light assigned to the bulge component.",
    ),
    CatalogColumn(
        "disk_r50",
        "disk_r50",
        "morphology",
        "float",
        "arcsec",
        "Disk half-light radius.",
    ),
    CatalogColumn(
        "bulge_r50",
        "bulge_r50",
        "morphology",
        "float",
        "arcsec",
        "Bulge half-light radius.",
    ),
    CatalogColumn(
        "eps1_gal",
        "eps1_gal",
        "morphology",
        "float",
        "dimensionless",
        "Galaxy ellipticity component e1.",
    ),
    CatalogColumn(
        "eps2_gal",
        "eps2_gal",
        "morphology",
        "float",
        "dimensionless",
        "Galaxy ellipticity component e2.",
    ),
    CatalogColumn(
        "disk_ellipticity",
        "disk_ellipticity",
        "morphology",
        "float",
        "dimensionless",
        "Disk ellipticity.",
    ),
    CatalogColumn(
        "bulge_ellipticity",
        "bulge_ellipticity",
        "morphology",
        "float",
        "dimensionless",
        "Bulge ellipticity.",
    ),
    CatalogColumn(
        "bulge_nsersic",
        "bulge_nsersic",
        "morphology",
        "float",
        "dimensionless",
        "Bulge Sersic index.",
    ),
    CatalogColumn(
        "disk_nsersic",
        "disk_nsersic",
        "morphology",
        "float",
        "dimensionless",
        "Disk Sersic index.",
    ),
    CatalogColumn(
        "lm_halo",
        "lm_halo",
        "halo",
        "float",
        "log10(Msun h^-1)",
        "Base-10 logarithm of halo mass.",
    ),
    CatalogColumn(
        "lmbound_halo",
        "lmbound_halo",
        "halo",
        "float",
        "log10(Msun h^-1)",
        "Base-10 logarithm of bound halo mass.",
    ),
    CatalogColumn(
        "r_halo",
        "r_halo",
        "halo",
        "float",
        "kpc h^-1",
        "Halo radial distance in the lightcone.",
    ),
    CatalogColumn(
        "x_halo", "x_halo", "halo", "float", "Mpc h^-1", "Halo Cartesian x coordinate."
    ),
    CatalogColumn(
        "y_halo", "y_halo", "halo", "float", "Mpc h^-1", "Halo Cartesian y coordinate."
    ),
    CatalogColumn(
        "z_halo", "z_halo", "halo", "float", "Mpc h^-1", "Halo Cartesian z coordinate."
    ),
    CatalogColumn(
        "vx_halo",
        "vx_halo",
        "halo",
        "float",
        "km s^-1",
        "Halo peculiar velocity x component.",
    ),
    CatalogColumn(
        "vy_halo",
        "vy_halo",
        "halo",
        "float",
        "km s^-1",
        "Halo peculiar velocity y component.",
    ),
    CatalogColumn(
        "vz_halo",
        "vz_halo",
        "halo",
        "float",
        "km s^-1",
        "Halo peculiar velocity z component.",
    ),
    CatalogColumn(
        "n_sats_halo",
        "n_sats_halo",
        "halo",
        "int32",
        "count",
        "Number of satellite galaxies assigned to the halo.",
    ),
    CatalogColumn(
        "num_p_halo",
        "num_p_halo",
        "halo",
        "int32",
        "count",
        "Number of simulation particles associated with the halo.",
    ),
    CatalogColumn(
        "conc_vir_halo",
        "conc_vir_halo",
        "halo",
        "float",
        "dimensionless",
        "Virial concentration of the halo.",
    ),
    CatalogColumn(
        "rs_halo", "rs_halo", "halo", "float", "kpc h^-1", "Halo NFW scale radius."
    ),
    CatalogColumn(
        "rvir_halo", "rvir_halo", "halo", "float", "kpc h^-1", "Halo virial radius."
    ),
)


CATALOG_COLUMN_BY_NAME = {column.name: column for column in CATALOG_COLUMNS}


def known_catalog_columns() -> set[str]:
    """Return the canonical local column names documented for catalog 353."""
    return set(CATALOG_COLUMN_BY_NAME)
