"""Generate a compact visual and LaTeX snippet for the Diffsky error model."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from euclid_dsps.photometric_uncertainty import (
    DEFAULT_MIN_SIGMA_FNU_CGS,
    DEFAULT_PHOTERR_SIGMA_SYS_MAG,
    default_m5_depth_error_model,
    effective_flux_sigma,
    flux_error_from_model,
)
from euclid_dsps.photometry import abmag_to_fnu_cgs
from euclid_dsps.reporting.core import configure_plot_style


COLORS = {
    "depth": "#2F5D8C",
    "systematic": "#B85C38",
    "catalog": "#3F7F5F",
    "floor": "#7A4E8A",
    "effective": "#C7352E",
}


def main() -> None:
    args = _parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = default_m5_depth_error_model()
    if args.sigma_sys_mag is not None:
        model["sigma_sys_mag"] = float(args.sigma_sys_mag)

    components = _compute_components(
        band=args.band,
        model=model,
        flux_floor_frac=float(args.floor_frac),
        min_flux_ratio=float(args.min_flux_ratio),
        max_flux_ratio=float(args.max_flux_ratio),
        n_grid=int(args.n_grid),
    )

    figure_base = out / "photerr_error_model_components"
    annotated_base = out / "photerr_error_model_annotated"
    _plot_components(components, figure_base)
    _plot_annotated_explainer(components, annotated_base)
    tex_path = out / "photerr_error_model_equations.tex"
    standalone_tex_path = out / "photerr_error_model_equations_standalone.tex"
    equations_pdf_path = out / "photerr_error_model_equations.pdf"
    md_path = out / "photerr_error_model_equations.md"
    summary_path = out / "photerr_error_model_summary.json"
    tex_path.write_text(_latex_snippet(components, figure_base.name), encoding="utf-8")
    standalone_tex_path.write_text(
        _standalone_latex_document("photerr_error_model_equations.tex"),
        encoding="utf-8",
    )
    _render_equations_pdf(components, equations_pdf_path)
    md_path.write_text(_markdown_snippet(components), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(components), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {figure_base.with_suffix('.png')}")
    print(f"Wrote {figure_base.with_suffix('.pdf')}")
    print(f"Wrote {figure_base.with_suffix('.svg')}")
    print(f"Wrote {annotated_base.with_suffix('.png')}")
    print(f"Wrote {annotated_base.with_suffix('.pdf')}")
    print(f"Wrote {annotated_base.with_suffix('.svg')}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {standalone_tex_path}")
    print(f"Wrote {equations_pdf_path}")
    print(f"Wrote {equations_pdf_path.with_suffix('.png')}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an explanatory PhotErr-style flux-error plot and "
            "LaTeX-ready equation snippet."
        )
    )
    parser.add_argument(
        "--out",
        default="outputs/reports/photerr_error_model_explainer",
        help="Output directory for the figure, TeX, Markdown, and JSON summary.",
    )
    parser.add_argument(
        "--band",
        default="lsst_i",
        help="Band to use for the component plot, e.g. lsst_i or roman_F146.",
    )
    parser.add_argument(
        "--floor-frac",
        type=float,
        default=0.02,
        help="Likelihood flux_error_floor_frac to show in sigma_eff.",
    )
    parser.add_argument(
        "--sigma-sys-mag",
        type=float,
        default=DEFAULT_PHOTERR_SIGMA_SYS_MAG,
        help="PhotErr-style systematic magnitude term.",
    )
    parser.add_argument(
        "--min-flux-ratio",
        type=float,
        default=1.0e-3,
        help="Minimum plotted |flux| as a ratio of f5.",
    )
    parser.add_argument(
        "--max-flux-ratio",
        type=float,
        default=1.0e6,
        help="Maximum plotted |flux| as a ratio of f5.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=512,
        help="Number of flux-grid points.",
    )
    return parser.parse_args()


def _compute_components(
    *,
    band: str,
    model: dict,
    flux_floor_frac: float,
    min_flux_ratio: float,
    max_flux_ratio: float,
    n_grid: int,
) -> dict:
    if min_flux_ratio <= 0.0 or max_flux_ratio <= min_flux_ratio:
        raise ValueError("Require 0 < min_flux_ratio < max_flux_ratio")
    if n_grid < 8:
        raise ValueError("Require at least eight flux-grid points")

    m5 = _band_value(model["m5"], band)
    f5 = float(abmag_to_fnu_cgs(m5))
    gamma = _gamma_value(model, band)
    sigma_sys_mag = float(model.get("sigma_sys_mag", DEFAULT_PHOTERR_SIGMA_SYS_MAG))
    sys_frac = float(np.expm1(np.log(10.0) * sigma_sys_mag / 2.5))
    flux = np.logspace(
        np.log10(f5 * min_flux_ratio),
        np.log10(f5 * max_flux_ratio),
        n_grid,
    )

    sigma_depth2 = (0.04 - gamma) * flux * f5 + gamma * f5**2
    sigma_depth = np.sqrt(np.maximum(sigma_depth2, 0.0))
    sigma_sys = sys_frac * flux
    sigma_catalog_manual = np.sqrt(sigma_depth**2 + sigma_sys**2)
    sigma_catalog = flux_error_from_model(flux, model, band_name=band)
    np.testing.assert_allclose(
        sigma_catalog,
        np.maximum(sigma_catalog_manual, DEFAULT_MIN_SIGMA_FNU_CGS),
        rtol=5.0e-13,
        atol=0.0,
    )

    sigma_floor = flux_floor_frac * flux
    sigma_eff = effective_flux_sigma(
        flux,
        sigma_catalog,
        error_floor_frac=flux_floor_frac,
        error_jitter=0.0,
        floor_reference="observed",
    )
    return {
        "band": band,
        "m5": float(m5),
        "f5": f5,
        "gamma": float(gamma),
        "sigma_sys_mag": sigma_sys_mag,
        "sys_frac": sys_frac,
        "flux_floor_frac": float(flux_floor_frac),
        "flux": flux,
        "sigma_depth": sigma_depth,
        "sigma_sys": sigma_sys,
        "sigma_catalog": sigma_catalog,
        "sigma_floor": sigma_floor,
        "sigma_eff": sigma_eff,
    }


def _band_value(values: dict, band: str) -> float:
    aliases = _band_aliases(band)
    for alias in aliases:
        if alias in values:
            return float(values[alias])
    raise ValueError(f"No value configured for band {band!r}")


def _gamma_value(model: dict, band: str) -> float:
    gamma = model.get("gamma") or {}
    for alias in _band_aliases(band):
        if alias in gamma:
            return float(gamma[alias])
    eta = model.get("eta") or {}
    for alias in _band_aliases(band):
        if alias in eta:
            return 0.04 * float(eta[alias])
    return 0.04 * float(model.get("default_eta", 1.0))


def _band_aliases(band: str) -> tuple[str, ...]:
    aliases = [band, band.lower()]
    lower = band.lower()
    if lower.startswith("lsst_"):
        suffix = lower.removeprefix("lsst_")
        aliases.extend([suffix, suffix.upper()])
    if lower.startswith("roman_"):
        suffix = band.split("_", 1)[1]
        aliases.extend([suffix, suffix.lower(), suffix.upper()])
    return tuple(dict.fromkeys(aliases))


def _plot_components(components: dict, figure_base: Path) -> None:
    configure_plot_style()
    flux = components["flux"]
    f5 = float(components["f5"])
    band = str(components["band"])
    m5 = float(components["m5"])
    floor = float(components["flux_floor_frac"])
    sys_frac = float(components["sys_frac"])
    floor_percent = 100.0 * floor

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.9),
        constrained_layout=True,
    )
    fig.suptitle(
        "Flux-space photometric uncertainty used by the Diffsky likelihood",
        fontsize=15,
        fontweight="bold",
    )

    labels = {
        "sigma_depth": r"$\sigma_{\rm rand}$: depth / PhotErr random",
        "sigma_sys": r"$\sigma_{\rm sys}$: PhotErr systematic",
        "sigma_catalog": r"$\sigma_{\rm cat}$: parquet fluxerr_*",
        "sigma_floor": rf"$\sigma_{{\rm floor}}$: {floor_percent:.0f} percent likelihood term",
        "sigma_eff": r"$\sigma_{\rm eff}$: total likelihood scale",
    }
    styles = {
        "sigma_depth": dict(color=COLORS["depth"], linewidth=2.2),
        "sigma_sys": dict(color=COLORS["systematic"], linewidth=2.2),
        "sigma_catalog": dict(color=COLORS["catalog"], linewidth=2.8),
        "sigma_floor": dict(color=COLORS["floor"], linewidth=2.2, linestyle="--"),
        "sigma_eff": dict(color=COLORS["effective"], linewidth=3.2),
    }

    ax = axes[0]
    for key in (
        "sigma_depth",
        "sigma_sys",
        "sigma_catalog",
        "sigma_floor",
        "sigma_eff",
    ):
        ax.loglog(flux, components[key], label=labels[key], **styles[key])
    ax.axvline(f5, color="#4B5563", linestyle=":", linewidth=1.1)
    ax.text(
        f5 * 1.12,
        ax.get_ylim()[0] * 1.6,
        rf"$f_5$ ({m5:.2f} AB)",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="#374151",
    )
    ax.set_title(f"{band}: absolute uncertainty components")
    ax.set_xlabel(r"$|f_{\rm obs}|$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax.set_ylabel(r"flux uncertainty  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax.legend(loc="upper left", fontsize=8.0)

    ax = axes[1]
    for key in (
        "sigma_depth",
        "sigma_sys",
        "sigma_catalog",
        "sigma_floor",
        "sigma_eff",
    ):
        ax.loglog(flux, components[key] / flux, label=labels[key], **styles[key])
    ax.axhline(sys_frac, color=COLORS["systematic"], linestyle=":", linewidth=1.0)
    ax.axhline(floor, color=COLORS["floor"], linestyle=":", linewidth=1.0)
    ax.axvline(f5, color="#4B5563", linestyle=":", linewidth=1.1)
    ax.text(
        flux[3],
        sys_frac * 1.09,
        rf"$s_{{\rm sys}}={sys_frac:.3g}$",
        fontsize=8.5,
        color=COLORS["systematic"],
        va="bottom",
    )
    ax.text(
        flux[3],
        floor * 1.09,
        rf"$\epsilon={floor_percent:.0f}\%$",
        fontsize=8.5,
        color=COLORS["floor"],
        va="bottom",
    )
    ax.set_title("Same model as fractional uncertainty")
    ax.set_xlabel(r"$|f_{\rm obs}|$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax.set_ylabel(r"$\sigma / |f_{\rm obs}|$")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.set_ylim(bottom=min(sys_frac, floor) / 4.0)

    fig.text(
        0.5,
        -0.035,
        (
            r"Shown with $f_{\rm ref}=f_{\rm obs}$ for the likelihood floor. "
            r"Amortized JAX likelihood uses $f_{\rm ref}=f_{\rm model}$."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )

    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(figure_base.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _plot_annotated_explainer(components: dict, figure_base: Path) -> None:
    configure_plot_style()
    flux = components["flux"]
    f5 = float(components["f5"])
    band = str(components["band"])
    m5 = float(components["m5"])
    floor = float(components["flux_floor_frac"])
    sys_frac = float(components["sys_frac"])
    floor_percent = 100.0 * floor

    fig = plt.figure(figsize=(13.4, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2.15, 1.2])
    ax_abs = fig.add_subplot(grid[0, 0])
    ax_frac = fig.add_subplot(grid[0, 1])
    ax_flow = fig.add_subplot(grid[0, 2])
    ax_cols = fig.add_subplot(grid[1, 0])
    ax_depth = fig.add_subplot(grid[1, 1])
    ax_likelihood = fig.add_subplot(grid[1, 2])

    fig.suptitle(
        "How the Diffsky photometric error is built from flux",
        fontsize=16,
        fontweight="bold",
    )
    labels = {
        "sigma_depth": r"$\sigma_{\rm rand}$ depth/random",
        "sigma_sys": r"$\sigma_{\rm sys}$ systematic",
        "sigma_catalog": r"$\sigma_{\rm cat}$ catalog fluxerr_*",
        "sigma_floor": rf"$\sigma_{{\rm floor}}$ {floor_percent:.0f} percent floor",
        "sigma_eff": r"$\sigma_{\rm eff}$ Student-t scale",
    }
    styles = {
        "sigma_depth": dict(color=COLORS["depth"], linewidth=2.1),
        "sigma_sys": dict(color=COLORS["systematic"], linewidth=2.1),
        "sigma_catalog": dict(color=COLORS["catalog"], linewidth=2.8),
        "sigma_floor": dict(color=COLORS["floor"], linewidth=2.2, linestyle="--"),
        "sigma_eff": dict(color=COLORS["effective"], linewidth=3.0),
    }

    for key in styles:
        ax_abs.loglog(flux, components[key], label=labels[key], **styles[key])
    ax_abs.axvline(f5, color="#4B5563", linestyle=":", linewidth=1.1)
    ax_abs.set_title(f"{band}: absolute terms")
    ax_abs.set_xlabel(r"$|f_{\rm obs}|$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax_abs.set_ylabel(r"$\sigma$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax_abs.legend(loc="upper left", fontsize=7.7)

    for key in styles:
        ax_frac.loglog(flux, components[key] / flux, label=labels[key], **styles[key])
    ax_frac.axhline(sys_frac, color=COLORS["systematic"], linestyle=":", linewidth=1.0)
    ax_frac.axhline(floor, color=COLORS["floor"], linestyle=":", linewidth=1.0)
    ax_frac.axvline(f5, color="#4B5563", linestyle=":", linewidth=1.1)
    ax_frac.text(
        flux[3],
        sys_frac * 1.08,
        rf"$s_{{\rm sys}}={sys_frac:.4g}$",
        color=COLORS["systematic"],
        fontsize=8.0,
    )
    ax_frac.text(
        flux[3],
        floor * 1.08,
        rf"$\epsilon={floor_percent:.0f}\%$",
        color=COLORS["floor"],
        fontsize=8.0,
    )
    ax_frac.set_title("Same terms as fractional uncertainty")
    ax_frac.set_xlabel(r"$|f_{\rm obs}|$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax_frac.set_ylabel(r"$\sigma / |f_{\rm obs}|$")
    ax_frac.set_ylim(bottom=min(sys_frac, floor) / 4.0)

    _draw_flow_panel(ax_flow, components)
    _draw_text_box(
        ax_cols,
        "Catalog columns",
        _catalog_column_text(components),
        color=COLORS["catalog"],
    )
    _draw_text_box(
        ax_depth,
        "Depth and error-model provenance",
        _depth_provenance_text(components),
        color=COLORS["depth"],
    )
    _draw_text_box(
        ax_likelihood,
        "Student-t likelihood inputs",
        _student_t_text(components),
        color=COLORS["effective"],
    )

    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(figure_base.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _draw_flow_panel(ax: plt.Axes, components: dict) -> None:
    ax.set_axis_off()
    band = str(components["band"])
    boxes = [
        ("Parquet", f"flux_{band}\n= f_obs\nfnu_cgs"),
        ("Manifest", "m5, gamma/eta,\nsigma_sys_mag"),
        ("Generated column", f"fluxerr_{band}\n= sigma_cat\nfnu_cgs"),
        ("Fit config", "epsilon=0.02,\njitter=0"),
        ("Likelihood", "Student-t:\ny=f_obs, loc=f_model,\nscale=sigma_eff"),
    ]
    y_positions = np.linspace(0.86, 0.14, len(boxes))
    for idx, ((title, body), y) in enumerate(zip(boxes, y_positions, strict=True)):
        color = [
            COLORS["catalog"],
            COLORS["depth"],
            COLORS["catalog"],
            COLORS["floor"],
            COLORS["effective"],
        ][idx]
        ax.text(
            0.5,
            y,
            f"{title}\n{body}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
            ),
        )
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[idx + 1] + 0.09),
                xytext=(0.5, y - 0.09),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#4B5563", linewidth=1.0),
            )
    ax.set_title("Data path into the likelihood")


def _draw_text_box(ax: plt.Axes, title: str, body: str, *, color: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=color,
    )
    ax.text(
        0.0,
        0.85,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        linespacing=1.3,
        color="#111827",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#FAFAFA",
            edgecolor=color,
            linewidth=1.0,
        ),
    )


def _wrap(text: str, width: int = 58) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _catalog_column_text(components: dict) -> str:
    band = str(components["band"])
    return "\n".join(
        [
            _wrap(f"`flux_{band}` is the observed target flux f_obs."),
            "Unit: fnu_cgs = erg s^-1 cm^-2 Hz^-1.",
            "",
            _wrap(f"`fluxerr_{band}` is the materialized catalog uncertainty sigma_cat."),
            _wrap("It is synthetic for this Diffsky subset, not a native HLTDS measurement error."),
            "",
            _wrap("There is no per-object m5 column: m5 is a per-band depth from the error-model manifest/config."),
        ]
    )


def _depth_provenance_text(components: dict) -> str:
    band = str(components["band"])
    if band.lower().startswith("lsst_"):
        m5_source = "LSST 10-year coadd default in default_m5_depth_error_model()."
        shape_source = "gamma is the LSST PhotErr-style band parameter."
    elif band.lower().startswith("roman_"):
        m5_source = "Roman WFI one-hour point-source default in default_m5_depth_error_model()."
        shape_source = "eta=0.95 is converted to gamma=0.04*eta."
    else:
        m5_source = "Configured per-band depth in the m5_depth error model."
        shape_source = "gamma or eta is read from the same error model."
    return "\n".join(
        [
            _wrap(f"m5 for `{band}` = {float(components['m5']):.2f} AB."),
            _wrap(m5_source),
            _wrap("f5 is obtained by AB conversion: 3631 Jy * 10^(-0.4 m5) * 1e-23."),
            "",
            _wrap(shape_source),
            _wrap(f"sigma_sys_mag={float(components['sigma_sys_mag']):.3f} mag gives s_sys={float(components['sys_frac']):.4g}."),
        ]
    )


def _student_t_text(components: dict) -> str:
    floor = float(components["flux_floor_frac"])
    return "\n".join(
        [
            _wrap("The Student-t sees y=f_obs, loc=f_model, and scale=sigma_eff."),
            _wrap(f"sigma_eff^2 = sigma_cat^2 + ({floor:.2f} |f_ref|)^2 + jitter^2."),
            _wrap("Standalone MAP/posterior reporting uses f_ref=f_obs; amortized JAX uses f_ref=f_model."),
            "",
            _wrap("Active configs use nu=2. MAP minimizes (nu+1) log(1+r^2/nu) per valid band."),
            _wrap("Amortized training uses the full Student-t log density, with the same residual scale r."),
        ]
    )


def _latex_snippet(components: dict, figure_stem: str) -> str:
    band = str(components["band"]).replace("_", r"\_")
    m5 = float(components["m5"])
    gamma = float(components["gamma"])
    sigma_sys_mag = float(components["sigma_sys_mag"])
    sys_frac = float(components["sys_frac"])
    floor = float(components["flux_floor_frac"])
    template = r"""% Requires \usepackage{amsmath}, \usepackage{xcolor}, \usepackage{graphicx}
\definecolor{dspsDepth}{HTML}{@@COLOR_DEPTH@@}
\definecolor{dspsSys}{HTML}{@@COLOR_SYS@@}
\definecolor{dspsCatalog}{HTML}{@@COLOR_CATALOG@@}
\definecolor{dspsFloor}{HTML}{@@COLOR_FLOOR@@}
\definecolor{dspsEffective}{HTML}{@@COLOR_EFFECTIVE@@}

\begin{align}
f_{5,b} &= 3631\,{\rm Jy}\,10^{-0.4m_{5,b}}\,(10^{-23})
\\
\textcolor{dspsDepth}{\sigma^2_{{\rm rand},ib}}
&= \textcolor{dspsDepth}{(0.04-\gamma_b)|f_{{\rm obs},ib}|f_{5,b}
   + \gamma_b f_{5,b}^2}
\\
s_{\rm sys} &= 10^{\sigma_{{\rm sys,mag}}/2.5} - 1
\\
\textcolor{dspsCatalog}{\sigma^2_{{\rm cat},ib}}
&= \textcolor{dspsDepth}{\sigma^2_{{\rm rand},ib}}
 + \textcolor{dspsSys}{(s_{\rm sys}|f_{{\rm obs},ib}|)^2}
\\
\textcolor{dspsEffective}{\sigma^2_{{\rm eff},ib}}
&= \textcolor{dspsCatalog}{\sigma^2_{{\rm cat},ib}}
 + \textcolor{dspsFloor}{(\epsilon |f_{{\rm ref},ib}|)^2}
 + \sigma_{\rm jitter}^2
\\
r_{ib} &= \frac{f_{{\rm obs},ib}-f_{{\rm model},ib}}
              {\textcolor{dspsEffective}{\sigma_{{\rm eff},ib}}}
\end{align}

\paragraph{Parameter provenance and units.}
\begin{itemize}
\item \(f_{{\rm obs},ib}\): observed flux read from \texttt{flux\_@@RAW_BAND@@};
unit \(f_\nu\) cgs, i.e. \({\rm erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}\).
\item \(\sigma_{{\rm cat},ib}\): catalog uncertainty read from
\texttt{fluxerr\_@@RAW_BAND@@}; same unit as flux. For this Diffsky subset this
column is synthetic, generated from the \texttt{m5\_depth} model, not native
HLTDS photometric noise.
\item \(m_{5,b}\): per-band 5-sigma depth from the error-model manifest/config,
not a per-object parquet column. For \texttt{lsst\_*} bands the defaults are
LSST 10-year coadd depths; for \texttt{roman\_*} bands the defaults are Roman
WFI one-hour point-source depths.
\item \(\gamma_b\): LSST PhotErr-style band parameter. Roman bands use
\(\gamma_b=0.04\eta_b\), with the current default \(\eta_b=0.95\).
\item \(\epsilon\): likelihood floor fraction from the fit config
(\texttt{flux\_error\_floor\_frac}); the active Diffsky configs use
\(\epsilon=0.02\).
\end{itemize}

\paragraph{Student-t likelihood.}
The active configs use \(\nu=2\). The Student-t receives
\[
y=f_{{\rm obs},ib}, \qquad
\mu=f_{{\rm model},ib}, \qquad
{\rm scale}=\sigma_{{\rm eff},ib}.
\]
Standalone MAP minimizes the per-band robust objective
\[
(\nu+1)\log\left(1+\frac{r_{ib}^2}{\nu}\right),
\]
while amortized training uses the full Student-t log density with the same
residual scale. In standalone MAP/reporting \(f_{\rm ref}=f_{\rm obs}\);
in the amortized JAX likelihood \(f_{\rm ref}=f_{\rm model}\).

For the plotted @@BAND@@ example:
\[
m_5=@@M5@@,\qquad
\gamma=@@GAMMA@@,\qquad
\sigma_{\rm sys,mag}=@@SIGMA_SYS_MAG@@,\qquad
s_{\rm sys}\simeq @@SYS_FRAC@@,\qquad
\epsilon=@@FLOOR@@.
\]

\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{@@FIGURE_STEM@@.pdf}
\caption{Color-coded decomposition of the catalog photometric error and the
likelihood error scale. The parquet \texttt{fluxerr\_*} value is
\(\sigma_{\rm cat}\); the likelihood uses \(\sigma_{\rm eff}\).}
\end{figure}
"""
    replacements = {
        "@@COLOR_DEPTH@@": COLORS["depth"].removeprefix("#"),
        "@@COLOR_SYS@@": COLORS["systematic"].removeprefix("#"),
        "@@COLOR_CATALOG@@": COLORS["catalog"].removeprefix("#"),
        "@@COLOR_FLOOR@@": COLORS["floor"].removeprefix("#"),
        "@@COLOR_EFFECTIVE@@": COLORS["effective"].removeprefix("#"),
        "@@BAND@@": band,
        "@@RAW_BAND@@": str(components["band"]).replace("_", r"\_"),
        "@@M5@@": f"{m5:.2f}",
        "@@GAMMA@@": f"{gamma:.3f}",
        "@@SIGMA_SYS_MAG@@": f"{sigma_sys_mag:.3f}",
        "@@SYS_FRAC@@": f"{sys_frac:.4g}",
        "@@FLOOR@@": f"{floor:.2f}",
        "@@FIGURE_STEM@@": figure_stem,
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    return template


def _standalone_latex_document(snippet_name: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\pagestyle{{empty}}

\begin{{document}}
\section*{{PhotErr-style flux-error model}}
\input{{{snippet_name}}}
\end{{document}}
"""


def _render_equations_pdf(components: dict, path: Path) -> None:
    configure_plot_style()
    band = str(components["band"]).replace("_", r"\_")
    lines = [
        (
            COLORS["depth"],
            r"$f_{5,b}=3631\,{\rm Jy}\,10^{-0.4m_{5,b}}\,10^{-23}$",
        ),
        (
            COLORS["depth"],
            (
                r"$\sigma^2_{{\rm rand},ib}=(0.04-\gamma_b)"
                r"|f_{{\rm obs},ib}|f_{5,b}+\gamma_b f_{5,b}^2$"
            ),
        ),
        (
            COLORS["systematic"],
            r"$s_{\rm sys}=10^{\sigma_{{\rm sys,mag}}/2.5}-1$",
        ),
        (
            COLORS["catalog"],
            (
                r"$\sigma^2_{{\rm cat},ib}=\sigma^2_{{\rm rand},ib}"
                r"+(s_{\rm sys}|f_{{\rm obs},ib}|)^2$"
            ),
        ),
        (
            COLORS["effective"],
            (
                r"$\sigma^2_{{\rm eff},ib}=\sigma^2_{{\rm cat},ib}"
                r"+(\epsilon |f_{{\rm ref},ib}|)^2+\sigma_{\rm jitter}^2$"
            ),
        ),
        (
            COLORS["effective"],
            (
                r"$r_{ib}=\dfrac{f_{{\rm obs},ib}-f_{{\rm model},ib}}"
                r"{\sigma_{{\rm eff},ib}}$"
            ),
        ),
    ]
    labels = [
        (COLORS["depth"], "rand"),
        (COLORS["systematic"], "sys"),
        (COLORS["catalog"], "cat"),
        (COLORS["floor"], "floor"),
        (COLORS["effective"], "eff"),
    ]

    fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.text(
        0.06,
        0.94,
        "PhotErr-style flux-error model used by the Diffsky likelihood",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.895,
        "Equations, parameter provenance, catalog columns, and Student-t inputs.",
        fontsize=10.5,
        color="#374151",
        ha="left",
        va="top",
    )

    y = 0.79
    for color, equation in lines:
        fig.text(0.06, y, equation, color=color, fontsize=15.5, ha="left", va="center")
        y -= 0.092

    x = 0.06
    for color, label in labels:
        fig.patches.append(
            plt.Rectangle(
                (x, 0.245),
                0.018,
                0.014,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
            )
        )
        fig.text(x + 0.024, 0.252, label, fontsize=8.5, ha="left", va="center")
        x += 0.105

    numeric = (
        rf"Plotted example: ${band}$, "
        rf"$m_5={float(components['m5']):.2f}$, "
        rf"$\gamma={float(components['gamma']):.3f}$, "
        rf"$\sigma_{{\rm sys,mag}}={float(components['sigma_sys_mag']):.3f}$, "
        rf"$s_{{\rm sys}}\simeq {float(components['sys_frac']):.4g}$, "
        rf"$\epsilon={float(components['flux_floor_frac']):.2f}$."
    )
    fig.text(0.06, 0.205, numeric, fontsize=10.2, color="#111827", ha="left")

    note_sections = [
        ("Catalog columns", _catalog_column_text(components), COLORS["catalog"]),
        ("Where m5 comes from", _depth_provenance_text(components), COLORS["depth"]),
        ("Student-t", _student_t_text(components), COLORS["effective"]),
    ]
    y_note = 0.80
    for title, body, color in note_sections:
        fig.text(
            0.63,
            y_note,
            title,
            fontsize=11.0,
            fontweight="bold",
            color=color,
            ha="left",
            va="top",
        )
        fig.text(
            0.63,
            y_note - 0.032,
            body.replace("`", ""),
            fontsize=8.2,
            color="#111827",
            ha="left",
            va="top",
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#FAFAFA",
                edgecolor=color,
                linewidth=0.9,
            ),
        )
        y_note -= 0.235

    fig.text(
        0.06,
        0.07,
        (
            r"Standalone TeX is saved alongside this PDF. The plotted PDF uses "
            r"Matplotlib math rendering, so it does not require a system TeX engine."
        ),
        fontsize=8.8,
        color="#4B5563",
        ha="left",
    )

    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _markdown_snippet(components: dict) -> str:
    band = str(components["band"])
    return f"""# PhotErr-style flux-error model

For each object `i` and band `b`, the active Diffsky parquet stores
`fluxerr_* = sigma_cat` in `fnu_cgs`. The model uses:

```tex
f_{{5,b}} = 3631\\,{{\\rm Jy}}\\,10^{{-0.4m_{{5,b}}}}\\,10^{{-23}}
sigma^2_{{rand,ib}} = (0.04 - gamma_b)|f_{{obs,ib}}|f_{{5,b}} + gamma_b f_{{5,b}}^2
s_{{sys}} = 10^(sigma_{{sys,mag}} / 2.5) - 1
sigma^2_{{cat,ib}} = sigma^2_{{rand,ib}} + (s_{{sys}} |f_{{obs,ib}}|)^2
sigma^2_{{eff,ib}} = sigma^2_{{cat,ib}} + (epsilon |f_{{ref,ib}}|)^2 + sigma^2_{{jitter}}
r_{{ib}} = (f_{{obs,ib}} - f_{{model,ib}}) / sigma_{{eff,ib}}
```

Color schema:

- depth/random term: `{COLORS["depth"]}`
- PhotErr systematic term: `{COLORS["systematic"]}`
- catalog `fluxerr_*`: `{COLORS["catalog"]}`
- likelihood fractional floor: `{COLORS["floor"]}`
- total likelihood sigma: `{COLORS["effective"]}`

Plotted band: `{components["band"]}`.
`m5={components["m5"]:.2f}`, `gamma={components["gamma"]:.3f}`,
`sigma_sys_mag={components["sigma_sys_mag"]:.3f}`,
`s_sys={components["sys_frac"]:.6g}`, and
`epsilon={components["flux_floor_frac"]:.3f}`.

The figure assumes `f_ref=f_obs` so the floor is visible as a simple function
of observed flux. The amortized JAX likelihood uses `f_ref=f_model`.

## Parameter provenance

### Catalog columns

{_catalog_column_text(components)}

For the plotted band, the active columns are:

- observed flux: `flux_{band}`
- catalog uncertainty: `fluxerr_{band}`

### Where `m5` comes from

{_depth_provenance_text(components)}

`m5` is therefore a per-band assumption recorded in the error-model manifest,
not a measured per-object column.

### Student-t use

{_student_t_text(components)}
"""


def _summary_payload(components: dict) -> dict[str, float | str | dict[str, str]]:
    flux = components["flux"]
    sigma_eff = components["sigma_eff"]
    band = str(components["band"])
    return {
        "band": band,
        "flux_column": f"flux_{band}",
        "flux_error_column": f"fluxerr_{band}",
        "flux_unit": "erg s^-1 cm^-2 Hz^-1",
        "m5_ab": float(components["m5"]),
        "m5_source": (
            "LSST 10-year coadd defaults for lsst_*; Roman WFI one-hour "
            "point-source defaults for roman_*; recorded in the m5_depth "
            "error-model manifest/config."
        ),
        "f5_fnu_cgs": float(components["f5"]),
        "gamma": float(components["gamma"]),
        "gamma_source": (
            "Configured LSST gamma value or Roman gamma=0.04*eta."
        ),
        "sigma_sys_mag": float(components["sigma_sys_mag"]),
        "sys_frac": float(components["sys_frac"]),
        "flux_error_floor_frac": float(components["flux_floor_frac"]),
        "student_t_dof": 2.0,
        "student_t_inputs": "y=f_obs, loc=f_model, scale=sigma_eff",
        "flux_min_fnu_cgs": float(np.nanmin(flux)),
        "flux_max_fnu_cgs": float(np.nanmax(flux)),
        "sigma_eff_min_fnu_cgs": float(np.nanmin(sigma_eff)),
        "sigma_eff_max_fnu_cgs": float(np.nanmax(sigma_eff)),
        "colors": dict(COLORS),
    }


if __name__ == "__main__":
    main()
