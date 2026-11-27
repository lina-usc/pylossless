# pylossless/diagnostic.py
"""Visualization helpers for bandpower + PSD before/after cleaning.

Functions to create diagnostic figures showing bandpower topomaps and PSD plots
before and after cleaning, highlighting dropped and fixed channels.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec

import mne

COLOR_REMOVED = "#FFD700"  # Yellow (Dropped)
COLOR_FIXED = "#32CD32"  # Green (Kept)


def _drop_duplicate_coords(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) < 2:
        return raw
    locs = np.array([raw.info["chs"][p]["loc"][:3] for p in eeg_picks])
    dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)
    dup_pairs = np.argwhere((dist < 1e-10) & (dist > 0))
    to_drop = set()
    for i, j in dup_pairs:
        to_drop.add(raw.ch_names[eeg_picks[j]])
    if to_drop:
        raw.drop_channels(list(sorted(to_drop)))
    return raw


def _apply_custom_montage_keep_all(
    raw: mne.io.BaseRaw,
    montage_path: Optional[Path] = None,
    montage_kwargs: Optional[dict] = None,
):
    montage_kwargs = montage_kwargs or {}
    if montage_path and Path(montage_path).exists():
        try:
            mont = mne.channels.read_custom_montage(str(montage_path))
            mont_chs = set(getattr(mont, "ch_names", []) or [])
            no_pos = [c for c in raw.ch_names if c not in mont_chs]
            if no_pos:
                try:
                    raw.set_channel_types({c: "misc" for c in no_pos})
                except Exception:
                    pass
            raw.set_montage(mont, **montage_kwargs)
        except Exception:
            # fallback to standard 10-20
            try:
                raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
            except Exception:
                pass
    else:
        try:
            raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
        except Exception:
            pass


def _ensure_montage(raw: mne.io.BaseRaw):
    if not raw.get_montage():
        # do not raise - simply warn in logs external to this function
        return


def _plot_topomap_with_highlights(ax, data, info, raw, title, highlights, cmap):
    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)
    im, _ = mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(vmin, vmax),
        contours=0,
        sensors=False,
        res=256,
        sphere=(0.0, 0.0, 0.0, 0.095),
        extrapolate="local",
    )
    ax.set_title(title, fontsize=12)

    # electrode dots
    for ch in info["chs"]:
        loc = ch.get("loc", None)
        if loc is not None and len(loc) >= 2:
            x, y = loc[:2]
            ax.scatter(x, y, s=20, color="black", zorder=4)

    # highlights
    for _, (ch_list, color) in highlights.items():
        for ch in ch_list:
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                loc = raw.info["chs"][idx].get("loc", None)
                if loc is not None and len(loc) >= 2:
                    x, y = loc[:2]
                    ax.scatter(
                        x,
                        y,
                        s=80,
                        color=color,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=6,
                    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power (dB)", fontsize=9)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))


def _draw_bandpower_with_highlight(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    axes,
    removed_mapped: List[str],
    fixed_mapped: List[str],
    fmin: float = 1.0,
    fmax: float = 40.0,
):
    _ensure_montage(raw_before)
    _ensure_montage(raw_after)

    psd_b = raw_before.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=2048, verbose=False
    )
    psd_a = raw_after.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=2048, verbose=False
    )

    pb = 10 * np.log10(psd_b.get_data().mean(axis=1))
    pa = 10 * np.log10(psd_a.get_data().mean(axis=1))

    info_b = mne.pick_info(raw_before.info, sel=range(len(pb)))
    info_a = mne.pick_info(raw_after.info, sel=range(len(pa)))

    cmap = plt.cm.coolwarm

    _plot_topomap_with_highlights(
        axes[0],
        pb,
        info_b,
        raw_before,
        "a) Raw Bandpower\nDropped Sensors (Yellow)",
        {"removed": (removed_mapped, COLOR_REMOVED)},
        cmap,
    )
    _plot_topomap_with_highlights(
        axes[1],
        pa,
        info_a,
        raw_after,
        "b) Cleaned Bandpower\nFixed (Green) | Dropped (Yellow)",
        {
            "fixed": (fixed_mapped, COLOR_FIXED),
            "removed": (removed_mapped, COLOR_REMOVED),
        },
        cmap,
    )

    legend_elements = [
        Patch(facecolor=COLOR_REMOVED, edgecolor="black", label="Dropped (Bad)"),
        Patch(facecolor=COLOR_FIXED, edgecolor="black", label="Fixed / Kept"),
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        frameon=False,
    )


def _draw_psd_split(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, axes):
    _ensure_montage(raw_before)
    _ensure_montage(raw_after)
    raw_before.plot_psd(
        fmin=1, fmax=40, ax=axes[0], average=False, picks="eeg", show=False
    )
    axes[0].set_title("Raw PSD", fontsize=11)
    try:
        ra_refed = mne.set_eeg_reference(raw_after.copy())[0]
    except Exception:
        ra_refed = raw_after
    ra_refed.plot_psd(
        fmin=1, fmax=40, ax=axes[1], average=False, picks="eeg", show=False
    )
    axes[1].set_title("Cleaned PSD", fontsize=11)


def create_bandpower_psd_flagged(
    vhdr_path: Path,
    fif_path: Path,
    flagged_tsv: Path,
    out_dir: Path,
    montage_path: Optional[Path] = None,
    fmin: float = 1.0,
    fmax: float = 40.0,
    overwrite: bool = False,
) -> Tuple[Path, dict]:
    """Create and save bandpower + PSD figure for a single subject.

    Parameters
    ----------
    vhdr_path : Path
        BrainVision .vhdr file for the raw (before) recording.
    fif_path : Path
        Cleaned .fif file (after).
    flagged_tsv : Path
        TSV produced by pylossless that lists flagged channels.
    out_dir : Path
        Directory where figure is saved.
    montage_path : Optional[Path]
        Path to a custom montage file. If None, attempt standard_1020 montage.
    fmin, fmax : float
        Frequency limits for PSD and bandpower calculation.
    overwrite : bool
        If True, overwrite existing figure.

    Returns
    -------
    out_path : Path
        Path to saved PNG figure.
    log : dict
        Dictionary with keys: 'status', 'message', 'dropped', 'fixed', 'out_path'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic inputs check
    for p in (vhdr_path, fif_path, flagged_tsv):
        if not Path(p).exists():
            return None, {"status": "FileMissing", "message": f"{p} not found"}

    try:
        raw_before = _drop_duplicate_coords(
            mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        )
        raw_after = _drop_duplicate_coords(
            mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        )
        _apply_custom_montage_keep_all(raw_before, montage_path)
        _apply_custom_montage_keep_all(raw_after, montage_path)

        df_flags = pd.read_csv(flagged_tsv, sep="\t")
        ch_col = next(
            (
                c
                for c in df_flags.columns
                if c.lower() in ["ch_name", "ch_names", "channel", "name"]
            ),
            None,
        )
        flag_col = next(
            (
                c
                for c in df_flags.columns
                if c.lower() in ["flag", "status", "reason", "labels", "note", "type"]
            ),
            None,
        )
        if not ch_col or not flag_col:
            raise ValueError("Could not find channel/flag columns in flagged file")

        df_flags["ch_norm"] = (
            df_flags[ch_col]
            .astype(str)
            .str.replace("EEG ", "", regex=False)
            .str.lower()
            .str.strip()
        )
        df_flags["flag_norm"] = df_flags[flag_col].astype(str).str.lower()

        removed = df_flags.loc[
            df_flags["flag_norm"].str.contains(
                "uncorrelated|flat|bad|remove|drop|noisy|rank", na=False
            ),
            "ch_norm",
        ].tolist()

        rb_names = [
            ch.lower().replace("eeg ", "").strip() for ch in raw_before.ch_names
        ]
        removed_mapped = [
            raw_before.ch_names[i] for i, ch in enumerate(rb_names) if ch in removed
        ]
        fixed_mapped = [ch for ch in raw_after.ch_names if ch not in removed_mapped]

        # outfile name
        subj_base = vhdr_path.stem
        out_path = out_dir / f"{subj_base}_bandpower_psd_flagged.png"
        if out_path.exists() and not overwrite:
            return (
                out_path,
                {
                    "status": "Exists",
                    "message": "File already exists",
                    "dropped": removed_mapped,
                    "fixed": fixed_mapped,
                    "out_path": str(out_path),
                },
            )

        # Plot and save
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_clean = fig.add_subplot(gs[0, 1])
        _draw_bandpower_with_highlight(
            raw_before,
            raw_after,
            [ax_raw, ax_clean],
            removed_mapped,
            fixed_mapped,
            fmin=fmin,
            fmax=fmax,
        )
        ax_psd1 = fig.add_subplot(gs[1, 0])
        ax_psd2 = fig.add_subplot(gs[1, 1])
        _draw_psd_split(raw_before, raw_after, [ax_psd1, ax_psd2])
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, wspace=0.25, hspace=0.4)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return (
            out_path,
            {
                "status": "Success",
                "message": "Saved",
                "dropped": removed_mapped,
                "fixed": fixed_mapped,
                "out_path": str(out_path),
            },
        )
    except Exception as e:
        return None, {"status": "Error", "message": f"{type(e).__name__}: {e}"}


def process_subject_from_paths(
    vhdr_path: Path,
    root_clean_dir: Path,
    root_pylossless_dir: Path,
    out_dir: Path,
    montage_path: Optional[Path] = None,
    overwrite: bool = False,
):
    """Locate matching cleaned .fif and flagged tsv.

    Given a BrainVision vhdr and search dirs, locate matching cleaned .fif and
    flagged tsv and run create_bandpower_psd_flagged().
    """
    # find fif and tsv similar to original script
    subj_base = vhdr_path.stem
    match = re.search(r"ABC(\d+)", subj_base)
    if not match:
        return None, {"status": "Skip", "message": "Subject pattern not found"}

    subj_id = match.group(1)
    subj_name = f"sub-{subj_id}"

    fif = next(
        Path(root_clean_dir).rglob(f"{subj_name}*_clean*.fif"),
        None,
    )
    flagged_file = next(
        Path(root_pylossless_dir).rglob(f"{subj_name}*_FlaggedChs.tsv"),
        None,
    )
    if not fif or not flagged_file:
        msg = "Missing FIF or TSV file"
        return (
            None,
            {
                "status": "FileMissing",
                "message": msg,
                "vhdr": str(vhdr_path),
                "fif": str(fif) if fif else None,
                "flagged": str(flagged_file) if flagged_file else None,
            },
        )

    return create_bandpower_psd_flagged(
        vhdr_path,
        fif,
        flagged_file,
        out_dir,
        montage_path=montage_path,
        overwrite=overwrite,
    )
