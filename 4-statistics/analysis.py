"""Adapted from upstream results_Statistics.ipynb at c85d68e. See README.md."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal as scipy_kruskal, spearmanr
import os

# -----------------------------
# Publication-ready visual style
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})
sns.set_style("white")

INTERNAL_COLOR = "#1f4e79"
EXTERNAL_COLOR = "#b85c00"

METHOD_COLORS = {
    "Patch": "#1f4e79",
    "Average": "#b85c00",
    "Max": "#8c564b",
}

FM_ORDER = ["Virchow2", "Virchow", "Conch v1.5", "UNI v2", "Phikon v2"]
FM_COLORS = {
    "Virchow2": "#c8a2c8",
    "Virchow": "#3ab09e",
    "Conch v1.5": "#ffbf00",
    "UNI v2": "#ab274f",
    "Phikon v2": "#8db600",
}

EXTERNAL_METRIC_COL = "external_test_no_ssl_mcc"  # displayed as "External Test"


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x)) if len(x) else np.nan
    sem = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    return mean, sem


def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def p_to_stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=8)


def pairwise_mwu_holm(groups_dict):
    keys = list(groups_dict.keys())
    rows = []
    pvals = []
    pairs = []
    for a, b in itertools.combinations(keys, 2):
        xa = np.asarray(groups_dict[a], dtype=float)
        xb = np.asarray(groups_dict[b], dtype=float)
        stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
        pvals.append(p)
        pairs.append((a, b, stat, p, xa, xb))
    if len(pvals) == 0:
        return pd.DataFrame()
    p_adj = holm_correction(pvals)
    for (a, b, stat, p_raw, xa, xb), pa in zip(pairs, p_adj):
        rows.append({
            "a": a,
            "b": b,
            "n_a": len(xa),
            "n_b": len(xb),
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "u_stat": float(stat),
            "p_raw": float(p_raw),
            "p_holm": float(pa),
            "stars": p_to_stars(pa) if p_to_stars(pa) else "ns",
        })
    return pd.DataFrame(rows).sort_values(["p_holm", "p_raw"]).reset_index(drop=True)

def fm_from_feature_subdir(x):
    name = Path(str(x)).name
    mapping = {
        "features_virchow2": "Virchow2",
        "features_virchow": "Virchow",
        "features_conch_v15": "Conch v1.5",
        "features_uni_v2": "UNI v2",
        "features_phikon_v2": "Phikon v2",
    }
    return mapping.get(name, name)

def fm_from_fm_name(x):
    s = str(x).lower()
    mapping = {
        "virchow2": "Virchow2",
        "virchow": "Virchow",
        "conch_v15": "Conch v1.5",
        "conch v1.5": "Conch v1.5",
        "uni_v2": "UNI v2",
        "uni v2": "UNI v2",
        "phikon_v2": "Phikon v2",
        "phikon v2": "Phikon v2",
    }
    return mapping.get(s, np.nan)


def display(value):
    print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)


def kruskal(*groups):
    # SciPy raises for all-identical values; the comparison is undefined.
    values = np.concatenate(groups)
    if np.unique(values).size < 2:
        return np.nan, np.nan
    return scipy_kruskal(*groups)


def configuration_label(cfg):
    return (f"{cfg['num_patches']} patches | {cfg['n_heads']} heads | "
            f"head={cfg['head_dim']} hidden={cfg['hidden_dim']} | "
            f"dropout={cfg['dropout']:g} gated={not cfg['no_gated']} | "
            f"lr={cfg['lr']:g} wd={cfg['weight_decay']:g}")


def render_reports(runs_csv, fm_csv, layers_csv, top64_csv, output, configs,
                   include_illustration=False):
    """Adapted upstream cells 1–10. Undefined comparisons stay NaN or are skipped."""
    OUT_DIR = Path(output)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_CSV = Path(runs_csv)
    FM_SHIFT_LONG_CSV = Path(fm_csv)
    ABMIL_LAYER_LONG_CSV = Path(layers_csv)
    ABMIL_TOPK_LONG_CSV = Path(top64_csv)
    df = pd.read_csv(RUNS_CSV)
    abmil = df[(df['mode'] == 'patch') & (df['tile_pooling'] == 'abmil')]
    present = set(abmil['feature_subdir'].map(fm_from_feature_subdir))
    FM_ORDER = [fm for fm in globals()['FM_ORDER'] if fm in present]
    if not FM_ORDER:
        raise ValueError('No ABMIL runs for the five foundation models supported upstream')
    panel_status = []

    def parse_knobs(run):
        return configs[run]

    def identify_abmil_configuration(run):
        return configuration_label(configs[run])

    # Adapted upstream notebook cell 1.
    # Panel - Internal vs external MCC by foundation model

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RUNS_CSV)


    panel_df = df.copy()
    panel_df["fm"] = panel_df["feature_subdir"].map(fm_from_feature_subdir)

    panel_df = panel_df[
        (panel_df["mode"] == "patch")
        & (panel_df["tile_pooling"] == "abmil")
        & (panel_df["fm"].isin(FM_ORDER))
    ].copy()

    summary_rows = []
    for split, col in [
        ("Internal test", "internal_test_mcc"),
        ("External test", EXTERNAL_METRIC_COL),
    ]:
        for fm in FM_ORDER:
            vals = panel_df.loc[panel_df["fm"] == fm, col].dropna().to_numpy(float)
            summary_rows.append({
                "split": split,
                "fm": fm,
                "n": len(vals),
                "mean": vals.mean(),
                "sem": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
                "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            })

    panel1A_summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    split_order = ["Internal test", "External test"]
    x = np.arange(len(split_order))
    width = 0.15

    for i, fm in enumerate(FM_ORDER):
        fm_summary = panel1A_summary[panel1A_summary["fm"] == fm].set_index("split").loc[split_order]
        xpos = x + (i - (len(FM_ORDER) - 1) / 2) * width

        ax.bar(
            xpos,
            fm_summary["mean"].to_numpy(),
            width=width,
            yerr=fm_summary["std"].fillna(0).to_numpy(),
            color=FM_COLORS[fm],
            edgecolor="white",
            linewidth=0.8,
            capsize=3,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
            label=fm,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(split_order)
    ax.set_xlabel("")
    ax.set_ylabel("MCC")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(-0.55, 1.55)

    ax.legend(
        frameon=False,
        title="",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    sns.despine(fig=fig)

    fig.savefig(OUT_DIR / "panel1A_internal_external_mcc_bar_sem.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "panel1A_internal_external_mcc_bar_sem.png", bbox_inches="tight")
    plt.close("all")

    panel1A_summary.to_csv(OUT_DIR / "panel1A_internal_external_mcc_summary.csv", index=False)
    panel1A_summary
    panel_status.append({'cell': 1, 'status': 'completed'})

    # Adapted upstream notebook cell 2.
    # Panel - Development-to-local MCC drop by FM and training/aggregation setting

    import re
    from matplotlib.lines import Line2D

    panel_df = df.copy()
    panel_df["fm"] = panel_df["feature_subdir"].map(fm_from_feature_subdir)

    panel_df = panel_df[
        (panel_df["mode"] == "patch")
        & (panel_df["tile_pooling"] == "abmil")
        & (panel_df["fm"].isin(FM_ORDER))
    ].copy()

    panel_df["mcc_drop"] = panel_df["internal_test_mcc"] - panel_df[EXTERNAL_METRIC_COL]
    panel_df["fm"] = pd.Categorical(panel_df["fm"], categories=FM_ORDER, ordered=True)

    config_order = sorted({identify_abmil_configuration(name) for name in panel_df['run_name']})
    config_markers = {name: ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h', '*', 'p'][i % 12]
                      for i, name in enumerate(config_order)}
    config_offsets = dict(zip(config_order, np.linspace(-0.25, 0.25, len(config_order))))

    panel_df["abmil_config"] = panel_df["run_name"].map(identify_abmil_configuration)
    unexpected_configs = sorted(set(panel_df["abmil_config"]) - set(config_order))
    if unexpected_configs:
        raise ValueError(f"Unexpected ABMIL configurations: {unexpected_configs}")

    fig, ax = plt.subplots(figsize=(8.4, 4.5))

    for fm_index, fm in enumerate(FM_ORDER):
        fm_rows = panel_df[panel_df["fm"] == fm]
        for row in fm_rows.itertuples(index=False):
            ax.scatter(
                fm_index + config_offsets[row.abmil_config],
                row.mcc_drop,
                marker=config_markers[row.abmil_config],
                s=82,
                color=FM_COLORS[fm],
                edgecolor="white",
                linewidth=0.65,
                alpha=0.9,
                zorder=3,
            )

    ax.axhline(0, color="0.25", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(np.arange(len(FM_ORDER)))
    ax.set_xticklabels(FM_ORDER, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("MCC drop")

    config_handles = [
        Line2D(
            [0], [0],
            marker=config_markers[config],
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=8,
            label=config,
        )
        for config in config_order
    ]
    # ax.legend(
    #     handles=config_handles,
    #     title="Training/aggregation setting",
    #     frameon=False,
    #     bbox_to_anchor=(1.02, 0.5),
    #     loc="center left",
    #     borderaxespad=0,
    #     ncol=1,
    # )

    sns.despine(fig=fig)
    fig.tight_layout()

    fig.savefig(OUT_DIR / "panel1B_mcc_drop_by_fm.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "panel1B_mcc_drop_by_fm.png", bbox_inches="tight", dpi=300)
    plt.close("all")

    panel1B_summary = (
        panel_df
        .groupby("fm", observed=True)
        .agg(
            n_runs=("run_name", "nunique"),
            mcc_drop_mean=("mcc_drop", "mean"),
            mcc_drop_sem=("mcc_drop", lambda x: mean_sem(x)[1]),
            mcc_drop_std=("mcc_drop", "std"),
            mcc_drop_median=("mcc_drop", "median"),
            mcc_drop_min=("mcc_drop", "min"),
            mcc_drop_max=("mcc_drop", "max"),
        )
        .reindex(FM_ORDER)
        .reset_index()
    )

    panel1B_config_summary = (
        panel_df[["fm", "abmil_config", "run_name", "internal_test_mcc", EXTERNAL_METRIC_COL, "mcc_drop"]]
        .sort_values(["fm", "abmil_config"])
        .reset_index(drop=True)
    )

    panel1B_summary.to_csv(OUT_DIR / "panel1B_mcc_drop_by_fm_summary.csv", index=False)
    panel1B_config_summary.to_csv(OUT_DIR / "panel1B_mcc_drop_by_fm_and_config_summary.csv", index=False)
    display(panel1B_summary)
    panel1B_config_summary
    panel_status.append({'cell': 2, 'status': 'completed'})

    # Adapted upstream notebook cell 3.
    # Supplementary figure - Method comparison: ABMIL vs Average vs Max


    df = pd.read_csv(RUNS_CSV)

    required = {"method_family", "internal_test_mcc", EXTERNAL_METRIC_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns for method comparison: {required - set(df.columns)}")

    method_map = {
        "patch_abmil": "ABMIL",
        "patch_mean": "Average",
        "patch_max": "Max",
    }

    method_groups = [m for m in ["ABMIL", "Average", "Max"] if m in set(df["method_family"].map(method_map))]

    method_colors = {
        "ABMIL": "#1f4e79",
        "Average": "#b85c00",
        "Max": "#8c564b",
    }

    method_df = df.copy()
    method_df["method_label"] = (
        method_df["method_family"]
        .astype(str)
        .str.lower()
        .map(method_map)
    )

    # Keep only patch methods, exclude slide-level models, PRISM/TITAN by construction.
    method_df = method_df[method_df["method_label"].isin(method_groups)].copy()

    method_internal = []
    method_external = []

    for g in method_groups:
        sub = method_df[method_df["method_label"] == g]
        method_internal.append(sub["internal_test_mcc"].dropna().to_numpy(float))
        method_external.append(sub[EXTERNAL_METRIC_COL].dropna().to_numpy(float))

    int_stats = [mean_sem(x) for x in method_internal]
    ext_stats = [mean_sem(x) for x in method_external]

    int_mean = [m for m, _ in int_stats]
    int_sem = [s for _, s in int_stats]
    ext_mean = [m for m, _ in ext_stats]
    ext_sem = [s for _, s in ext_stats]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    centers = np.array([0.0, 1.0])
    bar_w = 0.20
    offsets = (np.arange(len(method_groups)) - (len(method_groups) - 1) / 2) * bar_w

    for i, g in enumerate(method_groups):
        ax.bar(
            centers[0] + offsets[i],
            int_mean[i],
            bar_w,
            yerr=np.nan_to_num(int_sem[i]),
            capsize=3,
            color=method_colors[g],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
            label=g,
        )
        ax.bar(
            centers[1] + offsets[i],
            ext_mean[i],
            bar_w,
            yerr=np.nan_to_num(ext_sem[i]),
            capsize=3,
            color=method_colors[g],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(["Internal test", "External test"])
    ax.set_xlabel("")
    ax.set_ylabel("MCC")

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=method_colors[g],
            edgecolor="white",
            linewidth=0.8,
        )
        for g in method_groups
    ]
    ax.legend(
        handles,
        method_groups,
        frameon=False,
        title="",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    def annotate_within_set(values_list, x_center, means, sems):
        groups_dict = {
            method_groups[i]: values_list[i]
            for i in range(len(method_groups))
            if len(values_list[i]) > 0
        }

        pw = pairwise_mwu_holm(groups_dict)
        sig = pw[pw["p_holm"] < 0.05].copy() if len(pw) > 0 else pd.DataFrame()

        y0 = max(np.array(means) + np.nan_to_num(sems)) + 0.015
        step = 0.030
        h = 0.007

        for k, row in enumerate(sig.itertuples(index=False)):
            i = method_groups.index(row.a)
            j = method_groups.index(row.b)
            x1 = x_center + offsets[i]
            x2 = x_center + offsets[j]
            add_sig_bracket(ax, x1, x2, y0 + k * step, h, p_to_stars(row.p_holm))

        y_top = y0 + max(0, len(sig) - 1) * step + 0.04
        return pw, y_top

    pw_int, top_int = annotate_within_set(method_internal, centers[0], int_mean, int_sem)
    pw_ext, top_ext = annotate_within_set(method_external, centers[1], ext_mean, ext_sem)

    ax.set_ylim(-1.0, max(1.0, top_int, top_ext))
    sns.despine(fig=fig)

    fig.savefig(OUT_DIR / "supp_method_comparison_abmil_average_max.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "supp_method_comparison_abmil_average_max.png", bbox_inches="tight")
    plt.close("all")

    method_summary_rows = []
    for split_label, values_list in [
        ("Internal test", method_internal),
        ("External test", method_external),
    ]:
        for method, vals in zip(method_groups, values_list):
            method_summary_rows.append({
                "split": split_label,
                "method": method,
                "n": len(vals),
                "mean_mcc": float(np.mean(vals)),
                "sem_mcc": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "std_mcc": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "median_mcc": float(np.median(vals)),
                "min_mcc": float(np.min(vals)),
                "max_mcc": float(np.max(vals)),
            })

    method_summary_df = pd.DataFrame(method_summary_rows)
    method_summary_df.to_csv(OUT_DIR / "supp_method_comparison_abmil_average_max_summary.csv", index=False)

    pw_int_out = pw_int.copy()
    pw_int_out.insert(0, "split", "Internal test")

    pw_ext_out = pw_ext.copy()
    pw_ext_out.insert(0, "split", "External test")

    method_pairwise_df = pd.concat([pw_int_out, pw_ext_out], ignore_index=True)
    method_pairwise_df.to_csv(OUT_DIR / "supp_method_comparison_abmil_average_max_pairwise_holm.csv", index=False)

    print("Summary")
    display(method_summary_df)

    print("Pairwise statistics, Holm-corrected")
    display(method_pairwise_df)
    panel_status.append({'cell': 3, 'status': 'completed'})

    # Adapted upstream notebook cell 4.
    if any(abmil['run_name'].map(lambda name: configs[name][key]).nunique() > 1 for key in ('num_patches', 'no_gated', 'lr', 'weight_decay', 'dropout')):
        # effect of hyperparameters on ABMIL performance

        ab = df.copy()
        ab["fm"] = ab["feature_subdir"].map(fm_from_feature_subdir)

        ab = ab[
            (ab["mode"] == "patch")
            & (ab["tile_pooling"] == "abmil")
            & (ab["fm"].isin(FM_ORDER))
        ].copy()

        parsed = ab["run_name"].apply(parse_knobs).apply(pd.Series).add_prefix("parsed_")
        ab = pd.concat([ab.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

        ab = ab.dropna(subset=["parsed_num_patches"]).copy()
        ab["num_patches_lbl"] = ab["parsed_num_patches"].astype(int).astype(str)
        ab["no_gated_lbl"] = ab["parsed_no_gated"].map({False: "Gated", True: "No-gated"})
        ab["lr_lbl"] = ab["parsed_lr"].map(lambda x: str(x))
        ab["weight_decay_lbl"] = ab["parsed_weight_decay"].map(lambda x: str(x))
        ab["dropout_lbl"] = ab["parsed_dropout"].map(lambda x: str(x))

        knob_defs = [(col, label, sorted(ab[col].unique())) for col, label in [
            ("num_patches_lbl", "Number of patches"), ("no_gated_lbl", "Gating"),
            ("lr_lbl", "Learning rate"), ("weight_decay_lbl", "Weight decay"),
            ("dropout_lbl", "Dropout")]]

        endpoint_defs = [
            ("internal_test_mcc", "Development test"),
            (EXTERNAL_METRIC_COL, "Local validation"),
        ]

        fig, axes = plt.subplots(
            len(endpoint_defs),
            len(knob_defs),
            figsize=(3.0 * len(knob_defs), 6.4),
            sharey="row",
        )

        global_rows = []
        pairwise_rows = []
        summary_rows = []

        for r, (metric_col, endpoint_label) in enumerate(endpoint_defs):
            for c, (knob_col, knob_label, order) in enumerate(knob_defs):
                ax = axes[r, c]
                sub = ab[[knob_col, metric_col]].dropna().copy()
                levels = [lv for lv in order if lv in set(sub[knob_col])]

                if len(levels) < 2:
                    ax.axis("off")
                    continue

                groups_dict = {
                    lv: sub.loc[sub[knob_col] == lv, metric_col].to_numpy(float)
                    for lv in levels
                }

                means = np.array([np.mean(groups_dict[lv]) for lv in levels])
                sems = np.array([
                    np.std(groups_dict[lv], ddof=1) / np.sqrt(len(groups_dict[lv]))
                    if len(groups_dict[lv]) > 1 else 0.0
                    for lv in levels
                ])

                for lv, vals, mean, sem in zip(levels, groups_dict.values(), means, sems):
                    summary_rows.append({
                        "endpoint": endpoint_label,
                        "knob": knob_label,
                        "level": lv,
                        "n": len(vals),
                        "mean": float(mean),
                        "sem": float(sem),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                        "median": float(np.median(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    })

                arrs = [groups_dict[lv] for lv in levels]
                if len(levels) == 2:
                    stat, p_global = mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
                    test_name = "MWU"
                else:
                    stat, p_global = kruskal(*arrs)
                    test_name = "KW"

                global_rows.append({
                    "endpoint": endpoint_label,
                    "knob": knob_label,
                    "test": test_name,
                    "stat": float(stat),
                    "p_global": float(p_global),
                    "stars": p_to_stars(p_global) if p_to_stars(p_global) else "ns",
                })

                pw = pairwise_mwu_holm(groups_dict)
                if len(pw) > 0:
                    pw = pw.copy()
                    pw.insert(0, "endpoint", endpoint_label)
                    pw.insert(1, "knob", knob_label)
                    pairwise_rows.append(pw)

                x = np.arange(len(levels))
                colors = sns.color_palette("Set2", n_colors=len(levels))

                ax.bar(
                    x,
                    means,
                    yerr=sems,
                    color=colors,
                    edgecolor="black",
                    linewidth=1,
                    capsize=3,
                    error_kw={
                        "ecolor": "black",
                        "elinewidth": 1.0,
                        "capthick": 1.0,
                    },
                    zorder=2,
                )

                sig = pw[pw["p_holm"] < 0.05] if len(pw) > 0 else pd.DataFrame()
                y_base = max(means + sems) + 0.015
                step = 0.035
                h = 0.008

                for k, row in enumerate(sig.itertuples(index=False)):
                    i = levels.index(row.a)
                    j = levels.index(row.b)
                    add_sig_bracket(ax, i, j, y_base + k * step, h, p_to_stars(row.p_holm))

                y_top = max(0.98, y_base + max(0, len(sig) - 1) * step + 0.045)
                ax.set_ylim(min(-0.05, float(np.min(means - sems)) - 0.05), y_top)
                if r == 0:
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                if r == 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels(levels)
                    ax.set_xlabel(knob_label)

                if c == 0:
                    ax.set_ylabel(f"{endpoint_label} MCC")
                else:
                    ax.set_ylabel("")

                ax.text(
                    0.03,
                    0.97,
                    f"{test_name} p={p_global:.3g}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                )

                sns.despine(ax=ax)

        plt.tight_layout()

        fig.savefig(OUT_DIR / "supp_abmil_hyperparameter_effects.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "supp_abmil_hyperparameter_effects.png", bbox_inches="tight")
        plt.close("all")

        global_stats_df = pd.DataFrame(global_rows).sort_values(["endpoint", "p_global", "knob"]).reset_index(drop=True)
        pairwise_stats_df = pd.concat(pairwise_rows, ignore_index=True) if len(pairwise_rows) else pd.DataFrame()
        knob_summary_df = pd.DataFrame(summary_rows)

        global_stats_df.to_csv(OUT_DIR / "supp_abmil_hyperparameter_global_tests.csv", index=False)
        pairwise_stats_df.to_csv(OUT_DIR / "supp_abmil_hyperparameter_pairwise_holm_tests.csv", index=False)
        knob_summary_df.to_csv(OUT_DIR / "supp_abmil_hyperparameter_summary.csv", index=False)

        print("Global knob tests")
        display(global_stats_df)

        print("Pairwise knob tests, Holm-corrected")
        display(pairwise_stats_df)

        print("Summary")
        display(knob_summary_df)
        panel_status.append({'cell': 4, 'status': 'completed'})
    else:
        panel_status.append({'cell': 4, 'status': 'skipped', 'reason': 'No varying hyperparameters.'})

    # Adapted upstream notebook cell 5.
    # Supplementary figure - Prediction-output shift vs MCC drop




    df = pd.read_csv(RUNS_CSV)

    pred_df = df.copy()
    pred_df["fm"] = pred_df["feature_subdir"].map(fm_from_feature_subdir)

    pred_df = pred_df[
        (pred_df["mode"] == "patch")
        & (pred_df["tile_pooling"] == "abmil")
        & (pred_df["fm"].isin(FM_ORDER))
    ].copy()

    required_cols = {
        "internal_test_mcc",
        EXTERNAL_METRIC_COL,
        "entropy_shift",
        "confidence_drop",
        "margin_drop",
    }
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pred_df["mcc_drop"] = pred_df["internal_test_mcc"] - pred_df[EXTERNAL_METRIC_COL]

    output_shift_defs = [
        ("entropy_shift", "Entropy shift", "Prediction entropy"),
        ("confidence_drop", "Confidence drop", "Maximum probability"),
        ("margin_drop", "Margin drop", "Prediction margin"),
    ]

    corr_rows = []

    for metric_col, measure_label, feature_label in output_shift_defs:
        sub = pred_df[[metric_col, "mcc_drop", "fm", "run_name"]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(sub) < 3 or sub[metric_col].nunique() < 2 or sub["mcc_drop"].nunique() < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[metric_col], sub["mcc_drop"])

        corr_rows.append({
            "measure": measure_label,
            "feature_used": feature_label,
            "n_runs": len(sub),
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "p_value": float(pval) if pd.notna(pval) else np.nan,
            "rho_p_display": "" if pd.isna(rho) else f"{rho:.2f} (p={pval:.3g})",
        })

    corr_output_df = pd.DataFrame(corr_rows)
    corr_output_df.to_csv(OUT_DIR / "supp_prediction_output_shift_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)

    for ax, (metric_col, measure_label, feature_label) in zip(axes, output_shift_defs):
        sub = pred_df[[metric_col, "mcc_drop", "fm"]].replace([np.inf, -np.inf], np.nan).dropna()

        sns.scatterplot(
            data=sub,
            x=metric_col,
            y="mcc_drop",
            hue="fm",
            hue_order=FM_ORDER,
            palette=FM_COLORS,
            s=48,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            legend=False,
            ax=ax,
        )

        if len(sub) >= 3 and sub[metric_col].nunique() > 1 and sub["mcc_drop"].nunique() > 1:
            rho, pval = spearmanr(sub[metric_col], sub["mcc_drop"])
            ax.text(
                0.94,
                0.96,
                f"rho={rho:.2f}\np={pval:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=12,
            )

        ax.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
        ax.set_xlabel(measure_label)
        ax.set_ylabel("MCC drop" if ax is axes[0] else "")

    sns.despine(fig=fig)
    plt.tight_layout()

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FM_COLORS[fm],
            markeredgecolor="white",
            markersize=7,
            label=fm,
        )
        for fm in FM_ORDER
    ]

    fig.legend(
        handles=handles,
        labels=FM_ORDER,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    fig.savefig(OUT_DIR / "supp_prediction_output_shift_vs_mcc_drop.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "supp_prediction_output_shift_vs_mcc_drop.png", bbox_inches="tight")
    plt.close("all")

    display(corr_output_df)
    panel_status.append({'cell': 5, 'status': 'completed'})

    # Adapted upstream notebook cell 6.
    # Panel 2 - Repeated run-level correlations between shift metrics and MCC drop
    # Includes ABMIL representation-level metrics and raw FM feature-level metrics.
    # Raw FM shift metrics are broadcast to every ABMIL run from the same FM and repetition.
    # Reports Spearman rho as median [Q1, Q3] across repeated matched subsets.






    missing = [
        p for p in [
            ABMIL_LAYER_LONG_CSV,
            ABMIL_TOPK_LONG_CSV,
            FM_SHIFT_LONG_CSV,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    abmil_long = pd.concat(
        [
            pd.read_csv(ABMIL_LAYER_LONG_CSV),
            pd.read_csv(ABMIL_TOPK_LONG_CSV),
        ],
        ignore_index=True,
    )

    fm_long = pd.read_csv(FM_SHIFT_LONG_CSV)

    # -----------------------------
    # ABMIL representation-level data
    # -----------------------------
    abmil_long["fm"] = abmil_long["feature_subdir"].map(fm_from_feature_subdir)
    abmil_long = abmil_long[abmil_long["fm"].isin(FM_ORDER)].copy()

    required_abmil_cols = {
        "run_name",
        "feature_subdir",
        "fm",
        "layer_name",
        "repetition",
        "mcc_drop",
        "mmd_raw",
        "fid_raw",
        "coral_raw",
        "energy_raw",
        "knn_domain_mixing",
    }
    missing_abmil_cols = required_abmil_cols - set(abmil_long.columns)
    if missing_abmil_cols:
        raise ValueError(f"Missing ABMIL columns: {sorted(missing_abmil_cols)}")

    # -----------------------------
    # Raw FM feature-level data
    # -----------------------------
    fm_long["fm"] = fm_long["fm_name"].map(fm_from_fm_name)
    fm_long = fm_long[fm_long["fm"].isin(FM_ORDER)].copy()

    required_fm_cols = {
        "fm",
        "repetition",
        "mmd_raw",
        "fid_raw",
        "coral_raw",
        "energy_raw",
        "knn_domain_mixing",
    }
    missing_fm_cols = required_fm_cols - set(fm_long.columns)
    if missing_fm_cols:
        raise ValueError(f"Missing FM-shift columns: {sorted(missing_fm_cols)}")

    # Use one repeated MCC-drop value per ABMIL run/repetition.
    # mcc_drop is duplicated across layer rows, so use one layer to avoid duplicating runs.
    if "penultimate" in set(abmil_long["layer_name"]):
        mcc_source_layer = "penultimate"
    else:
        mcc_source_layer = sorted(abmil_long["layer_name"].dropna().unique().tolist())[0]

    run_rep_mcc = (
        abmil_long[abmil_long["layer_name"] == mcc_source_layer]
        [
            [
                "run_name",
                "fm",
                "feature_subdir",
                "repetition",
                "mcc_drop",
            ]
        ]
        .drop_duplicates(["run_name", "repetition"])
        .copy()
    )

    # Broadcast each FM-level shift metric to every ABMIL run from that FM at the same repetition.
    fm_level_long = run_rep_mcc.merge(
        fm_long[
            [
                "fm",
                "repetition",
                "mmd_raw",
                "fid_raw",
                "coral_raw",
                "energy_raw",
                "knn_domain_mixing",
            ]
        ],
        on=["fm", "repetition"],
        how="inner",
    )

    fm_level_long["layer_name"] = "raw_fm_features"

    # Combine ABMIL layers + broadcast raw FM feature level.
    combined_long = pd.concat(
        [
            abmil_long[
                [
                    "run_name",
                    "fm",
                    "feature_subdir",
                    "layer_name",
                    "repetition",
                    "mcc_drop",
                    "mmd_raw",
                    "fid_raw",
                    "coral_raw",
                    "energy_raw",
                    "knn_domain_mixing",
                ]
            ],
            fm_level_long[
                [
                    "run_name",
                    "fm",
                    "feature_subdir",
                    "layer_name",
                    "repetition",
                    "mcc_drop",
                    "mmd_raw",
                    "fid_raw",
                    "coral_raw",
                    "energy_raw",
                    "knn_domain_mixing",
                ]
            ],
        ],
        ignore_index=True,
    )

    feature_order = [
        "raw_fm_features",
        "pre_attention_mean",
        "post_attention",
        "penultimate",
        "topk64_positive_evidence",
    ]

    feature_labels = {
        "raw_fm_features": "Raw FM features",
        "pre_attention_mean": "Pre-attention mean",
        "post_attention": "Post-attention",
        "penultimate": "Penultimate",
        "topk64_positive_evidence": "Top-64 positive evidence",
    }

    metric_defs = [
        ("mmd_raw", "MMD"),
        ("fid_raw", "FID"),
        ("coral_raw", "CORAL"),
        ("energy_raw", "Energy"),
        ("knn_domain_mixing", "1 - kNN mixing"),
    ]

    available_features = [x for x in feature_order if x in set(combined_long["layer_name"])]
    available_metrics = [(col, lab) for col, lab in metric_defs if col in combined_long.columns]

    corr_rows = []
    repeat_rows = []

    for feature in available_features:
        feature_df = combined_long[combined_long["layer_name"] == feature].copy()

        for metric_col, measure_label in available_metrics:
            tmp = feature_df[["run_name", "repetition", metric_col, "mcc_drop"]].copy()
            tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

            if metric_col == "knn_domain_mixing":
                tmp["metric_value"] = 1.0 - tmp[metric_col]
            else:
                tmp["metric_value"] = tmp[metric_col]

            rhos = []
            ns = []

            for rep, rep_df in tmp.groupby("repetition", sort=True):
                rep_df = rep_df.dropna(subset=["metric_value", "mcc_drop"])

                if (
                    len(rep_df) < 3
                    or rep_df["metric_value"].nunique() < 2
                    or rep_df["mcc_drop"].nunique() < 2
                ):
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = spearmanr(rep_df["metric_value"], rep_df["mcc_drop"])

                rhos.append(rho)
                ns.append(len(rep_df))

                repeat_rows.append({
                    "measure": measure_label,
                    "feature_used": feature_labels.get(feature, feature),
                    "layer_name": feature,
                    "metric_col": metric_col,
                    "repetition": rep,
                    "n_observations": len(rep_df),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "p_value": float(pval) if pd.notna(pval) else np.nan,
                })

            rhos = np.asarray(rhos, dtype=float)
            valid_rhos = rhos[np.isfinite(rhos)]

            if len(valid_rhos) == 0:
                med = q1 = q3 = iqr = np.nan
            else:
                med = float(np.nanmedian(valid_rhos))
                q1 = float(np.nanquantile(valid_rhos, 0.25))
                q3 = float(np.nanquantile(valid_rhos, 0.75))
                iqr = float(q3 - q1)

            corr_rows.append({
                "measure": measure_label,
                "feature_used": feature_labels.get(feature, feature),
                "layer_name": feature,
                "metric_col": metric_col,
                "n_repeats": int(len(valid_rhos)),
                "median_spearman_rho": med,
                "q1_spearman_rho": q1,
                "q3_spearman_rho": q3,
                "iqr_spearman_rho": iqr,
                "rho_display": "" if pd.isna(med) else f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
                "median_n_observations": float(np.nanmedian(ns)) if len(ns) else np.nan,
            })

    corr_table = pd.DataFrame(corr_rows)
    repeat_corr_df = pd.DataFrame(repeat_rows)

    measure_order = [x[1] for x in metric_defs]
    feature_label_order = [feature_labels[x] for x in available_features]

    corr_table["measure"] = pd.Categorical(corr_table["measure"], categories=measure_order, ordered=True)
    corr_table["feature_used"] = pd.Categorical(corr_table["feature_used"], categories=feature_label_order, ordered=True)
    corr_table = corr_table.sort_values(["measure", "feature_used"]).reset_index(drop=True)

    repeat_corr_df["measure"] = pd.Categorical(repeat_corr_df["measure"], categories=measure_order, ordered=True)
    repeat_corr_df["feature_used"] = pd.Categorical(repeat_corr_df["feature_used"], categories=feature_label_order, ordered=True)

    corr_table.to_csv(OUT_DIR / "panel2_shift_metric_correlation_table_with_raw_fm.csv", index=False)
    repeat_corr_df.to_csv(OUT_DIR / "panel2_shift_metric_repeat_correlations_with_raw_fm.csv", index=False)

    display(
        corr_table[
            [
                "measure",
                "feature_used",
                "rho_display",
                "n_repeats",
                "median_n_observations",
            ]
        ]
    )

    # Figure: repeat-wise Spearman correlations.
    plot_df = repeat_corr_df.dropna(subset=["spearman_rho"]).copy()
    plot_df["feature_used"] = plot_df["feature_used"].replace({
        "Raw FM features": "Raw FM",
        "Pre-attention mean": "Pre-attention",
        "Top-k evidence": "Top-64 evidence",
        "Top-64 positive evidence": "Top-64 evidence",
    })
    plot_df["measure"] = plot_df["measure"].replace({"1 - kNN mixing": "1 - kNN\nmixing"})

    measure_order_plot = ["MMD", "FID", "CORAL", "Energy", "1 - kNN\nmixing"]
    feature_label_order_plot = [
        "Raw FM",
        "Pre-attention",
        "Post-attention",
        "Penultimate",
        "Top-64 evidence",
    ]
    feature_palette = {
        "Raw FM": "#66c2a5",
        "Pre-attention": "#fc8d62",
        "Post-attention": "#8da0cb",
        "Penultimate": "#e78ac3",
        "Top-64 evidence": "#a6d854",
    }

    if not plot_df.empty:
        with sns.plotting_context("paper", font_scale=1.25):
            fig, ax = plt.subplots(figsize=(5.7, 5.0))

            sns.boxplot(
                data=plot_df,
                x="measure",
                y="spearman_rho",
                hue="feature_used",
                order=measure_order_plot,
                hue_order=feature_label_order_plot,
                palette=feature_palette,
                width=0.72,
                linewidth=1.25,
                fliersize=2.6,
                ax=ax,
            )

            for patch in ax.patches:
                patch.set_edgecolor("0.35")
                patch.set_linewidth(1.25)
            for line in ax.lines:
                line.set_color("0.35")
                line.set_linewidth(1.25)

            ax.axhline(0, color="0.25", linewidth=1.0, linestyle="--", zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel("Spearman rho", fontweight="bold")
            ax.set_ylim(-1.0, 1.0)
            ax.grid(axis="y", color="0.88", linewidth=0.8)
            ax.grid(axis="x", visible=False)

            for side in ["left", "bottom"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(1.7)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)

            ax.tick_params(axis="both", width=1.4, length=4, color="black")

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                frameon=False,
                title="",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=2,
                columnspacing=1.8,
                handlelength=1.5,
                handletextpad=0.5,
            )

            fig.savefig(OUT_DIR / "panel2_shift_metric_repeat_correlation_boxplots_with_raw_fm.pdf", bbox_inches="tight")
            fig.savefig(OUT_DIR / "panel2_shift_metric_repeat_correlation_boxplots_with_raw_fm.png", dpi=300, bbox_inches="tight")
            plt.close("all")
    else:
        panel_status.append({'panel': '2 correlation figure', 'status': 'skipped',
                             'reason': 'No estimable correlations; CSVs retain NaN values.'})
    panel_status.append({'cell': 6, 'status': 'completed'})

    performance_fm_order = FM_ORDER
    counts = abmil.assign(fm=abmil['feature_subdir'].map(fm_from_feature_subdir)).groupby('fm')['run_name'].nunique()
    FM_ORDER = [fm for fm in FM_ORDER if counts.get(fm, 0) >= 2]
    # Adapted upstream notebook cell 7.
    if bool(FM_ORDER):
        # Panel 3 - Repetition-matched FM shift vs ABMIL stability
        # Stability is computed per FM x repetition from repeated ABMIL mcc_drop values.
        # One scatter plot per shift metric, colored by FM.







        abmil_long = pd.concat(
            [pd.read_csv(ABMIL_LAYER_LONG_CSV), pd.read_csv(ABMIL_TOPK_LONG_CSV)],
            ignore_index=True,
        )
        fm_shift_long = pd.read_csv(FM_SHIFT_LONG_CSV)

        abmil_long["fm"] = abmil_long["feature_subdir"].map(fm_from_feature_subdir)
        abmil_long = abmil_long[abmil_long["fm"].isin(FM_ORDER)].copy()

        # mcc_drop is duplicated across layer rows; select one layer to avoid duplicate runs.
        mcc_source_layer = "penultimate" if "penultimate" in set(abmil_long["layer_name"]) else abmil_long["layer_name"].iloc[0]

        run_rep_mcc = (
            abmil_long[abmil_long["layer_name"] == mcc_source_layer]
            [["run_name", "fm", "repetition", "mcc_internal", "mcc_external", "mcc_drop"]]
            .drop_duplicates(["run_name", "repetition"])
            .copy()
        )

        fm_rep_stability = (
            run_rep_mcc
            .groupby(["fm", "repetition"], observed=True)
            .agg(
                n_runs=("run_name", "nunique"),
                mcc_drop_mean=("mcc_drop", "mean"),
                mcc_drop_sd=("mcc_drop", "std"),
                external_mcc_mean=("mcc_external", "mean"),
                external_mcc_sd=("mcc_external", "std"),
            )
            .reset_index()
        )

        fm_shift_long["fm"] = fm_shift_long["fm_name"].map(fm_from_fm_name)
        fm_shift_long = fm_shift_long[fm_shift_long["fm"].isin(FM_ORDER)].copy()
        fm_shift_long["one_minus_knn_mixing"] = 1.0 - fm_shift_long["knn_domain_mixing"]

        plot_df = fm_shift_long.merge(
            fm_rep_stability,
            on=["fm", "repetition"],
            how="inner",
        )

        shift_defs = [
            ("mmd_raw", "MMD"),
            ("fid_raw", "FID"),
            ("coral_raw", "CORAL"),
            ("energy_raw", "Energy"),
            ("one_minus_knn_mixing", "1 - kNN mixing"),
        ]

        target_col = "mcc_drop_sd"
        target_label = "MCC-drop instability"

        corr_rows = []

        for shift_col, shift_label in shift_defs:
            rhos = []
            ns = []

            for rep, rep_df in plot_df.groupby("repetition", sort=True):
                sub = rep_df[[shift_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

                if len(sub) < 3 or sub[shift_col].nunique() < 2 or sub[target_col].nunique() < 2:
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = spearmanr(sub[shift_col], sub[target_col])

                rhos.append(rho)
                ns.append(len(sub))

            valid = np.asarray(rhos, dtype=float)
            valid = valid[np.isfinite(valid)]

            med = np.nanmedian(valid) if len(valid) else np.nan
            q1 = np.nanquantile(valid, 0.25) if len(valid) else np.nan
            q3 = np.nanquantile(valid, 0.75) if len(valid) else np.nan

            corr_rows.append({
                "shift_measure": shift_label,
                "target": target_label,
                "n_repeats": len(valid),
                "median_spearman_rho": float(med),
                "q1_spearman_rho": float(q1),
                "q3_spearman_rho": float(q3),
                "rho_display": f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
                "median_n_fm": float(np.nanmedian(ns)),
            })

            fig, ax = plt.subplots(figsize=(4.6, 4.0))

            sns.scatterplot(
                data=plot_df,
                x=shift_col,
                y=target_col,
                hue="fm",
                hue_order=FM_ORDER,
                palette=FM_COLORS,
                s=46,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
                ax=ax,
            )

            ax.set_xlabel(shift_label)
            ax.set_ylabel(target_label)
            ax.legend(
                frameon=False,
                title="",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )


            sns.despine(fig=fig)

            safe_name = shift_label.lower().replace(" ", "_").replace("-", "minus")
            fig.savefig(OUT_DIR / f"panel3_fm_shift_vs_repeated_stability_{safe_name}.pdf", bbox_inches="tight")
            fig.savefig(OUT_DIR / f"panel3_fm_shift_vs_repeated_stability_{safe_name}.png", bbox_inches="tight")
            plt.close("all")

        stability_corr_df = pd.DataFrame(corr_rows)
        stability_corr_df.to_csv(OUT_DIR / "panel3_repeated_stability_shift_correlations.csv", index=False)
        plot_df.to_csv(OUT_DIR / "panel3_repeated_fm_stability_with_shift_metrics.csv", index=False)

        display(stability_corr_df)
        panel_status.append({'cell': 7, 'status': 'completed'})
    else:
        panel_status.append({'cell': 7, 'status': 'skipped', 'reason': 'Need at least two ABMIL runs per foundation model for instability.'})

    # Adapted upstream notebook cell 8.
    if bool(FM_ORDER):
        # Panel 3A - FM-level MCC-drop instability across ABMIL configurations
        # Instability per FM x repetition = SD of repeated mcc_drop across ABMIL runs/configs.
        # Bar height = mean instability across repetitions. Error bar = SEM across repetitions.






        missing = [p for p in [ABMIL_LAYER_LONG_CSV, ABMIL_TOPK_LONG_CSV] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

        abmil_long = pd.concat(
            [
                pd.read_csv(ABMIL_LAYER_LONG_CSV),
                pd.read_csv(ABMIL_TOPK_LONG_CSV),
            ],
            ignore_index=True,
        )

        abmil_long["fm"] = abmil_long["feature_subdir"].map(fm_from_feature_subdir)
        abmil_long = abmil_long[abmil_long["fm"].isin(FM_ORDER)].copy()

        required_cols = {"run_name", "fm", "layer_name", "repetition", "mcc_drop"}
        missing_cols = required_cols - set(abmil_long.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        # mcc_drop is repeated across layer rows, so select one layer to avoid duplicate runs.
        mcc_source_layer = "penultimate" if "penultimate" in set(abmil_long["layer_name"]) else abmil_long["layer_name"].iloc[0]

        run_rep_mcc = (
            abmil_long[abmil_long["layer_name"] == mcc_source_layer]
            [["run_name", "fm", "repetition", "mcc_drop"]]
            .drop_duplicates(["run_name", "repetition"])
            .copy()
        )

        fm_rep_instability = (
            run_rep_mcc
            .groupby(["fm", "repetition"], observed=True)
            .agg(
                n_runs=("run_name", "nunique"),
                mcc_drop_mean=("mcc_drop", "mean"),
                mcc_drop_instability=("mcc_drop", "std"),
            )
            .reset_index()
        )

        panel3A_summary = (
            fm_rep_instability
            .groupby("fm", observed=True)
            .agg(
                n_repeats=("repetition", "nunique"),
                n_runs_median=("n_runs", "median"),
                instability_mean=("mcc_drop_instability", "mean"),
                instability_sem=("mcc_drop_instability", lambda x: mean_sem(x)[1]),
                instability_std=("mcc_drop_instability", "std"),
                instability_median=("mcc_drop_instability", "median"),
                instability_q25=("mcc_drop_instability", lambda x: float(np.nanquantile(x, 0.25))),
                instability_q75=("mcc_drop_instability", lambda x: float(np.nanquantile(x, 0.75))),
            )
            .reindex(FM_ORDER)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(6.4, 4.2))

        x = np.arange(len(FM_ORDER))
        y = panel3A_summary["instability_mean"].to_numpy(float)
        err = panel3A_summary["instability_sem"].to_numpy(float)

        ax.bar(
            x,
            y,
            yerr=np.nan_to_num(err),
            color=[FM_COLORS[fm] for fm in FM_ORDER],
            edgecolor="white",
            linewidth=0.8,
            capsize=3,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.1,
                "capthick": 1.1,
            },
        )

        ax.set_xticks(x)
        ax.set_xticklabels(FM_ORDER, rotation=25)
        ax.set_xlabel("")
        ax.set_ylabel("MCC-drop instability")
        ax.set_ylim(0, max(0.01, float(np.nanmax(y + np.nan_to_num(err))) * 1.25))

        sns.despine(fig=fig)

        fig.savefig(OUT_DIR / "panel3A_fm_mcc_drop_instability_bar.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "panel3A_fm_mcc_drop_instability_bar.png", bbox_inches="tight")
        plt.close("all")

        fm_rep_instability.to_csv(OUT_DIR / "panel3A_fm_repetition_mcc_drop_instability.csv", index=False)
        panel3A_summary.to_csv(OUT_DIR / "panel3A_fm_mcc_drop_instability_summary.csv", index=False)

        display(panel3A_summary)
        panel_status.append({'cell': 8, 'status': 'completed'})
    else:
        panel_status.append({'cell': 8, 'status': 'skipped', 'reason': 'Need at least two ABMIL runs per foundation model for instability.'})

    # Adapted upstream notebook cell 9.
    if bool(FM_ORDER):
        # Panel 3B - FM-level shift metrics vs MCC-drop instability
        # One separate figure per shift metric.
        # One point per FM.
        # X = mean shift metric across repeats.
        # Y = mean MCC-drop instability across repeats.
        # Error bars = SEM across repeats.






        missing = [p for p in [ABMIL_LAYER_LONG_CSV, ABMIL_TOPK_LONG_CSV, FM_SHIFT_LONG_CSV] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

        abmil_long = pd.concat(
            [
                pd.read_csv(ABMIL_LAYER_LONG_CSV),
                pd.read_csv(ABMIL_TOPK_LONG_CSV),
            ],
            ignore_index=True,
        )
        fm_shift_long = pd.read_csv(FM_SHIFT_LONG_CSV)


        # -----------------------------
        # Repetition-matched MCC-drop instability per FM
        # -----------------------------
        abmil_long["fm"] = abmil_long["feature_subdir"].map(fm_from_feature_subdir)
        abmil_long = abmil_long[abmil_long["fm"].isin(FM_ORDER)].copy()

        required_abmil_cols = {"run_name", "fm", "feature_subdir", "layer_name", "repetition", "mcc_drop"}
        missing_abmil = required_abmil_cols - set(abmil_long.columns)
        if missing_abmil:
            raise ValueError(f"Missing ABMIL columns: {sorted(missing_abmil)}")

        # mcc_drop is duplicated across layer rows, so select one layer to avoid duplicate runs.
        mcc_source_layer = "penultimate" if "penultimate" in set(abmil_long["layer_name"]) else abmil_long["layer_name"].iloc[0]

        run_rep_mcc = (
            abmil_long[abmil_long["layer_name"] == mcc_source_layer]
            [["run_name", "fm", "repetition", "mcc_drop"]]
            .drop_duplicates(["run_name", "repetition"])
            .copy()
        )

        fm_rep_instability = (
            run_rep_mcc
            .groupby(["fm", "repetition"], observed=True)
            .agg(
                n_runs=("run_name", "nunique"),
                mcc_drop_instability=("mcc_drop", "std"),
            )
            .reset_index()
        )

        fm_instability_summary = (
            fm_rep_instability
            .groupby("fm", observed=True)
            .agg(
                instability_mean=("mcc_drop_instability", "mean"),
                instability_sem=("mcc_drop_instability", lambda x: mean_sem(x)[1]),
                instability_std=("mcc_drop_instability", "std"),
                instability_median=("mcc_drop_instability", "median"),
                instability_q25=("mcc_drop_instability", lambda x: float(np.nanquantile(x, 0.25))),
                instability_q75=("mcc_drop_instability", lambda x: float(np.nanquantile(x, 0.75))),
                n_repeats=("repetition", "nunique"),
                n_runs_median=("n_runs", "median"),
            )
            .reindex(FM_ORDER)
            .reset_index()
        )

        # -----------------------------
        # Raw FM shift metric summaries
        # -----------------------------
        fm_shift_long["fm"] = fm_shift_long["fm_name"].map(fm_from_fm_name)
        fm_shift_long = fm_shift_long[fm_shift_long["fm"].isin(FM_ORDER)].copy()

        required_fm_cols = {
            "fm",
            "repetition",
            "mmd_raw",
            "fid_raw",
            "coral_raw",
            "energy_raw",
            "knn_domain_mixing",
        }
        missing_fm = required_fm_cols - set(fm_shift_long.columns)
        if missing_fm:
            raise ValueError(f"Missing FM-shift columns: {sorted(missing_fm)}")

        fm_shift_long["one_minus_knn_mixing"] = 1.0 - fm_shift_long["knn_domain_mixing"]

        shift_defs = [
            ("mmd_raw", "MMD"),
            ("fid_raw", "FID"),
            ("coral_raw", "CORAL"),
            ("energy_raw", "Energy"),
            ("one_minus_knn_mixing", "1 - kNN mixing"),
        ]

        shift_summary_parts = []
        for col, label in shift_defs:
            part = (
                fm_shift_long
                .groupby("fm", observed=True)
                .agg(
                    shift_mean=(col, "mean"),
                    shift_sem=(col, lambda x: mean_sem(x)[1]),
                    shift_std=(col, "std"),
                    shift_median=(col, "median"),
                    shift_q25=(col, lambda x: float(np.nanquantile(x, 0.25))),
                    shift_q75=(col, lambda x: float(np.nanquantile(x, 0.75))),
                    n_repeats=("repetition", "nunique"),
                )
                .reindex(FM_ORDER)
                .reset_index()
            )
            part["shift_col"] = col
            part["shift_measure"] = label
            shift_summary_parts.append(part)

        shift_summary = pd.concat(shift_summary_parts, ignore_index=True)

        plot_df = shift_summary.merge(
            fm_instability_summary,
            on="fm",
            how="inner",
        )

        # -----------------------------
        # Correlations and separate figures
        # -----------------------------
        corr_rows = []

        for shift_col, shift_label in shift_defs:
            sub = plot_df[plot_df["shift_measure"] == shift_label].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["shift_mean", "instability_mean"]
            )

            if len(sub) >= 3 and sub["shift_mean"].nunique() > 1 and sub["instability_mean"].nunique() > 1:
                rho, pval = spearmanr(sub["shift_mean"], sub["instability_mean"])
            else:
                rho, pval = np.nan, np.nan

            corr_rows.append({
                "shift_measure": shift_label,
                "n_fm": len(sub),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(pval) if pd.notna(pval) else np.nan,
            })

            fig, ax = plt.subplots(figsize=(4.8, 4.2))

            for _, row in sub.iterrows():
                fm = row["fm"]
                ax.errorbar(
                    row["shift_mean"],
                    row["instability_mean"],
                    xerr=np.nan_to_num(row["shift_sem"]),
                    yerr=np.nan_to_num(row["instability_sem"]),
                    fmt="o",
                    markersize=8,
                    color=FM_COLORS[fm],
                    ecolor=FM_COLORS[fm],
                    elinewidth=1.1,
                    capsize=2.5,
                    alpha=0.92,
                )

            ax.text(
                0.04,
                0.96,
                "" if pd.isna(rho) else f"rho={rho:.2f}\np={pval:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

            ax.set_xlabel(shift_label)
            ax.set_ylabel("MCC-drop instability")

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=FM_COLORS[fm],
                    markeredgecolor=FM_COLORS[fm],
                    markersize=7,
                    label=fm,
                )
                for fm in FM_ORDER
            ]

            ax.legend(
                handles=handles,
                labels=FM_ORDER,
                frameon=False,
                title="",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )

            sns.despine(fig=fig)

            safe_name = (
                shift_label
                .lower()
                .replace(" ", "_")
                .replace("-", "minus")
                .replace("knn", "knn")
            )

            fig.savefig(
                OUT_DIR / f"panel3B_fm_shift_vs_instability_{safe_name}.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                OUT_DIR / f"panel3B_fm_shift_vs_instability_{safe_name}.png",
                bbox_inches="tight",
            )
            plt.close("all")

        panel3B_corr_df = pd.DataFrame(corr_rows)
        panel3B_corr_df.to_csv(OUT_DIR / "panel3B_fm_shift_metrics_vs_instability_correlations.csv", index=False)
        plot_df.to_csv(OUT_DIR / "panel3B_fm_shift_metrics_vs_instability_plot_data.csv", index=False)
        fm_instability_summary.to_csv(OUT_DIR / "panel3B_fm_instability_summary.csv", index=False)
        shift_summary.to_csv(OUT_DIR / "panel3B_fm_shift_summary.csv", index=False)

        display(panel3B_corr_df)
        display(plot_df)
        panel_status.append({'cell': 9, 'status': 'completed'})
    else:
        panel_status.append({'cell': 9, 'status': 'skipped', 'reason': 'Need at least two ABMIL runs per foundation model for instability.'})

    FM_ORDER = performance_fm_order
    # Adapted upstream notebook cell 10.
    if include_illustration:

        rng = np.random.default_rng(1234)

        # Generate two partially overlapping representation distributions
        development = rng.multivariate_normal(
            mean=[-0.45, 0.10],
            cov=[[0.55, 0.12], [0.12, 0.40]],
            size=120,
        )

        local = rng.multivariate_normal(
            mean=[0.55, -0.05],
            cov=[[0.55, -0.08], [-0.08, 0.40]],
            size=120,
        )

        fig, ax = plt.subplots(figsize=(5, 4.5))

        ax.scatter(
            development[:, 0],
            development[:, 1],
            s=42,
            color="#2C7FB8",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label="Development cohort",
        )

        ax.scatter(
            local[:, 0],
            local[:, 1],
            s=42,
            color="#F28E2B",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label="Local cohort",
        )


        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()

        fig.savefig(
            OUT_DIR / "illustration_representation_shift.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            OUT_DIR / "illustration_representation_shift.svg",
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close("all")
        panel_status.append({'cell': 10, 'status': 'completed'})
    else:
        panel_status.append({'cell': 10, 'status': 'skipped', 'reason': 'Optional synthetic illustration disabled.'})

    return panel_status
