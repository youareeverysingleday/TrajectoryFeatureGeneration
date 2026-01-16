# 统计用户时间间隔的分布

"""
GeoLife stay-gap analysis (stime/etime)

What this script does:
1) For ALL users, compute inter-stay time gaps:
      gap_i = stime_{i+1} - etime_i
   (i.e., time between the end of current stay and start of next stay)
2) Provide BOTH:
   - raw gaps (minutes)
   - log gaps: log(gap_minutes + eps)
3) Plot histograms and fitted PDFs, and save figures
4) MLE fit distributions and print parameters:
   - Normal on log-gaps
   - Student-t on log-gaps
   - LogNormal on raw gaps (optional but common)
5) QQ-plots:
   - log-gaps vs Normal
   - log-gaps vs Student-t
   - raw-gaps vs LogNormal (optional)

Usage:
  python geolife_gap_stats.py --csv ./GeoLife_all.csv --out_dir ./gap_figs

Assumptions:
  CSV has columns: userID, stime, etime
  stime/etime parseable by pandas.to_datetime
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- SciPy is strongly recommended for MLE & QQ plot ----
try:
    from scipy import stats
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
    stats = None


def compute_gaps_minutes(df: pd.DataFrame,
                         user_col="userID",
                         stime_col="stime",
                         etime_col="etime") -> pd.Series:
    """Compute inter-stay gap: next_stime - current_etime (minutes), per user."""
    df = df[[user_col, stime_col, etime_col]].copy()
    df[stime_col] = pd.to_datetime(df[stime_col], errors="coerce")
    df[etime_col] = pd.to_datetime(df[etime_col], errors="coerce")
    df = df.dropna(subset=[user_col, stime_col, etime_col])
    df[user_col] = df[user_col].astype(int)

    # sort within user by start time
    df = df.sort_values([user_col, stime_col]).reset_index(drop=True)

    # shift next start time within each user
    df["next_stime"] = df.groupby(user_col, sort=False)[stime_col].shift(-1)

    # gap = next start - current end
    gap = (df["next_stime"] - df[etime_col]).dt.total_seconds() / 60.0  # minutes

    # keep finite
    gap = gap.replace([np.inf, -np.inf], np.nan).dropna()

    return gap


def safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """log(x + eps) with x>=0 enforced."""
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    return np.log(x + eps)


def fit_distributions(raw_gaps_min: np.ndarray,
                      log_gaps: np.ndarray):
    """
    Return fitted params dict.
    - Normal on log_gaps: (mu, sigma)
    - Student-t on log_gaps: (df, loc, scale)
    - LogNormal on raw_gaps_min: (shape, loc, scale)  [optional but useful]
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy is not available.")

    params = {}

    # Normal MLE on log-gaps
    mu, sigma = stats.norm.fit(log_gaps)
    params["normal_loggap"] = {"mu": float(mu), "sigma": float(sigma)}

    # Student-t MLE on log-gaps
    df_t, loc_t, scale_t = stats.t.fit(log_gaps)
    params["studentt_loggap"] = {"nu": float(df_t), "loc": float(loc_t), "scale": float(scale_t)}

    # --- LogNormal MLE on raw gaps (MUST be strictly > 0) ---
    raw_pos = raw_gaps_min[np.isfinite(raw_gaps_min) & (raw_gaps_min > 0)]

    if raw_pos.size < 50:
        # 数据太少就别拟合，避免误导
        params["lognorm_rawgap"] = None
        print("[Warn] Too few positive gaps for LogNormal fit. Skip lognorm.")
    else:
        # 常用做法：固定 loc=0
        s, loc_lg, scale_lg = stats.lognorm.fit(raw_pos, floc=0)
        params["lognorm_rawgap"] = {"shape": float(s), "loc": float(loc_lg), "scale": float(scale_lg)}

    return params


def plot_hist_with_fits(raw_gaps_min: np.ndarray,
                        log_gaps: np.ndarray,
                        params: dict,
                        out_dir: str,
                        bins_raw: int = 200,
                        bins_log: int = 200):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Plot 1: raw gaps histogram (minutes) + LogNormal fit ----------
    plt.figure(figsize=(7.2, 4.2))
    # Use log-scale x for visibility (optional); here keep linear histogram but you can switch
    plt.hist(raw_gaps_min, bins=bins_raw, density=True)
    plt.xlabel("Inter-stay gap (minutes)")
    plt.ylabel("Density")
    plt.title("Raw inter-stay gaps (minutes)")

    if SCIPY_OK and params is not None and "lognorm_rawgap" in params:
        s = params["lognorm_rawgap"]["shape"]
        loc = params["lognorm_rawgap"]["loc"]
        scale = params["lognorm_rawgap"]["scale"]
        # x grid (avoid extreme tail dominating)
        x_max = np.percentile(raw_gaps_min, 99.5)
        xs = np.linspace(0, max(1e-6, x_max), 800)
        pdf = stats.lognorm.pdf(xs, s=s, loc=loc, scale=scale)
        plt.plot(xs, pdf)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_rawgap_with_lognorm.pdf"), dpi=300)
    plt.close()

    # ---------- Plot 2: log gaps histogram + Normal & Student-t fits ----------
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(log_gaps, bins=bins_log, density=True)
    plt.xlabel("log(gap_minutes + eps)")
    plt.ylabel("Density")
    plt.title("Log inter-stay gaps")

    if SCIPY_OK and params is not None:
        # x grid
        x_min, x_max = np.percentile(log_gaps, [0.5, 99.5])
        xs = np.linspace(x_min, x_max, 800)

        # Normal
        mu = params["normal_loggap"]["mu"]
        sigma = params["normal_loggap"]["sigma"]
        plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma))

        # Student-t
        nu = params["studentt_loggap"]["nu"]
        loc = params["studentt_loggap"]["loc"]
        scale = params["studentt_loggap"]["scale"]
        plt.plot(xs, stats.t.pdf(xs, df=nu, loc=loc, scale=scale))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_loggap_with_normal_studentt.pdf"), dpi=300)
    plt.close()


def qq_plots(raw_gaps_min: np.ndarray,
             log_gaps: np.ndarray,
             params: dict,
             out_dir: str):
    if not SCIPY_OK:
        raise RuntimeError("SciPy is not available. Please install scipy to use QQ plots.")
    os.makedirs(out_dir, exist_ok=True)

    # -------- QQ 1: log_gaps vs Normal --------
    plt.figure(figsize=(5.2, 5.2))
    mu = params["normal_loggap"]["mu"]
    sigma = params["normal_loggap"]["sigma"]
    stats.probplot(log_gaps, dist=stats.norm, sparams=(mu, sigma), plot=plt)
    plt.title("QQ-Plot: log gaps vs Normal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qq_loggap_normal.pdf"), dpi=300)
    plt.close()

    # -------- QQ 2: log_gaps vs Student-t --------
    # SciPy probplot supports dist=stats.t with sparams=(df, loc, scale)
    plt.figure(figsize=(5.2, 5.2))
    nu = params["studentt_loggap"]["nu"]
    loc = params["studentt_loggap"]["loc"]
    scale = params["studentt_loggap"]["scale"]
    stats.probplot(log_gaps, dist=stats.t, sparams=(nu, loc, scale), plot=plt)
    plt.title("QQ-Plot: log gaps vs Student-t")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qq_loggap_studentt.pdf"), dpi=300)
    plt.close()

    # -------- QQ 3 (optional): raw gaps vs LogNormal --------
    # For lognormal, probplot wants dist=stats.lognorm, sparams=(shape, loc, scale)
    plt.figure(figsize=(5.2, 5.2))
    s = params["lognorm_rawgap"]["shape"]
    loc = params["lognorm_rawgap"]["loc"]
    scale = params["lognorm_rawgap"]["scale"]
    # If many zeros, probplot may look weird; consider using raw_gaps_min[raw_gaps_min>0]
    raw_pos = raw_gaps_min[raw_gaps_min > 0]
    if raw_pos.size >= 50:
        stats.probplot(raw_pos, dist=stats.lognorm, sparams=(s, loc, scale), plot=plt)
        plt.title("QQ-Plot: raw gaps (>0) vs LogNormal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qq_rawgap_lognorm.pdf"), dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to GeoLife stays CSV containing userID, stime, etime")
    ap.add_argument("--out_dir", type=str, default="./gap_figs", help="Directory to save plots")
    ap.add_argument("--user_col", type=str, default="userID")
    ap.add_argument("--stime_col", type=str, default="stime")
    ap.add_argument("--etime_col", type=str, default="etime")
    ap.add_argument("--eps", type=float, default=1e-6, help="epsilon for log(gap+eps)")
    ap.add_argument("--min_gap_min", type=float, default=0.0, help="Filter: keep gaps >= this minutes (e.g., 0)")
    ap.add_argument("--max_gap_pct", type=float, default=99.9, help="Optional tail clip percentile (e.g., 99.9). Use 100 for no clip.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    gaps_min = compute_gaps_minutes(df, user_col=args.user_col, stime_col=args.stime_col, etime_col=args.etime_col)

    # Filter negatives (overlaps) & extremely tiny
    gaps_min = gaps_min[gaps_min >= args.min_gap_min]

    # Optional tail clipping for plotting/fit stability (you can set max_gap_pct=100 to disable)
    if args.max_gap_pct < 100.0:
        cap = np.percentile(gaps_min.values, args.max_gap_pct)
        gaps_min = gaps_min[gaps_min <= cap]

    raw = gaps_min.values.astype(np.float64)
    print(f"[Gap sign] <=0 ratio: {(raw<=0).mean():.4f}  "
        f"(zero: {(raw==0).mean():.4f}, neg: {(raw<0).mean():.4f})")
    logg = safe_log(raw, eps=args.eps)

    print(f"[Stats] #gaps = {raw.size}")
    print(f"[Stats] raw gaps minutes: mean={raw.mean():.3f}, median={np.median(raw):.3f}, p95={np.percentile(raw,95):.3f}, p99={np.percentile(raw,99):.3f}")
    print(f"[Stats] log gaps: mean={logg.mean():.3f}, std={logg.std():.3f}")

    if not SCIPY_OK:
        print("\n[Warn] SciPy not available. Install with: pip install scipy\n"
            "Will only plot histograms without MLE fits/QQ plots.")
        params = None
        os.makedirs(args.out_dir, exist_ok=True)
        plt.figure(figsize=(7.2, 4.2))
        plt.hist(raw, bins=200, density=True)
        plt.xlabel("Inter-stay gap (minutes)"); plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_rawgap.pdf"), dpi=300)
        plt.close()

        plt.figure(figsize=(7.2, 4.2))
        plt.hist(logg, bins=200, density=True)
        plt.xlabel("log(gap_minutes + eps)"); plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_loggap.pdf"), dpi=300)
        plt.close()
        return

    # MLE fits
    params = fit_distributions(raw_gaps_min=raw, log_gaps=logg)
    if params is not None and params.get("lognorm_rawgap") is not None:
        print("\n[MLE Parameters]")
        print("Normal on log-gaps:")
        print(f"  mu={params['normal_loggap']['mu']:.6f}, sigma={params['normal_loggap']['sigma']:.6f}")

        print("Student-t on log-gaps:")
        print(f"  nu(df)={params['studentt_loggap']['nu']:.6f}, loc={params['studentt_loggap']['loc']:.6f}, scale={params['studentt_loggap']['scale']:.6f}")

        print("LogNormal on raw gaps (minutes):")
        print(f"  shape(s)={params['lognorm_rawgap']['shape']:.6f}, loc={params['lognorm_rawgap']['loc']:.6f}, scale={params['lognorm_rawgap']['scale']:.6f}")

        # Plots
        plot_hist_with_fits(raw_gaps_min=raw, log_gaps=logg, params=params, out_dir=args.out_dir)
        qq_plots(raw_gaps_min=raw, log_gaps=logg, params=params, out_dir=args.out_dir)

        print(f"\n[Done] Figures saved to: {os.path.abspath(args.out_dir)}")
        print("  - hist_rawgap_with_lognorm.pdf")
        print("  - hist_loggap_with_normal_studentt.pdf")
        print("  - qq_loggap_normal.pdf")
        print("  - qq_loggap_studentt.pdf")
        print("  - qq_rawgap_lognorm.pdf (if enough positive gaps)")


if __name__ == "__main__":
    main()



# python 6DistributionofUserTimeIntervals.py --csv ./Data/Output/all_users_context_combined.csv --out_dir ./Pictures/GeoLife/
# python 6DistributionofUserTimeIntervals.py --csv ./Data/Output/all_users_context_combined_merged30min.csv --out_dir ./Pictures/GeoLife/
# python 6DistributionofUserTimeIntervals.py --csv ./Data/Output/all_users_context_combined_gapLE0_merged.csv --out_dir ./Pictures/GeoLife/

# python 6DistributionofUserTimeIntervals.py --csv ./Data/MoreUser/all.csv --out_dir ./Pictures/MoreUser/
# python 6DistributionofUserTimeIntervals.py --csv ./Data/MoreUser/all_merged30min.csv --out_dir ./Pictures/MoreUser/
# python 6DistributionofUserTimeIntervals.py --csv ./Data/MoreUser/all_gapLE0_merged.csv --out_dir ./Pictures/MoreUser/
