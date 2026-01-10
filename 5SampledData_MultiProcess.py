

# ====================
# 0) 核心工具函数
# ======================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def _infer_columns(df: pd.DataFrame, user_col: str | None, time_col: str | None):
    if user_col is None:
        for c in ["user_id", "userID", "uid", "UserId", "user"]:
            if c in df.columns:
                user_col = c
                break
    if time_col is None:
        for c in ["timestamp", "time", "datetime", "date_time", "t"]:
            if c in df.columns:
                time_col = c
                break
    if user_col is None or time_col is None:
        raise ValueError(f"无法自动识别 user/time 列，请显式传入。当前列名：{list(df.columns)}")
    return user_col, time_col


def _ensure_sorted(df: pd.DataFrame, user_col: str, time_col: str) -> pd.DataFrame:
    return df.sort_values([user_col, time_col]).reset_index(drop=True)


def _save_df(df: pd.DataFrame, save_dir: str, filename: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / filename
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


def _sample_users_with_min_stays(counts: pd.Series,
                                 U: int,
                                 min_stays: int,
                                 rng: np.random.Generator) -> np.ndarray:
    eligible = counts[counts >= min_stays].index.to_numpy()
    if len(eligible) < U:
        raise ValueError(f"满足 min_stays={min_stays} 的用户只有 {len(eligible)} 个，不足 U={U}")
    return rng.choice(eligible, size=U, replace=False)


# ---------- 多进程 worker（只做索引切片，返回 take_idx） ----------
def _worker_take_idx(args):
    uid, L, idx, slice_mode, seed = args
    rng = np.random.default_rng(seed)

    n = len(idx)
    if n < L:
        # 防御性：理论上不会出现（前面已过滤）
        return np.empty(0, dtype=idx.dtype)

    if slice_mode == "prefix":
        return idx[:L]

    if slice_mode == "random_contiguous":
        if n == L:
            return idx
        start = rng.integers(0, n - L + 1)
        return idx[start:start + L]

    raise ValueError("slice_mode must be 'random_contiguous' or 'prefix'")


def _parallel_collect_indices(sampled_users: np.ndarray,
                              lengths: np.ndarray,
                              user2idx: dict,
                              slice_mode: str,
                              base_seed: int,
                              n_jobs: int | None):
    """
    并行返回所有用户的 take_idx（索引），确保随机性可复现：
    seed = base_seed + stable_hash(uid) 之类的方式
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)

    # 为了可复现：每个 uid 独立 seed（不依赖进程执行顺序）
    # 注意：Python 的 hash 默认会随机化，为稳定起见用 uid 转字符串再做简单哈希
    def stable_uid_seed(u):
        s = str(u).encode("utf-8")
        h = 0
        for b in s:
            h = (h * 131 + b) % (2**32)
        return (base_seed + h) % (2**32)

    tasks = []
    for uid, L in zip(sampled_users, lengths):
        idx = user2idx[uid]
        seed = stable_uid_seed(uid)
        tasks.append((uid, int(L), idx, slice_mode, seed))

    # n_jobs=1 时不走多进程（方便调试/小数据）
    if n_jobs == 1:
        out = [_worker_take_idx(t) for t in tasks]
        return out

    # Linux 推荐 fork；Windows 会 spawn（开销大）
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        out = list(ex.map(_worker_take_idx, tasks, chunksize=64))
    return out


# ====================
# 1) 1) 固定总样本数（S1，N_total=1M）并行版
# ======================

def sample_fixed_total_stays_mp(
    csv_path: str,
    U: int,
    N_total: int = 1_000_000,
    min_stays: int = 65,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",  # or "prefix"
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    # 关键优化：一次性建立 user->indices（避免重复扫描）
    user2idx = df.groupby(user_col).indices  # dict: uid -> np.ndarray(row indices)
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})

    # 分配每用户长度，总和==N_total
    base = N_total // U
    rem = N_total - base * U
    lengths = np.full(U, base, dtype=int)
    if rem > 0:
        lengths[:rem] += 1
    rng.shuffle(lengths)  # 避免长度和用户一一对应产生偏差

    per_user_min_needed = max(min_stays, base + (1 if rem > 0 else 0))
    sampled_users = _sample_users_with_min_stays(counts, U, per_user_min_needed, rng)

    # 多核并行：为每个用户计算 take_idx
    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    all_idx = np.concatenate(idx_list)
    out = df.loc[all_idx].copy()
    out = _ensure_sorted(out, user_col, time_col)

    # 防御性裁剪
    if len(out) != N_total:
        out = out.iloc[:N_total].copy()
        out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S1_fixedTotal_N{N_total}_U{U}_min{min_stays}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out



# ====================
# 2) 2) 固定每用户长度（S2，L=500）并行版
# ======================

def sample_fixed_per_user_stays_mp(
    csv_path: str,
    U: int,
    L: int = 500,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    user2idx = df.groupby(user_col).indices
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})
    sampled_users = _sample_users_with_min_stays(counts, U, L, rng)

    lengths = np.full(U, L, dtype=int)

    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    out = df.loc[np.concatenate(idx_list)].copy()
    out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S2_fixedPerUser_U{U}_L{L}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out



# ====================
# 3) 固定用户数（S3，U=2000，变 L）并行版
# ======================

def sample_fixed_user_count_mp(
    csv_path: str,
    U: int = 2000,
    L: int = 500,
    min_stays: int | None = None,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    user2idx = df.groupby(user_col).indices
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})

    need = max(L, (min_stays if min_stays is not None else 0))
    sampled_users = _sample_users_with_min_stays(counts, U, need, rng)
    lengths = np.full(U, L, dtype=int)

    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    out = df.loc[np.concatenate(idx_list)].copy()
    out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S3_fixedUsers_U{U}_L{L}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out




# ====================
# 三组处理同时运行（多核）
# ======================

def run_all_sampling_grids_mp(
    csv_path: str,
    save_dir: str,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    n_jobs: int | None = None,
):
    U_list_1 = [1000, 2000, 4000, 8000, 9000]
    # 满足 min_stays=102 的用户只有 9468 个，不足 U=9900 .

    # S1: fixed total N_total=1M
    for U in U_list_1:
        sample_fixed_total_stays_mp(
            csv_path=csv_path, U=U, N_total=1_000_000, min_stays=65,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )

    U_list_2 = [500, 1000, 2000, 4000, 6000]
    # 满足 min_stays=500 的用户只有 6358 个，不足 U=8000 .
    # S2: fixed per-user L=500
    for U in U_list_2:
        sample_fixed_per_user_stays_mp(
            csv_path=csv_path, U=U, L=500,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )

    # S3: fixed users U=2000, vary L
    for L in [100, 200, 500, 800, 1000]:
        sample_fixed_user_count_mp(
            csv_path=csv_path, U=2000, L=L,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )


if __name__ == "__main__":
    csv_path = "./Data/MoreUser/all.csv"
    run_all_sampling_grids_mp(
        csv_path=csv_path,
        save_dir="./Data/MoreUser/Sampled/",
        random_state=42,
        slice_mode="random_contiguous",
        n_jobs=None,  # None=自动(用cpu_count-1)，也可以手动写 8/16/32
    )
