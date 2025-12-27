import pandas as pd
from datetime import timedelta
import random
import os
import numpy as np
import multiprocessing

import time
from tqdm import tqdm

# --- 配置和常量定义 ---
N_TOP_FREQUENT = 3 # 排除的周期停留点数量
TIME_LIMITS = {
    'month': timedelta(days=30),
    'week': timedelta(days=7),
    'day': timedelta(days=1)
}

# --- 辅助函数：模糊时间生成（与之前保持一致） ---
def get_fuzzy_time_expression(time_delta, target_time):
    """根据时间差和目标时间生成模糊时间表达式。"""
    
    if time_delta > TIME_LIMITS['month']:
        months = round(time_delta.days / 30)
        return f"in about {months} months" if months > 1 else "in the next month"
    
    elif time_delta > TIME_LIMITS['week']:
        weeks = round(time_delta.days / 7)
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_time.weekday()]
        return f"in about {weeks} weeks, specifically next {weekday}"
    
    elif time_delta > TIME_LIMITS['day']:
        days = time_delta.days
        return f"in about {days} days, on {target_time.date().strftime('%Y-%m-%d')}"
    
    else: # Time Delta <= 1 day
        hour = target_time.hour
        if 5 <= hour < 12:
            period = "in the morning"
        elif 12 <= hour < 18:
            period = "in the afternoon"
        elif 18 <= hour < 22:
            period = "in the evening"
        else:
            period = "late at night"
        return f"later today, {period}"


# --- 核心函数：生成上下文（同时生成模糊和精确时间上下文） ---
def generate_context_for_df(df_input: pd.DataFrame, N_top_frequent: int,
                            StaySavePath: str) -> None:
    """
    根据给定的详细逻辑，为单个用户的停留点数据生成上下文。
    同时生成两列：context_fuzzy（模糊时间）和 context_precise（精确时间）。
    """
    df = df_input.copy()
    
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])
    df['context_fuzzy'] = None  # 包含模糊时间的上下文
    df['context_precise'] = None  # 包含精确时间的上下文
    
    if len(df) < N_top_frequent or df.empty:
        return df

    # 2. 识别周期停留点并构建 aperiodic_stay_list
    top_n_grids = df['grid'].value_counts().nlargest(N_top_frequent).index.tolist()
    # print(top_n_grids)
    aperiodic_stay_list = df[~df['grid'].isin(top_n_grids)].index.tolist()
    current_stay_candidates = df.index[:-1].tolist()

    while aperiodic_stay_list and current_stay_candidates:
        current_stay_idx = random.choice(current_stay_candidates)
        
        future_aperiodic_stays = [
            target_idx for target_idx in aperiodic_stay_list 
            if target_idx > current_stay_idx
        ]
        
        if not future_aperiodic_stays:
            current_stay_candidates.remove(current_stay_idx)
            continue

        generate_context_stay_idx = random.choice(future_aperiodic_stays)
        
        current_stay = df.loc[current_stay_idx]
        generate_context_stay = df.loc[generate_context_stay_idx]
        
        start_grid = current_stay['grid']
        end_grid = generate_context_stay['grid']
        user_id = current_stay['userID']
        
        time_delta = generate_context_stay['stime'] - current_stay['etime']
        
        # 精确时间（始终生成）
        precise_time_str = generate_context_stay['stime'].strftime('%Y-%m-%d %H:%M:%S')
        context_precise = (
            f"User {user_id} will move from grid {start_grid} to grid {end_grid}, "
            f"at {precise_time_str}."
        )
        
        # 模糊时间
        fuzzy_time_expression = None
        should_use_fuzzy = False

        # --- 时间差逻辑判断 ---
        if time_delta > TIME_LIMITS['month']:
            if random.random() >= 0.1: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        elif time_delta > TIME_LIMITS['week']:
            if random.random() >= 0.3: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        elif time_delta > TIME_LIMITS['day']:
            if random.random() >= 0.5: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True
        else: # Time Delta <= 1 day
            if random.random() >= 0.7: 
                fuzzy_time_expression = get_fuzzy_time_expression(time_delta, generate_context_stay['stime'])
                should_use_fuzzy = True

        # 生成模糊时间上下文
        if should_use_fuzzy and fuzzy_time_expression:
            context_fuzzy = (
                f"User {user_id} will move from grid {start_grid} to grid {end_grid}, "
                f"arriving around {fuzzy_time_expression}."
            )
        else:
            # 如果不使用模糊时间，则模糊上下文也使用精确时间
            context_fuzzy = context_precise

        # 同时保存两种上下文
        df.loc[current_stay_idx, 'context_fuzzy'] = context_fuzzy
        df.loc[current_stay_idx, 'context_precise'] = context_precise
        
        aperiodic_stay_list.remove(generate_context_stay_idx)
        current_stay_candidates.remove(current_stay_idx)

        df.to_csv(StaySavePath+f"{user_id}.csv", index=True)
    
    # 合并单个文件为整体文件。
    # with Lock:
    #     if ShareData.dat is None:
    #         ShareData.dat = df
    #     else:
    #         ShareData.dat = pd.concat([ShareData.dat, df], ignore_index=True)


def get_all_file_paths(directory):
    """_summary_
    提供所有用户的停留点文件路径列表。
    Args:
        directory (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def merge_csvs(folder_path:str, output_path:str) -> None:
    """
    将指定文件夹下的所有 CSV 文件合并为一个 DataFrame 并保存。
    """
    all_files = get_all_file_paths(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    if not csv_files:
        print("没有找到 CSV 文件。")
        return
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"成功合并 {len(csv_files)} 个 CSV 文件并保存到 {output_path}")


# 使用示例
# all_files = get_all_file_paths('./Data/Output/Stays')
# print(all_files)


# --- 执行示例 (需要替换路径) ---
if __name__ == "__main__":


    start_time = time.time()
    # all_files = get_all_file_paths('./Data/Test/Stays/')
    # print(all_files)

    # 将资源管理放循环外
    ProcessManager = multiprocessing.Manager()
    # Lock = ProcessManager.Lock()
    # ShareData = ProcessManager.Namespace()
    # ShareData.dat = None
    ProcessPool = multiprocessing.Pool()

    # gUserTrajPath = './Data/MoreUser/Input'
    # OutputStayPath = "./Data/MoreUser/Output/"
    # OutputAllStayPath = "./Data/MoreUser/Output/all.csv"
    
    gUserTrajPath = './Data/Test/Stays/'
    OutputStayPath = "./Data/Test/Context/"
    OutputAllStayPath = "./Data/Test/all.csv"

    all_files = get_all_file_paths(gUserTrajPath)

    # 高频停留点获取的数量。
    N_top_frequent = 3

    # 每个文件启动一个进程进行处理。
    for singleuserfile in tqdm(all_files):
        singleUserDf = pd.read_csv(singleuserfile, index_col=0)

        ProcessPool.apply_async(generate_context_for_df, 
                                    args=(singleUserDf, N_top_frequent,
                                            OutputStayPath))
    ProcessPool.close()
    ProcessPool.join()
    ProcessManager.shutdown()

    print("所有用户数据处理完毕，开始合并数据...")
    # 合并所有生成的上下文文件为一个 DataFrame 并保存
    merge_csvs(OutputStayPath, OutputAllStayPath)

    print(f"全部数据处理完成，总耗时 {time.time() - start_time:.2f} 秒")