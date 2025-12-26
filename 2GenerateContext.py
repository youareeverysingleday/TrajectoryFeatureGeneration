import pandas as pd
from datetime import timedelta
import random
import os
import numpy as np
import multiprocessing

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
def generate_context_for_df(df_input: pd.DataFrame, N_top_frequent: int = 3) -> pd.DataFrame:
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
    print(top_n_grids)
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

    return df


# --- 主控制函数：处理指定文件夹下的所有数据 ---
def process_data_directory(
    data_dir: str, 
    individual_output_dir: str,
    combined_output_filepath: str,
    N_top_frequent: int = 3
):
    """
    处理指定输入文件夹下的所有用户停留点数据，并将结果保存到指定的输出路径。

    参数:
    data_dir (str): 原始用户停留点数据文件 (.csv) 所在的文件夹路径。
    individual_output_dir (str): 用于保存每个用户单独上下文文件的文件夹路径。
    combined_output_filepath (str): 用于保存所有用户合并数据的完整文件路径（包括文件名）。
    N_top_frequent (int): 排除的周期停留点数量。
    """
    
    if not os.path.exists(data_dir):
        print(f"错误：未找到指定的输入文件夹路径 -> {data_dir}")
        return
        
    # 确保单个用户输出文件夹存在
    if not os.path.exists(individual_output_dir):
        os.makedirs(individual_output_dir)
        print(f"创建输出文件夹: {individual_output_dir}")

    all_users_data = []
    
    # 遍历输入文件夹下的所有文件
    for filename in os.listdir(data_dir):
        # 跳过已经生成的上下文文件和合并文件
        # if filename.startswith(OUTPUT_PREFIX) or filename == os.path.basename(combined_output_filepath):
        #     continue
            
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            
            print(f"\n--- 正在处理文件: {filename} ---")
            try:
                # 1. 读取数据
                df_raw = pd.read_csv(filepath, index_col=0) 
                
                if df_raw.empty or len(df_raw) < 2:
                    print(f"Skipping: 数据为空或行数不足。")
                    continue
                
                # 2. 调用核心函数处理数据
                df_processed = generate_context_for_df(df_raw, N_top_frequent)
                
                # 确保 userID 列存在
                if 'userID' not in df_processed.columns:
                    print("警告: 缺少 'userID' 列。尝试从文件名推断 UserID。")
                    user_id = filename.split('_')[0] 
                    df_processed.insert(0, 'userID', user_id)
                
                # 形式一：每个用户单独保存（保存到指定 Individual 路径）
                # 保存后包含两列上下文：context_fuzzy 和 context_precise
                output_filename_single = f"{filename}"
                output_filepath_single = os.path.join(individual_output_dir, output_filename_single)
                df_processed.to_csv(output_filepath_single, index=True)
                print(f" 保存单个用户数据到: {output_filepath_single}")
                print(f" 包含上下文列: context_fuzzy 和 context_precise")
                
                all_users_data.append(df_processed)

            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")
                continue

    # 形式二：所有用户数据合并保存（保存到指定 Combined 路径）
    if all_users_data:
        df_all = pd.concat(all_users_data)
        df_all.to_csv(combined_output_filepath, index=True)
        print(f"\n 成功合并所有用户数据并保存到: {combined_output_filepath}")
    else:
        print("\n 没有新的 CSV 文件被成功处理和合并。")

import time
gUserTrajPath = './Data/MoreUser/Input'

# --- 执行示例 (需要替换路径) ---
if __name__ == "__main__":
    # # --- 示例路径配置 ---
    # INPUT_FOLDER = "./Data/Output/Stays" 
    # INDIVIDUAL_OUTPUT_FOLDER = "./Data/Output/Context"
    # COMBINED_OUTPUT_FILE = "./Data/Output/all_users_context_combined.csv" 
    
    # process_data_directory(
    #     data_dir=INPUT_FOLDER,
    #     individual_output_dir=INDIVIDUAL_OUTPUT_FOLDER,
    #     combined_output_filepath=COMBINED_OUTPUT_FILE
    # )

    start_time = time.time()

    # 获取所有轨迹用户ID。
    userList = next(os.walk(gUserTrajPath))[2]
    userList = [x.split('.')[0] for x in userList]
    users_pd = pd.DataFrame(userList, columns=['ID'])
    
    # 分为测试机和训练集。
    train_users = users_pd.sample(frac=0.8, random_state=42)
    test_users = users_pd.drop(train_users.index)

    # 将资源管理放循环外
    ProcessManager = multiprocessing.Manager()
    Lock = ProcessManager.Lock()
    ShareData = ProcessManager.Namespace()
    ShareData.dat = None

    trainOutputStayPath = "./Data/MoreUser/Output/Train/"
    testOutputStayPath = "./Data/MoreUser/Output/Test/"

    trainOutputAllStayPath = "./Data/MoreUser/Output/Train.csv"
    testOutputAllStayPath = "./Data/MoreUser/Output/Test.csv"

    for dataset_type, users, stay_path, all_stay_path in [
        ("train", train_users, trainOutputStayPath, trainOutputAllStayPath),
        ("test", test_users, testOutputStayPath, testOutputAllStayPath)]:
        for user in users:
            p = multiprocessing.Process(target=generate_context_for_df, 
                                        args=(user, dataset_type, stay_path, all_stay_path, Lock, ShareData))
            p.start()
            p.join()