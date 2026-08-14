import numpy as np
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
import logging
import os
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_values_from_nested(data):
    """
    递归函数，用于从嵌套的列表和字典中提取所有值，同时保留列表的维度。

    参数:
        data (list or dict): 嵌套的列表或字典结构。

    返回:
        list: 包含所有值的嵌套列表，保持原始维度。
    """
    if isinstance(data, dict):  # 如果当前数据是字典
        # 提取字典中的所有值，返回一个列表
        return [extract_values_from_nested(v) for v in data.values()]
    elif isinstance(data, list):  # 如果当前数据是列表
        # 递归处理列表中的每个元素，保持列表结构
        return [extract_values_from_nested(item) for item in data]
    else:
        # 如果是其他类型，直接返回该值
        return data


def generate_layered_model(layer_params: List[Dict[str, Tuple[float, float]]],
                          require_increasing: List[str]) -> Optional[List[Dict[str, float]]]:
    """
    生成满足约束的分层模型

    参数:
        layer_params: 每层的参数范围，例如 [{'vs': (100, 500), ...}, ...]
        require_increasing: 需要检查的参数名称列表（不允许连续3层递减）

    返回:
        生成的有效模型参数列表，如果失败则返回 None
    """
    model = []
    previous_values = {param: [None, None] for param in require_increasing}  # 记录前两层的值

    for layer in layer_params:
        current_layer = {}
        for param, (min_val, max_val) in layer.items():
            current_layer[param] = np.random.uniform(min_val, max_val)

            # 检查连续3层递减
            if param in require_increasing:
                # 获取前两层的值
                prev1, prev2 = previous_values[param]

                # 如果前两层的值都存在，检查是否连续3层递减
                if prev1 is not None and prev2 is not None:
                    if current_layer[param] < prev1 < prev2:
                        return None  # 连续3层递减，返回 None

                # 更新前两层的值
                previous_values[param] = [current_layer[param], prev1]

        # 将当前层添加到模型中
        model.append(current_layer)

    return model


def generate_increasing_model(layer_params: List[Dict[str, Tuple[float, float]]],
                             require_increasing: List[str]) -> Optional[List[Dict[str, float]]]:
    """
    生成满足递增约束的分层模型

    参数:
        layer_params: 每层的参数范围，例如 [{'vs': (100, 500), ...}, ...]
        require_increasing: 需要检查的参数名称列表（必须递增）

    返回:
        生成的有效模型参数列表，如果失败则返回 None
    """
    model = []
    previous_values = {param: None for param in require_increasing}  # 记录前一层的值

    for layer in layer_params:
        current_layer = {}
        for param, (min_val, max_val) in layer.items():
            if param in require_increasing:
                if previous_values[param] is None:
                    current_layer[param] = np.random.uniform(min_val, max_val)
                else:
                    current_layer[param] = np.random.uniform(previous_values[param], max_val)
            else:
                current_layer[param] = np.random.uniform(min_val, max_val)

            # 更新前一层的值
            if param in require_increasing:
                previous_values[param] = current_layer[param]

        # 将当前层添加到模型中
        model.append(current_layer)

    return model


def _monte_carlo_worker(args: tuple) -> Optional[List[Dict[str, float]]]:
    """
    修改后的单线程工作函数（移除多进程相关逻辑）
    """
    layer_params, require_increasing, max_attempts, force_increasing = args

    # 设置随机种子（使用时间戳保证每次不同）
    seed = int(time.time() * 1000) % (2 ** 32)
    np.random.seed(seed)

    for attempt in range(max_attempts):
        if force_increasing:
            model = generate_increasing_model(layer_params, require_increasing)
        else:
            model = generate_layered_model(layer_params, require_increasing)
        if model is not None:
            return model

    logging.warning(f"在 {max_attempts} 次尝试后未能生成有效模型")
    return None


def monte_carlo_sampling(
        layer_params: List[Dict[str, Tuple[float, float]]],
        n_models: int,
        max_attempts: int = 10000,
        require_increasing=None
) -> List[List[Dict[str, float]]]:
    """
    修改后的单线程蒙特卡洛采样
    """
    # 计算需要生成多少递增模型
    if require_increasing is None:
        require_increasing = ['vs']
    n_increasing = int(int(n_models) * 0.05)
    n_normal = n_models - n_increasing

    # 准备参数列表
    worker_args = [(layer_params, require_increasing, max_attempts, True)] * n_increasing + \
                  [(layer_params, require_increasing, max_attempts, False)] * n_normal

    results = []
    for args in worker_args:
        result = _monte_carlo_worker(args)
        if result is not None:
            results.append(result)
            if len(results) >= n_models:
                break  # 达到需求数量后提前终止

    return results[:n_models]
#
#
# def _monte_carlo_worker(args: tuple) -> Optional[List[Dict[str, float]]]:
#     """
#     多进程工作函数，用于生成单个模型
#
#     参数:
#         args: 包含 (layer_params, require_increasing, max_attempts, force_increasing) 的元组
#
#     返回:
#         生成的模型或 None（如果失败）
#     """
#     layer_params, require_increasing, max_attempts, force_increasing = args
#     seed = int(time.time() * os.getpid()) % (2**32)  # 确保种子值在 [0, 2**32 - 1] 范围内
#     np.random.seed(seed)  # 设置随机种子
#
#     for attempt in range(max_attempts):
#         if force_increasing:
#             model = generate_increasing_model(layer_params, require_increasing)
#         else:
#             model = generate_layered_model(layer_params, require_increasing)
#         if model is not None:
#             return model
#
#     logging.warning(f"在 {max_attempts} 次尝试后未能生成有效模型")
#     return None
#
#
# def monte_carlo_sampling(
#         layer_params: List[Dict[str, Tuple[float, float]]],
#         n_models: int,
#         max_attempts: int = 1000,
#         n_workers: int = None,
#         require_increasing: List[str] = ['vs', 'density']
# ) -> List[List[Dict[str, float]]]:
#     """
#     并行蒙特卡洛采样生成多个有效模型
#
#     参数:
#         layer_params: 每层的参数范围
#         n_models: 需要生成的模型数量
#         max_attempts: 每个模型的最大尝试次数
#         n_workers: 使用的进程数
#         require_increasing: 需要递增的参数列表
#
#     返回:
#         生成的有效模型列表
#     """
#     if n_workers is None:
#         n_workers = max(1, cpu_count() - 1)  # 默认使用CPU核心数-1
#
#     # 计算需要生成多少递增模型
#     n_increasing = int(n_models * 0.3)
#     n_normal = n_models - n_increasing
#
#     # 准备参数列表
#     worker_args = [(layer_params, require_increasing, max_attempts, True)] * n_increasing + \
#                   [(layer_params, require_increasing, max_attempts, False)] * n_normal
#
#     with Pool(processes=n_workers) as pool:
#         results = []
#         for result in pool.imap_unordered(_monte_carlo_worker, worker_args):
#             if result is not None:
#                 results.append(result)
#                 if len(results) >= n_models:
#                     break  # 提前终止
#
#         pool.close()  # 关闭进程池
#         pool.join()  # 等待所有进程结束
#
#     return results[:n_models]


if __name__ == '__main__':
    # 示例用法
    layer_definitions = [
        {
            'd': (5, 5),
            'vs': (100, 500),
            'density': (1.8, 2.2)
        },
        {
            'd': (5, 5),
            'vs': (200, 800),
            'density': (2.3, 2.8)
        },
        {
            'd': (5, 5),
            'vs': (200, 800),
            'density': (2.3, 2.8)
        },
        {
            'd': (5, 5),
            'vs': (200, 800),
            'density': (2.3, 2.8)
        },
    ]

    models = monte_carlo_sampling(
        layer_params=layer_definitions,
        n_models=1000,
    )
    init_model = models
    init_model = extract_values_from_nested(models)
    init_model = np.array(extract_values_from_nested(models))
    logging.info(f"成功生成 {len(models)} 个有效模型")
    print("第一个模型示例：")
    for i, layer in enumerate(models[3]):
        print(f"层 {i + 1}: {layer}")