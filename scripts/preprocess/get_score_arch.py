import json
import os

import pandas as pd


def get_enable_layers(scores: pd.DataFrame, budget_portion: float):
    """
    根据零成本代理分数选择要启用LoRA专家的层

    参数:
        scores: DataFrame，包含层名和对应的零成本代理分数
        budget_portion: 浮点数，参数预算比例（0-1之间）

    返回:
        enable_layers: 列表，应该启用LoRA专家的层名列表
    """
    # ========== 步骤1：分离LLM和ViT层分数 ==========
    llm_scores = (
        scores[scores["layer"].str.contains("language_model")]
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
    )
    vit_scores = (
        scores[scores["layer"].str.contains("vision_model")]
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
    )
    
    # 记录各模态的层数
    num_llm_layers = len(llm_scores)
    num_vit_layers = len(vit_scores)
    total_layers = num_llm_layers + num_vit_layers
    
    print(f"\n层数统计: LLM层={num_llm_layers}, ViT层={num_vit_layers}, 总计={total_layers}")

    # ========== 步骤2：计算自适应预算比例 ==========
    total_score = llm_scores["score"].sum() + vit_scores["score"].sum()
    
    if total_score > 0:
        llm_score_ratio = llm_scores["score"].sum() / total_score
        vit_score_ratio = vit_scores["score"].sum() / total_score
    else:
        llm_score_ratio = 0.5
        vit_score_ratio = 0.5

    # ========== 步骤3：计算总预算 ==========
    total_budget = int(total_layers * budget_portion + 0.5)
    total_budget = min(total_budget, total_layers)  # 不超过总层数
    
    print(f"预算比例: {budget_portion:.0%}, 总预算: {total_budget}/{total_layers}")
    
    if total_budget == 0:
        print("警告：预算为0，没有层被启用")
        return []
    
    if total_budget == total_layers:
        print("预算为100%，启用所有层")
        # 直接返回所有层，简化处理
        all_layers = pd.concat([llm_scores, vit_scores])["layer"].tolist()
        # 格式化层名
        if all_layers and "base_model" in all_layers[0]:
            all_layers = [".".join(layer.split(".")[2:]) for layer in all_layers]
        return all_layers

    # ========== 步骤4：动态分配预算 ==========
    # 基于分数比例计算初始预算
    vit_target_budget = int(total_budget * vit_score_ratio + 0.5)
    llm_target_budget = total_budget - vit_target_budget
    
    # 确保每个模态至少分配25%的总预算（但不超过该模态的层数）
    vit_min_budget = min(
        num_vit_layers,
        max(int(total_budget * 0.25), vit_target_budget)
    )
    llm_min_budget = min(
        num_llm_layers,
        max(int(total_budget * 0.25), llm_target_budget)
    )
    
    # 调整预算：确保总和等于total_budget
    # 如果两个模态的最小预算之和超过总预算，按比例缩减
    if vit_min_budget + llm_min_budget > total_budget:
        # 按比例缩减
        scale = total_budget / (vit_min_budget + llm_min_budget)
        vit_min_budget = max(1, int(vit_min_budget * scale))
        llm_min_budget = total_budget - vit_min_budget
    # 如果小于总预算，将剩余预算分配给有更多层数的模态
    elif vit_min_budget + llm_min_budget < total_budget:
        remaining = total_budget - (vit_min_budget + llm_min_budget)
        # 优先分配给还能容纳更多层的模态
        if vit_min_budget < num_vit_layers and llm_min_budget < num_llm_layers:
            # 两个都能加，按分数比例分配剩余
            vit_extra = int(remaining * (vit_score_ratio))
            llm_extra = remaining - vit_extra
            vit_min_budget = min(num_vit_layers, vit_min_budget + vit_extra)
            llm_min_budget = min(num_llm_layers, llm_min_budget + llm_extra)
            # 如果还有剩余，全给还有容量的模态
            if vit_min_budget + llm_min_budget < total_budget:
                if vit_min_budget < num_vit_layers:
                    vit_min_budget = min(num_vit_layers, vit_min_budget + (total_budget - vit_min_budget - llm_min_budget))
                else:
                    llm_min_budget = min(num_llm_layers, llm_min_budget + (total_budget - vit_min_budget - llm_min_budget))
        elif vit_min_budget < num_vit_layers:
            vit_min_budget = min(num_vit_layers, vit_min_budget + remaining)
        else:
            llm_min_budget = min(num_llm_layers, llm_min_budget + remaining)
    
    # 最终确保总和等于total_budget
    if vit_min_budget + llm_min_budget != total_budget:
        # 调整差值
        diff = total_budget - (vit_min_budget + llm_min_budget)
        if diff > 0 and vit_min_budget < num_vit_layers:
            vit_min_budget = min(num_vit_layers, vit_min_budget + diff)
        elif diff > 0:
            llm_min_budget = min(num_llm_layers, llm_min_budget + diff)
    
    print(f"\n预算分配:")
    print(f"  语言层: {llm_min_budget}/{num_llm_layers} ({llm_min_budget/total_budget:.1%})")
    print(f"  视觉层: {vit_min_budget}/{num_vit_layers} ({vit_min_budget/total_budget:.1%})")

    # ========== 步骤5：选择要启用的层 ==========
    enable_layers = []
    
    # 选择LLM层
    if llm_min_budget > 0 and num_llm_layers > 0:
        # 直接取前llm_min_budget个最高分的层
        selected_llm_layers = llm_scores.head(llm_min_budget)["layer"].tolist()
        enable_layers.extend(selected_llm_layers)
        print(f"  选择了 {len(selected_llm_layers)} 个语言层")
    
    # 选择ViT层
    if vit_min_budget > 0 and num_vit_layers > 0:
        # 直接取前vit_min_budget个最高分的层
        selected_vit_layers = vit_scores.head(vit_min_budget)["layer"].tolist()
        enable_layers.extend(selected_vit_layers)
        print(f"  选择了 {len(selected_vit_layers)} 个视觉层")

    # ========== 步骤6：格式化层名 ==========
    if enable_layers:
        if "base_model" in enable_layers[0]:
            enable_layers = [".".join(layer.split(".")[2:]) for layer in enable_layers]
        else:
            enable_layers = [".".join(layer.split(".")[0:]) for layer in enable_layers]

    # ========== 步骤7：验证结果 ==========
    print(f"最终启用的层数: {len(enable_layers)}/{total_budget}")
    
    if len(enable_layers) != total_budget:
        print(f"警告: 实际启用的层数({len(enable_layers)})与预算({total_budget})不符")
        # 如果数量不对，尝试修正
        if len(enable_layers) < total_budget:
            # 这种情况不应该发生，但如果发生了，从所有未选中的层中补充
            all_layers_set = set(scores["layer"])
            selected_set = set(enable_layers)
            remaining = list(all_layers_set - selected_set)
            if remaining:
                # 随机补充不足的层
                import random
                random.seed(42)
                needed = total_budget - len(enable_layers)
                enable_layers.extend(random.sample(remaining, min(needed, len(remaining))))
        else:
            # 超过了就截断
            enable_layers = enable_layers[:total_budget]
    
    assert len(enable_layers) == total_budget, \
        f"启用的层数({len(enable_layers)})不等于预算({total_budget})"

    return enable_layers


def main():
    # ========== 主流程：为所有任务生成D-MoLE架构 ==========
    # 零成本分数目录
    # 根据统计结果，ViT最小预算 = max(总预算的25%, 按比例计算的ViT预算) == 总预算的25%，根据评分计算预算的方法失去作用！（总预算50%）
    # v0
    zc_scores_dir = "results/zc_scores"

    
    tasks = [
        "vizwiz_caption",
        "skvg",
        "textcaps",
        "iconqa",
        "ocrvqa",
        "flickr30k",
        "vizwiz",
        "kvqa",
        "pmcvqa",
    ]
    num_tasks = len(tasks)

    dataframes = {}

    cur_arch = {}

    for i in range(1, num_tasks + 1):
        taskname = tasks[i - 1]  # Map the task names dynamically
        print("任务：", taskname)
        file_name = os.path.join(zc_scores_dir, f"{i}_InternVL2-2B_{taskname}_score.csv")
        if os.path.exists(file_name):
            dataframes[taskname] = pd.read_csv(file_name)

        for layer in dataframes[taskname]["layer"]:
            if layer not in cur_arch:
                cur_arch[layer] = []

        enable_layers = get_enable_layers(dataframes[taskname], budget_portion=0.25)
        for layer in enable_layers:
            if layer not in cur_arch:
                cur_arch[layer] = [i]
            else:
                cur_arch[layer].append(i)

        os.makedirs("dmole_arch", exist_ok=True)
        with open(f"dmole_arch/{i}_InternVL2-2B_{taskname}_arch.json", "w") as f:
            json.dump(cur_arch, f, indent=4, sort_keys=True)

    print("D-MoLE architecture saved.")


if __name__ == "__main__":
    main()