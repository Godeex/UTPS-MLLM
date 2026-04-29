import json
import re
import glob
import os


def extract_layers_by_number(json_file_path, target_number):
    """
    从JSON文件中提取包含指定数字的层名称，并按视觉层和语言层分开排序输出

    Args:
        json_file_path: JSON文件路径
        target_number: 要搜索的数字
    """
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 将目标数字转换为字符串用于匹配
    target_str = str(target_number)

    # 分别存储视觉层和语言层
    vision_layers = []
    language_layers = []

    # 遍历所有层名称
    for layer_name in data.keys():
        # 检查该层的数字列表中是否包含目标数字
        if target_number in data[layer_name]:
            # 根据层名称前缀分类
            if layer_name.startswith('vision_model'):
                vision_layers.append(layer_name)
            elif layer_name.startswith('language_model'):
                language_layers.append(layer_name)
            else:
                # 如果还有其他类型的层，可以在这里处理
                print(f"未知类型的层: {layer_name}")

    # 定义提取层序号的函数
    def extract_layer_number(layer_name, model_type):
        if model_type == 'vision':
            # 匹配视觉层的序号: vision_model.encoder.layers.X
            match = re.search(r'layers\.(\d+)', layer_name)
        else:
            # 匹配语言层的序号: language_model.model.layers.X
            match = re.search(r'layers\.(\d+)', layer_name)

        if match:
            return int(match.group(1))
        return -1

    # 按层序号排序
    vision_layers.sort(key=lambda x: extract_layer_number(x, 'vision'))
    language_layers.sort(key=lambda x: extract_layer_number(x, 'language'))

    # 输出结果
    print(f"包含数字 {target_number} 的层：")
    print("-" * 50)

    print("\n视觉层 (Vision Layers):", len(vision_layers))
    if vision_layers:
        for i, layer in enumerate(vision_layers, 1):
            print(f"  {i}. {layer}")
    else:
        print("  无")

    print("\n语言层 (Language Layers):", len(language_layers))
    if language_layers:
        for i, layer in enumerate(language_layers, 1):
            print(f"  {i}. {layer}")
    else:
        print("  无")


def find_and_read_first_json(target_number, directory='.'):
    """
    在指定目录下查找以target_number开头的第一个JSON文件并读取

    Args:
        target_number: 要匹配的文件名前缀
        directory: 要搜索的目录，默认为当前目录

    Returns:
        读取的JSON数据，如果未找到则返回None
    """
    # 构建匹配模式：target_number*.json
    pattern = os.path.join(directory, f"{target_number}*.json")

    # 查找所有匹配的文件
    matched_files = glob.glob(pattern)

    if not matched_files:
        print(f"未找到以 '{target_number}' 开头的JSON文件")
        return None

    # 获取第一个匹配的文件
    first_match = matched_files[0]

    return first_match


def main():
    # INDEX = 2.2
    # if INDEX == 0:
    #     json_file_path = "dmole_arch/1_InternVL2-2B_vizwiz_caption_arch.json"
    # elif INDEX == 2: # activation & g
    #     json_file_path = "dmole_arch_v_2/1_InternVL2-2B_vizwiz_caption_arch.json"
    # elif INDEX == 2.1:  # weight * g
    #     json_file_path = "dmole_arch_v_2_1/1_InternVL2-2B_vizwiz_caption_arch.json"
    # elif INDEX == 2.2:  # weight & activation
    #     json_file_path = "dmole_arch_v_2_2/1_InternVL2-2B_vizwiz_caption_arch.json"

    target_number = 8
    json_file_path = find_and_read_first_json(target_number, "dmole_arch_v_2_2")
    # json_file_path = find_and_read_first_json(target_number, "dmole_arch_v_2_5")
    # json_file_path = find_and_read_first_json(target_number, "arch_sparse_v_1")
    # json_file_path = find_and_read_first_json(target_number, "arch_losa_v_1")



    try:
        extract_layers_by_number(json_file_path, target_number)
    except ValueError:
        print("错误：请输入有效的整数")
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

# v1