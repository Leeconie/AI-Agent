import json
import re

def process_json_array_to_jsonl(input_file, output_file):
    # 步骤1：完整读取整个 JSON 文件（适配多行长格式 JSON 数组）
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            # 读取全部内容，忽略格式中的空白/换行，解析为 JSON 对象（数组）
            json_data = json.load(f_in)
    except json.JSONDecodeError as e:
        print(f"错误：整个 JSON 文件格式无效，无法解析。详细错误：{e}")
        return
    except FileNotFoundError as e:
        print(f"错误：文件不存在。详细错误：{e}")
        return

    # 步骤2：遍历 JSON 数组，处理每个 chat 对象，写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item_num, item in enumerate(json_data, start=1):
            # 跳过缺少 chat 字段的对象
            if 'chat' not in item:
                print(f"提示：跳过第 {item_num} 个对象（缺少 'chat' 字段）")
                continue

            # 处理 chat 数组，移除多余换行和首尾空白
            processed_chat = []
            for msg in item['chat']:
                if isinstance(msg, str):
                    # 移除字符串内所有换行符，清理首尾空白，确保格式规整
                    cleaned_msg = re.sub(r'\n+', '', msg.strip())
                else:
                    cleaned_msg = str(msg)
                processed_chat.append(cleaned_msg)

            # 步骤3：构造新的 JSON 对象，按 JSONL 格式写入（一行一个）
            new_data = {"chat": processed_chat}
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write('\n')

    print("处理完成！合法内容已写入 JSONL 输出文件。")

# 示例调用（替换为你的实际文件路径）
if __name__ == "__main__":
    # 你的输入文件：多行长格式 JSON 数组文件（后缀可是 .json 或 .jsonl，内容符合即可）
    input_path = r".\mira\code_jokes.jsonl"
    # 输出文件：符合要求的 JSONL 文件（一行一个 {"chat": [...]}）
    output_path = r".\mira\tests\output.jsonl"

    process_json_array_to_jsonl(input_path, output_path)