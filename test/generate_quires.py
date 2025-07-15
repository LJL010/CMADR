import json
import random

# 生成指定范围内的随机整数
def generate_random_number(min_value, max_value):
    return random.randint(min_value, max_value)

# 生成 train_queries 列表
def generate_train_queries(num_entries):
    train_queries = []
    for _ in range(num_entries):
        query = {
            "src": generate_random_number(0, 259),
            "dst": generate_random_number(0, 259),
            "time": generate_random_number(0, 476)
        }
        train_queries.append(query)
    return train_queries

# 生成 predict_queries 列表
def generate_predict_queries(num_entries):
    predict_queries = []
    for _ in range(num_entries):
        query = {
            "src": generate_random_number(0, 259),
            "dst": generate_random_number(0, 259),
            "time": generate_random_number(0, 476)
        }
        predict_queries.append(query)
    return predict_queries

# 生成 JSON 数据
def generate_json_data():
    data = {
        "train_queries": generate_train_queries(1000),
        "predict_queries": generate_predict_queries(1000)
    }
    return data

# 将 JSON 数据保存到本地文件
def save_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 主函数
if __name__ == "__main__":
    json_data = generate_json_data()
    save_to_file(json_data, 'random_data.json')
    print("数据已生成并保存到 random_data.json 文件中。")