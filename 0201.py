# -*- coding: utf-8 -*-
"""
Lecture 0201 练习：CLI + IPO + OOP + Backend-Model-Process
功能：生成数据 → 读取数据 → 计算指标 → 面向对象结构化
"""

import csv
import random
import sys

# ====================== 挑战1：生成数据 ======================
def generate_data(filepath="data.csv", n=10000):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for _ in range(n):
            writer.writerow([random.uniform(0, 1), random.uniform(0, 1)])
    print(f"✅ 已生成 {n} 行数据：{filepath}")

# ====================== 挑战2：处理函数（IPO） ======================
def process_data_point(point):
    """单个数据点处理：返回 x+y、x*y 等指标（可自定义）"""
    x, y = point
    return {
        "x": x,
        "y": y,
        "sum": x + y,
        "product": x * y
    }

def aggregate_results(results):
    """汇总统计：平均值、总数"""
    count = len(results)
    sum_sum = sum(r["sum"] for r in results)
    sum_product = sum(r["product"] for r in results)
    return {
        "count": count,
        "avg_sum": sum_sum / count,
        "avg_product": sum_product / count
    }

# ====================== 挑战3：OOP 后端 + 模型 ======================
class DataBackend:
    """后端：负责读取文件，yield 数据点"""
    def __init__(self, filepath):
        self.filepath = filepath

    def read(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row["x"])
                y = float(row["y"])
                yield (x, y)

class DataProcessor:
    """模型/处理层：接收后端 + 处理函数"""
    def __init__(self, backend, process_func):
        self.backend = backend
        self.process_func = process_func
        self.results = []

    def run(self):
        """遍历所有数据点并处理"""
        for point in self.backend.read():
            res = self.process_func(point)
            self.results.append(res)
        return self.results

# ====================== CLI 入口 ======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  生成数据：python main.py generate")
        print("  处理数据：python main.py process")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "generate":
        generate_data()

    elif mode == "process":
        backend = DataBackend("data.csv")
        processor = DataProcessor(backend, process_data_point)
        results = processor.run()
        stats = aggregate_results(results)

        print("📊 处理完成，统计结果：")
        print(f"总行数：{stats['count']}")
        print(f"x+y 平均值：{stats['avg_sum']:.4f}")
        print(f"x*y 平均值：{stats['avg_product']:.4f}")

    else:
        print("❌ 无效模式，可选：generate / process")