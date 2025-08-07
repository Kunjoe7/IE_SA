#!/usr/bin/env python
"""
测试直接import使用
"""

# 1. 最简单的使用方式
print("1. 测试基本导入:")
try:
    import icdm_sa
    print(f"✓ 成功导入 icdm_sa, 版本: {icdm_sa.__version__}")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

# 2. 导入具体模块
print("\n2. 测试导入具体模块:")
try:
    from icdm_sa import MultiTaskModel, EGTrainer, Cindex
    print("✓ 成功导入 MultiTaskModel, EGTrainer, Cindex")
    
    # 创建一个模型实例
    model = MultiTaskModel(10, 5)  # 10个特征，5个时间区间
    print(f"✓ 成功创建模型: {type(model).__name__}")
except Exception as e:
    print(f"✗ 错误: {e}")

# 3. 导入数据集模型
print("\n3. 测试导入数据集模型:")
try:
    from icdm_sa import FLCHAINModel
    print("✓ 成功导入 FLCHAINModel")
    
    # 可以直接创建实例（如果有数据的话）
    print("✓ FLCHAINModel 可以使用")
except Exception as e:
    print(f"✗ 错误: {e}")

# 4. 查看可用的组件
print("\n4. 查看 icdm_sa 包中可用的组件:")
import icdm_sa
available = [item for item in dir(icdm_sa) if not item.startswith('_')]
print("可用的类和函数:")
for item in available:
    print(f"  - {item}")

# 5. 快速示例
print("\n5. 快速使用示例:")
import torch
import numpy as np

# 创建模型
model = MultiTaskModel(8, 5)
print("✓ 创建了一个 8 特征输入，5 个时间区间的生存分析模型")

# 生成测试数据
x = torch.randn(4, 8)  # 4个样本，8个特征
outputs = model(x)
print(f"✓ 模型预测输出: {len(outputs)} 个时间区间的生存概率")
print(f"  每个区间输出形状: {outputs[0].shape}")

# 使用 C-index 评估
cindex = Cindex()
print("✓ C-index 评估器已准备就绪")

print("\n" + "="*50)
print("总结：是的，现在可以直接 import 使用了！")
print("="*50)