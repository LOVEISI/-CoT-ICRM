#!/usr/bin/env python3
"""
简单的Memory Mosaic导入测试
避免内存不足问题
"""

import os
import sys

def test_basic_imports():
    """测试基本导入，无内存压力"""
    
    print("=== 简单导入测试 ===")
    
    try:
        print("1. 测试Python基础模块...")
        import json
        import argparse
        print("✓ 基础模块OK")
        
        print("2. 测试numpy (小心内存)...")
        import numpy as np
        print(f"✓ NumPy OK (版本: {np.__version__})")
        
        print("3. 测试PyTorch (小心内存)...")
        import torch
        print(f"✓ PyTorch OK (版本: {torch.__version__})")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        print("4. 测试ICRM模块...")
        import utils
        print("✓ utils模块OK")
        
        import hparams_registry
        print("✓ hparams_registry模块OK")
        
        print("5. 测试networks模块 (包含Memory Mosaic)...")
        import networks
        print("✓ networks模块导入成功")
        
        print("6. 测试Memory Mosaic可用性...")
        # 使用最小的参数测试
        hparams = {
            'is_transformer': True,
            'use_mosaic': True,
            'n_embd': 32,      # 很小的参数
            'n_layer': 2,
            'n_head': 2,
            'context_length': 10,
            'pmem_size': 64,
            'pmem_count': 1,
            'dropout': 0.0,
            'nonlinear_classifier': False
        }
        
        # 尝试创建一个很小的分类器
        classifier = networks.Classifier(10, 2, hparams)  # 10输入，2输出
        print(f"✓ Memory Mosaic分类器创建成功: {type(classifier).__name__}")
        
        if hasattr(classifier, 'name'):
            print(f"  模型配置: {classifier.name}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in classifier.parameters())
        print(f"  参数数量: {total_params:,}")
        
        print("\n🎉 所有测试通过！Memory Mosaic集成工作正常！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_registry():
    """测试数据集注册表"""
    
    print("\n=== 数据集注册表测试 ===")
    
    try:
        import dataset as dataset_file
        from hparams_registry import default_hparams
        
        # 测试获取hparams
        hparams = default_hparams('ICRM', 'ColouredMNIST')
        print(f"✓ ColouredMNIST hparams获取成功")
        print(f"  关键参数: batch_size={hparams.get('batch_size')}, context_length={hparams.get('context_length')}")
        
        # 测试数据集类获取 (不实际创建数据，避免内存问题)
        dataset_class = dataset_file.get_dataset_class('ColouredMNIST')
        print(f"✓ ColouredMNIST数据集类获取成功: {dataset_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始简单测试，避免内存问题...")
    print("=" * 50)
    
    success1 = test_basic_imports()
    success2 = test_data_registry()
    
    print("=" * 50)
    if success1 and success2:
        print("🎉 全部测试通过！")
        print("\n下一步建议:")
        print("1. 使用 python run_low_memory.py 进行内存优化运行")
        print("2. 或者先重启系统释放内存，再运行原始命令")
        print("3. 检查系统虚拟内存设置")
    else:
        print("❌ 部分测试失败")
        print("\n故障排除:")
        print("1. 重启Python环境")  
        print("2. 检查MemoryMosaics路径")
        print("3. 确保所有依赖包已安装") 