#!/usr/bin/env python3
"""
内存优化的ICRM运行脚本
专门针对内存不足的情况进行优化
"""

import os
import sys
import torch
import gc

def optimize_memory():
    """内存优化设置"""
    # 设置环境变量减少内存使用
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'  
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # 禁用数据加载器的多进程
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # 强制垃圾回收
    gc.collect()
    
    print("=== 内存优化设置完成 ===")
    print(f"• 线程数限制为1")
    print(f"• 禁用CUDA异步执行") 
    print(f"• 强制垃圾回收")

def run_with_low_memory():
    """运行ICRM，内存优化版本"""
    
    # 先进行内存优化
    optimize_memory()
    
    # 导入必要模块（延迟导入减少初始内存占用）
    print("=== 开始导入模块 ===")
    try:
        import argparse
        import json
        print("✓ 基础模块导入成功")
        
        import numpy as np  
        print("✓ NumPy导入成功")
        
        import torch
        print(f"✓ PyTorch导入成功 (版本: {torch.__version__})")
        
        import utils
        import dataset as dataset_file  
        import hparams_registry
        import algorithms
        print("✓ ICRM模块导入成功")
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    # 设置参数
    print("=== 设置运行参数 ===")
    
    # 内存优化的超参数
    hparams = hparams_registry.default_hparams('ICRM', 'ColouredMNIST')
    
    # 关键设置
    hparams.update({
        'use_mosaic': True,           # 启用Memory Mosaic
        'pretrained_model_path': 'G:/feta/FeAT/ColoredMNIST/mmmlp_best.pth',
        
        # 内存优化设置
        'batch_size': 16,            # 减小batch size 
        'test_batch_size': 32,       # 减小test batch size
        'n_embd': 64,               # 减小embedding维度
        'n_layer': 4,               # 减少层数
        'n_head': 4,                # 减少注意力头数
        'pmem_size': 512,           # 减小持久记忆大小
        'pmem_count': 1,            # 减少持久记忆数量
        'context_length': 50,       # 减小上下文长度
        
        # 其他优化
        'is_parallel': False,       # 禁用并行
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    print("内存优化参数:")
    memory_params = ['batch_size', 'n_embd', 'n_layer', 'pmem_size', 'use_mosaic']
    for param in memory_params:
        print(f"  {param}: {hparams[param]}")
    
    # 创建数据集
    print("=== 创建数据集 ===")
    try:
        dataset = dataset_file.get_dataset_class('ColouredMNIST')(
            root='/mnt/data02/gll_yong/ICRM/data/MNIST',  # 你的数据路径
            test_envs=[2], 
            hparams=hparams
        )
        print("✓ 数据集创建成功")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        print("请检查数据路径是否正确")
        return False
    
    # 创建算法
    print("=== 创建ICRM算法 ===")
    try:
        algorithm = algorithms.ICRM(
            dataset.input_shape,
            dataset.num_classes, 
            hparams
        )
        print("✓ ICRM算法创建成功")
        print(f"✓ 模型类型: {type(algorithm.classifier).__name__}")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in algorithm.network.parameters())
        print(f"✓ 总参数数量: {total_params:,}")
        
    except Exception as e:
        print(f"✗ 算法创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 内存优化启动成功！")
    print("现在可以进行训练或测试了。")
    return True

if __name__ == "__main__":
    success = run_with_low_memory()
    
    if success:
        print("\n=== 内存使用建议 ===")
        print("1. 如果仍然内存不足，可以进一步减小:")
        print("   • batch_size (当前16，可以试8或4)")
        print("   • n_embd (当前64，可以试32)")
        print("   • pmem_size (当前512，可以试256)")
        print("2. 监控系统内存使用情况")
        print("3. 考虑增加虚拟内存大小")
    else:
        print("\n=== 故障排除建议 ===")
        print("1. 重启系统释放内存")
        print("2. 关闭其他程序")
        print("3. 增加虚拟内存:")
        print("   控制面板 → 系统 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存")
        print("4. 如果有独立显卡，确保PyTorch使用GPU而不是CPU") 