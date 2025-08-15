#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：如何使用hparams_registry配置Memory Mosaic Transformer

这个脚本展示了如何通过标准的ICRM配置系统来使用Memory Mosaic
"""

import torch
import networks
from hparams_registry import default_hparams, random_hparams


def demo_default_hparams():
    """演示使用默认超参数配置"""
    print("=== 使用默认hparams配置Memory Mosaic ===")
    
    # 获取ICRM算法在ColoredMNIST数据集上的默认配置
    hparams = default_hparams('ICRM', 'ColoredMNIST')
    
    # 启用Memory Mosaic
    hparams['use_mosaic'] = True  # 手动启用，或者修改registry中的默认值
    
    print("默认配置参数:")
    mosaic_params = ['use_mosaic', 'n_embd', 'n_layer', 'n_head', 'context_length', 
                     'pmem_size', 'pmem_count', 'dropout']
    for param in mosaic_params:
        if param in hparams:
            print(f"  {param}: {hparams[param]}")
    
    # 创建模型
    input_features = 784  # MNIST: 28x28
    output_classes = 10
    
    # 添加必要的参数
    hparams['is_transformer'] = True
    hparams['nonlinear_classifier'] = False
    
    classifier = networks.Classifier(input_features, output_classes, hparams)
    
    print(f"\n创建的模型: {type(classifier).__name__}")
    if hasattr(classifier, 'name'):
        print(f"模型配置: {classifier.name}")
    
    return classifier, hparams


def demo_random_hparams():
    """演示使用随机超参数搜索"""
    print("\n=== 使用随机hparams进行超参数搜索 ===")
    
    seeds = [42, 123, 456]  # 不同的随机种子
    
    for i, seed in enumerate(seeds):
        print(f"\n--- 配置 {i+1} (seed={seed}) ---")
        
        # 获取随机配置
        hparams = random_hparams('ICRM', 'ColoredMNIST', seed)
        hparams['use_mosaic'] = True
        hparams['is_transformer'] = True
        
        print("随机配置:")
        mosaic_params = ['n_embd', 'n_layer', 'n_head', 'pmem_size', 'pmem_count', 'dropout']
        for param in mosaic_params:
            if param in hparams:
                print(f"  {param}: {hparams[param]}")
        
        # 估算参数数量（简单估算）
        n_embd = hparams['n_embd']
        n_layer = hparams['n_layer']
        pmem_size = hparams['pmem_size']
        pmem_count = hparams['pmem_count']
        
        # 简单估算（不完全准确，仅供参考）
        estimated_params = (n_embd * 784 +  # 输入投影
                           n_layer * n_embd * n_embd * 4 +  # 每层的大致参数
                           pmem_count * pmem_size * n_embd +  # 持久记忆
                           n_embd * 10)  # 输出投影
        
        print(f"  估算参数数量: {estimated_params:,}")


def demo_dataset_specific_configs():
    """展示不同数据集的特定配置"""
    print("\n=== 不同数据集的Memory Mosaic配置 ===")
    
    datasets = ['FEMNIST', 'WILDSCamelyon', 'ColoredMNIST']
    
    for dataset in datasets:
        print(f"\n--- {dataset} 数据集 ---")
        hparams = default_hparams('ICRM', dataset)
        hparams['use_mosaic'] = True
        
        key_params = ['n_embd', 'n_layer', 'n_head', 'pmem_size', 'pmem_count', 
                      'batch_size', 'context_length']
        for param in key_params:
            if param in hparams:
                print(f"  {param}: {hparams[param]}")


def demo_complete_training_config():
    """展示完整的训练配置示例"""
    print("\n=== 完整的Memory Mosaic训练配置 ===")
    
    # 基础配置
    algorithm = 'ICRM'
    dataset = 'ColoredMNIST'
    
    # 获取默认配置
    hparams = default_hparams(algorithm, dataset)
    
    # 启用Memory Mosaic并自定义一些参数
    hparams.update({
        'use_mosaic': True,
        'n_embd': 256,      # 可以覆盖默认值
        'pmem_count': 2,    # 使用更多的持久记忆
        'dropout': 0.1,
        
        # 训练特定参数
        'max_steps': 5000,
        'eval_every': 500,
        'patience': 3,
        
        # 数据和模型路径
        'data_dir': './data/',
        'output_dir': './results/',
        'pretrained_model_path': './feat_pretrained_model/model.pkl'
    })
    
    print("完整训练配置:")
    print("```python")
    print("hparams = {")
    for key, value in sorted(hparams.items()):
        print(f"    '{key}': {repr(value)},")
    print("}")
    print("```")
    
    return hparams


def compare_gpt2_vs_mosaic():
    """比较GPT2和Memory Mosaic的配置"""
    print("\n=== GPT2 vs Memory Mosaic 配置对比 ===")
    
    base_hparams = default_hparams('ICRM', 'ColoredMNIST')
    
    # GPT2配置
    gpt2_hparams = base_hparams.copy()
    gpt2_hparams.update({
        'is_transformer': True,
        'use_mosaic': False,  # 使用GPT2
    })
    
    # Memory Mosaic配置
    mosaic_hparams = base_hparams.copy() 
    mosaic_hparams.update({
        'is_transformer': True,
        'use_mosaic': True,   # 使用Memory Mosaic
    })
    
    print("GPT2 Transformer 配置:")
    gpt2_keys = ['n_embd', 'n_layer', 'n_head', 'context_length']
    for key in gpt2_keys:
        if key in gpt2_hparams:
            print(f"  {key}: {gpt2_hparams[key]}")
    
    print("\nMemory Mosaic Transformer 配置:")
    mosaic_keys = ['n_embd', 'n_layer', 'n_head', 'context_length', 
                   'pmem_size', 'pmem_count', 'dropout']
    for key in mosaic_keys:
        if key in mosaic_hparams:
            print(f"  {key}: {mosaic_hparams[key]}")
    
    print(f"\n主要区别:")
    print(f"• GPT2: 标准的transformer注意力机制")
    print(f"• Memory Mosaic: Context Memory + Persistent Memory ({mosaic_hparams.get('pmem_count', 1)} × {mosaic_hparams.get('pmem_size', 2688)})")


if __name__ == "__main__":
    print("=== ICRM + Memory Mosaic 配置系统演示 ===\n")
    
    try:
        # 演示各种配置方式
        demo_default_hparams()
        demo_random_hparams() 
        demo_dataset_specific_configs()
        compare_gpt2_vs_mosaic()
        
        # 生成完整配置
        complete_config = demo_complete_training_config()
        
        print(f"\n=== 使用建议 ===")
        print(f"1. 直接使用: hparams = default_hparams('ICRM', 'YourDataset')")
        print(f"2. 启用Memory Mosaic: hparams['use_mosaic'] = True")
        print(f"3. 自定义参数: hparams.update({{'pmem_size': 4096, 'pmem_count': 4}})")
        print(f"4. 超参数搜索: random_hparams('ICRM', 'YourDataset', seed)")
        print(f"\n现在所有Memory Mosaic参数都已集成到标准的hparams系统中！")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc() 