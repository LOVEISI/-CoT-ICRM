#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：如何在ICRM中使用Memory Mosaic Transformer

这个脚本展示了如何设置hparams来使用MosaicTransformer而不是GPT2Transformer
"""

import torch
import networks

def example_usage():
    """演示如何使用MosaicTransformer"""
    
    # 设置超参数
    hparams = {
        'is_transformer': True,
        'use_mosaic': True,  # 设置为True来使用Memory Mosaic
        
        # Memory Mosaic特定参数
        'n_embd': 256,        # embedding维度
        'n_layer': 6,         # transformer层数  
        'n_head': 8,          # 注意力头数
        'context_length': 512, # 上下文长度
        'dropout': 0.1,       # dropout率
        'pmem_size': 1024,    # 持久记忆大小
        'pmem_count': 2,      # 持久记忆数量
        
        # 其他参数
        'nonlinear_classifier': False
    }
    
    # 输入和输出维度
    input_features = 784  # 例如MNIST的28x28=784
    output_classes = 10   # 分类数量
    
    # 创建分类器（会自动选择MosaicTransformer）
    classifier = networks.Classifier(input_features, output_classes, hparams)
    
    print(f"创建的分类器: {type(classifier).__name__}")
    if hasattr(classifier, 'name'):
        print(f"模型名称: {classifier.name}")
    
    # 创建示例输入数据
    batch_size = 4
    seq_length = 32
    xs = torch.randn(batch_size, seq_length, input_features)
    ys = torch.randint(0, output_classes, (batch_size, seq_length))
    inds = None  # 使用所有位置
    past_key_values = None
    
    inputs = (xs, ys, inds, past_key_values)
    
    # 前向传播
    with torch.no_grad():
        outputs, new_past = classifier(inputs)
        print(f"输出形状: {outputs.shape}")
        print(f"期望形状: ({batch_size}, {seq_length}, {output_classes})")
        print(f"新的缓存: {type(new_past) if new_past is not None else None}")
    
    return classifier

def test_context_caching():
    """测试上下文缓存功能"""
    
    print("\n=== 测试上下文缓存 ===")
    
    # 设置较小的参数以便快速测试
    hparams = {
        'is_transformer': True,
        'use_mosaic': True,
        'n_embd': 128,
        'n_layer': 2,
        'n_head': 4,
        'context_length': 256,
        'dropout': 0.0,
        'pmem_size': 256,
        'pmem_count': 1,
        'nonlinear_classifier': False
    }
    
    input_features = 100
    output_classes = 5
    
    classifier = networks.Classifier(input_features, output_classes, hparams)
    
    batch_size = 2
    seq_length1 = 10
    seq_length2 = 15
    
    # 第一次前向传播
    xs1 = torch.randn(batch_size, seq_length1, input_features)
    ys1 = torch.randint(0, output_classes, (batch_size, seq_length1))
    inputs1 = (xs1, ys1, None, None)
    
    with torch.no_grad():
        outputs1, past1 = classifier(inputs1)
        print(f"第一次输出形状: {outputs1.shape}")
        print(f"缓存类型: {type(past1)}")
        if past1 is not None and 'cached_embeds' in past1:
            print(f"缓存的embeddings形状: {past1['cached_embeds'].shape}")
    
    # 第二次前向传播，使用之前的缓存
    xs2 = torch.randn(batch_size, seq_length2, input_features)  
    ys2 = torch.randint(0, output_classes, (batch_size, seq_length2))
    inputs2 = (xs2, ys2, None, past1)  # 使用之前的缓存
    
    with torch.no_grad():
        outputs2, past2 = classifier(inputs2)
        print(f"第二次输出形状: {outputs2.shape}")
        print(f"应该等于当前序列长度: {seq_length2}")
        if past2 is not None and 'cached_embeds' in past2:
            print(f"新缓存的embeddings形状: {past2['cached_embeds'].shape}")
            print(f"总上下文长度: {past2['seq_len']}")

def compare_models():
    """比较GPT2Transformer和MosaicTransformer的参数数量"""
    
    input_features = 784
    output_classes = 10
    
    base_hparams = {
        'is_transformer': True,
        'n_embd': 256,
        'n_layer': 6,
        'n_head': 8, 
        'context_length': 512,
        'dropout': 0.1,
        'nonlinear_classifier': False
    }
    
    # GPT2 Transformer
    gpt2_hparams = base_hparams.copy()
    gpt2_hparams['use_mosaic'] = False
    
    try:
        gpt2_classifier = networks.Classifier(input_features, output_classes, gpt2_hparams)
        gpt2_params = sum(p.numel() for p in gpt2_classifier.parameters())
        print(f"GPT2 Transformer 参数数量: {gpt2_params:,}")
    except Exception as e:
        print(f"GPT2 Transformer 创建失败: {e}")
    
    # Mosaic Transformer  
    mosaic_hparams = base_hparams.copy()
    mosaic_hparams['use_mosaic'] = True
    mosaic_hparams['pmem_size'] = 1024
    mosaic_hparams['pmem_count'] = 2
    
    try:
        mosaic_classifier = networks.Classifier(input_features, output_classes, mosaic_hparams)
        mosaic_params = sum(p.numel() for p in mosaic_classifier.parameters())
        print(f"Memory Mosaic Transformer 参数数量: {mosaic_params:,}")
    except Exception as e:
        print(f"Memory Mosaic Transformer 创建失败: {e}")

def create_icrm_hparams_example():
    """展示ICRM算法中使用Memory Mosaic的完整hparams配置"""
    
    print("\n=== ICRM算法中使用Memory Mosaic的配置示例 ===")
    
    # 完整的ICRM + Memory Mosaic配置
    hparams = {
        # 基本设置
        'algorithm': 'ICRM',
        'dataset': 'ColoredMNIST',  # 或其他数据集
        
        # Transformer设置  
        'is_transformer': True,
        'use_mosaic': True,  # 关键：使用Memory Mosaic
        
        # Memory Mosaic特定参数
        'n_embd': 512,        # embedding维度
        'n_layer': 8,         # transformer层数
        'n_head': 8,          # 注意力头数  
        'context_length': 100, # 上下文长度
        'dropout': 0.1,       # dropout率
        'pmem_size': 2048,    # 持久记忆大小
        'pmem_count': 2,      # 持久记忆数量
        
        # ICRM特定设置
        'pretrained_model_path': './feat_pretrained_model/model.pkl',
        'metrics': ['accuracy'],
        
        # 训练设置
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'max_steps': 10000,
        
        # 网络架构
        'nonlinear_classifier': False,
        'mlp_width': 500,
        'mlp_depth': 3,
        
        # 其他设置
        'seed': 42,
        'data_dir': './data/',
    }
    
    print("完整的hparams配置示例:")
    for key, value in hparams.items():
        print(f"  '{key}': {repr(value)},")
    
    print(f"\n使用这个配置时，ICRM会自动选择MosaicTransformer作为分类器。")
    return hparams

if __name__ == "__main__":
    print("=== Memory Mosaic Transformer 使用示例 ===")
    print()
    
    print("1. 基本使用示例:")
    try:
        classifier = example_usage()
        print("✓ 成功创建并测试MosaicTransformer")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    print("\n2. 上下文缓存测试:")
    try:
        test_context_caching()
        print("✓ 上下文缓存功能正常")
    except Exception as e:
        print(f"✗ 缓存测试失败: {e}")
    
    print("\n3. 模型对比:")
    try:
        compare_models()
    except Exception as e:
        print(f"✗ 对比失败: {e}")
    
    # 展示完整配置
    create_icrm_hparams_example()
    
    print("\n=== 使用说明 ===")
    print("要在你的ICRM训练代码中使用Memory Mosaic Transformer，只需:")
    print("1. 在hparams中设置 'is_transformer': True")
    print("2. 在hparams中设置 'use_mosaic': True") 
    print("3. 配置Memory Mosaic特定参数如pmem_size, pmem_count等")
    print("4. 其他训练过程与使用GPT2Transformer完全相同")
    print("5. 模型会自动处理上下文缓存和快速推理")
    
    print("\n=== 重要提醒 ===")
    print("• Memory Mosaic使用持久记忆机制，可能比GPT2有更好的长上下文性能")
    print("• pmem_size和pmem_count参数控制持久记忆的大小和数量")
    print("• 建议先用较小的参数测试，再根据性能需求调整")
    print("• 上下文缓存机制已经兼容ICRM的评估逻辑") 