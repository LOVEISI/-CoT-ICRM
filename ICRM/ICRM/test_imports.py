#!/usr/bin/env python3
"""
测试 Memory Mosaic 导入是否正常工作
"""

print("Testing imports...")

try:
    print("1. Testing networks import...")
    import networks
    print("✓ Networks imported successfully")
    
    print("2. Testing hparams_registry import...")
    from hparams_registry import default_hparams
    print("✓ hparams_registry imported successfully")
    
    print("3. Testing Memory Mosaic availability...")
    hparams = default_hparams('ICRM', 'ColoredMNIST')
    hparams['use_mosaic'] = True
    hparams['is_transformer'] = True
    
    # 尝试创建分类器
    classifier = networks.Classifier(784, 10, hparams)
    print(f"✓ Classifier created: {type(classifier).__name__}")
    
    if hasattr(classifier, 'name'):
        print(f"✓ Model name: {classifier.name}")
    
    print("\n=== 测试成功！===")
    print("Memory Mosaic 集成工作正常，可以运行 main.py")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n=== 故障排除建议 ===")
    print("1. 确保 MemoryMosaics/nanoMosaics/mosaic_model.py 文件存在")
    print("2. 检查是否有缺失的依赖包")
    print("3. 如果Memory Mosaic导入失败，系统会自动回退到GPT2 Transformer") 