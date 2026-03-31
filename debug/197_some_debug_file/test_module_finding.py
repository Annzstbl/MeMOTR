#!/usr/bin/env python3
"""
Test script for the improved module finding functionality
"""
import torch
import torch.nn as nn

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.linear = nn.Linear(128, 10)
        
        # 嵌套模块
        self.nested = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # 更深层的嵌套
        self.deep_nested = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.Linear(10, 15),
                nn.ReLU()
            ),
            'layer2': nn.Sequential(
                nn.Linear(15, 20),
                nn.Tanh()
            )
        })

def test_find_modules():
    """Test the find_modules_by_names function"""
    from decoder_spectral import find_modules_by_names, list_all_modules
    
    # 创建测试模型
    model = TestModule()
    
    print("=== Testing Module Finding ===")
    
    # 测试列出所有模块
    print("\n1. All available modules:")
    all_modules = list_all_modules(model)
    for i, name in enumerate(all_modules):
        print(f"   {i+1:2d}. {name}")
    
    # 测试查找模块
    print("\n2. Testing module finding:")
    
    # 测试精确路径
    test_names = [
        "conv1",
        "nested",
        "deep_nested.layer1",
        "deep_nested.layer2.0",
        "MeMOTR-Decoder-Spectral_embed",  # 这个应该找不到
        "Linear",  # 这个应该通过模糊匹配找到
        "Conv2d"   # 这个应该通过模糊匹配找到
    ]
    
    found_modules = find_modules_by_names(model, test_names)
    
    for name in test_names:
        if name in found_modules:
            module = found_modules[name]
            print(f"   ✓ Found '{name}': {type(module).__name__}")
        else:
            print(f"   ✗ Not found: '{name}'")
    
    print(f"\nTotal modules found: {len(found_modules)}")

if __name__ == "__main__":
    test_find_modules() 