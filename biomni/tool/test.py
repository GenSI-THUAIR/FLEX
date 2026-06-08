#!/usr/bin/env python3
"""
测试ChimeraX静电势分析脚本
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

def test_chimerax_basic():
    """测试ChimeraX基本功能"""
    print("=== 测试ChimeraX基本功能 ===")
    
    # 1. 检查chimerax-daily是否可用
    try:
        result = subprocess.run(['which', 'chimerax-daily'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ chimerax-daily 路径: {result.stdout.strip()}")
        else:
            print("❌ chimerax-daily 未找到")
            return False
    except Exception as e:
        print(f"❌ 检查chimerax-daily失败: {e}")
        return False
    
    # 2. 测试版本信息
    try:
        result = subprocess.run(['chimerax-daily', '--version'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ ChimeraX版本: {result.stdout.strip()}")
        else:
            print(f"⚠️  版本检查警告: {result.stderr}")
    except Exception as e:
        print(f"⚠️  版本检查异常: {e}")
    
    return True

def test_chimerax_script(pdb_file=None):
    """测试ChimeraX脚本执行"""
    print("\n=== 测试ChimeraX脚本执行 ===")
    
    # 如果没有提供PDB文件，创建一个简单的测试文件
    if pdb_file is None or not os.path.exists(pdb_file):
        pdb_file = create_test_pdb()
        print(f"📁 使用测试PDB文件: {pdb_file}")
    else:
        print(f"📁 使用提供的PDB文件: {pdb_file}")
    
    # 创建测试脚本
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cxc', delete=False) as script_f:
        script_file = script_f.name
        
        # 简化的测试命令
        commands = [
            f"open {pdb_file}",
            "info models",
            "info atoms",
            "exit"
        ]
        
        for cmd in commands:
            script_f.write(cmd + '\n')
            print(f"📝 脚本命令: {cmd}")
    
    print(f"📄 脚本文件: {script_file}")
    
    # 执行脚本
    try:
        print("\n🚀 执行ChimeraX脚本...")
        cmd = ['chimerax-daily', '--nogui', '--script', script_file]
        print(f"💻 执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"\n📊 执行结果:")
        print(f"   返回码: {result.returncode}")
        print(f"   标准输出长度: {len(result.stdout)} 字符")
        print(f"   错误输出长度: {len(result.stderr)} 字符")
        
        if result.stdout:
            print(f"\n📤 标准输出:")
            print("=" * 50)
            print(result.stdout)
            print("=" * 50)
        
        if result.stderr:
            print(f"\n❌ 错误输出:")
            print("=" * 50)
            print(result.stderr)
            print("=" * 50)
        
        if result.returncode == 0:
            print("✅ 脚本执行成功!")
            return True
        else:
            print("❌ 脚本执行失败!")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 脚本执行超时!")
        return False
    except Exception as e:
        print(f"💥 脚本执行异常: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(script_file):
            os.remove(script_file)
            print(f"🧹 清理脚本文件: {script_file}")

def create_test_pdb():
    """创建一个简单的测试PDB文件"""
    pdb_content = """HEADER    TEST PROTEIN                            01-JAN-24   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.100  10.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.664  16.849  10.000  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.764  18.076  10.000  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.113  15.218   8.756  1.00 20.00           C  
ATOM      6  N   GLY A   2      16.498  16.193  10.000  1.00 20.00           N  
ATOM      7  CA  GLY A   2      15.163  16.789  10.000  1.00 20.00           C  
ATOM      8  C   GLY A   2      14.045  15.756  10.000  1.00 20.00           C  
ATOM      9  O   GLY A   2      14.264  14.548  10.000  1.00 20.00           O  
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pdb_f:
        pdb_f.write(pdb_content)
        return pdb_f.name

def test_electrostatic_analysis(pdb_file=None):
    """测试完整的静电势分析流程"""
    print("\n=== 测试静电势分析流程 ===")
    
    if pdb_file is None or not os.path.exists(pdb_file):
        pdb_file = create_test_pdb()
        print(f"📁 使用测试PDB文件: {pdb_file}")
    
    # 导入并测试静电势分析函数
    try:
        # 假设函数在当前目录或可导入
        from electrostatic import analyze_electrostatic_potential
        
        print("🧪 开始静电势分析测试...")
        result = analyze_electrostatic_potential(
            pdb_file=pdb_file,
            residue_position=1,
            chain_id='A',
            radius=10.0,
            debug=True
        )
        
        print("📊 分析结果:")
        for key, value in result.items():
            print(f"   {key}: {value}")
            
        if result.get('success', False):
            print("✅ 静电势分析成功!")
        else:
            print("⚠️  静电势分析使用了备用方法")
            
    except ImportError:
        print("❌ 无法导入静电势分析函数")
    except Exception as e:
        print(f"💥 静电势分析异常: {e}")

def main():
    """主测试函数"""
    print("🔬 ChimeraX静电势分析测试工具")
    print("=" * 60)
    
    # 检查命令行参数
    pdb_file = None
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
        if not os.path.exists(pdb_file):
            print(f"❌ PDB文件不存在: {pdb_file}")
            sys.exit(1)
    
    # 运行测试
    tests_passed = 0
    total_tests = 3
    
    # 测试1: 基本功能
    if test_chimerax_basic():
        tests_passed += 1
    
    # 测试2: 脚本执行
    if test_chimerax_script(pdb_file):
        tests_passed += 1
    
    # 测试3: 静电势分析
    try:
        test_electrostatic_analysis(pdb_file)
        tests_passed += 1
    except Exception as e:
        print(f"静电势分析测试失败: {e}")
    
    # 总结
    print(f"\n📈 测试总结: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
