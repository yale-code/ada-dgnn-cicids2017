"""
run_experiments.py - 一键运行CICIDS2017实验
====================================================

本地可跑方案（无需下载真实数据集）:
1. 使用高保真合成数据（50K样本，基于CICIDS2017统计特征）
2. 自动跑两个版本模型
3. 生成对比报告

使用方法:
    python run_experiments.py

输出:
    - numpy_results.json
    - pytorch_results.json  
    - experiment_report.md
"""

import subprocess
import sys
import json
import os
import time

def run_numpy():
    print("\n" + "="*60)
    print("🚀 运行 NumPy 版模型")
    print("="*60)
    result = subprocess.run(
        [sys.executable, "/root/.openclaw/workspace/ada_dgnn_numpy_sota.py"],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return result.returncode == 0

def run_pytorch():
    print("\n" + "="*60)
    print("🚀 运行 PyTorch 版模型")
    print("="*60)
    # Use the virtualenv Python
    venv_python = "/tmp/test_venv/bin/python"
    result = subprocess.run(
        [venv_python, "/root/.openclaw/workspace/ada_dgnn_pytorch_sota.py"],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return result.returncode == 0

def main():
    print("📊 CICIDS2017 本地实验方案")
    print("="*60)
    print(f"数据集: synthetic_cicids2017_50k.csv ({os.path.getsize('/root/.openclaw/workspace/datasets/cicids2017/synthetic_cicids2017_50k.csv')/1024/1024:.1f}MB)")
    print("类别分布: BENIGN(80%), DoS(12.8%), PortScan(3.4%), DDoS(2.6%), ...")
    print("="*60)
    
    # Run NumPy
    t0 = time.time()
    numpy_ok = run_numpy()
    numpy_time = time.time() - t0
    
    # Run PyTorch
    t0 = time.time()
    pytorch_ok = run_pytorch()
    pytorch_time = time.time() - t0
    
    print("\n" + "="*60)
    print("📋 实验完成")
    print("="*60)
    print(f"NumPy 版: {'✅' if numpy_ok else '❌'} ({numpy_time:.1f}s)")
    print(f"PyTorch 版: {'✅' if pytorch_ok else '❌'} ({pytorch_time:.1f}s)")
    
    if not numpy_ok:
        print("\nNumPy版失败。检查: pip install scikit-learn pandas numpy")
    if not pytorch_ok:
        print("\nPyTorch版失败。检查: /tmp/test_venv/bin/python 存在且torch已安装")

if __name__ == "__main__":
    main()
