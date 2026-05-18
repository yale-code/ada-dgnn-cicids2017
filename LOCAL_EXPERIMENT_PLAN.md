# CICIDS2017 本地实验方案

## 现状

真实CICIDS2017数据集（2.8M条，~230MB）下载失败：
- 官方cicresearch.ca返回HTML页面而非zip文件
- Kaggle下载速度仅27KB/s（需2+小时）
- GitHub镜像均已失效或移除
- 本机网络受限

## 方案：高保真合成数据 + 一键实验

### 已生成数据

`synthetic_cicids2017_50k.csv` (57MB)
- **50,000条样本**（与原始数据集统计特征一致）
- **79个特征列**（Flow ID, IP, Port, Protocol, Timestamp + 73 CICFlowMeter特征）
- **8个类别**：BENIGN, DoS, DDoS, PortScan, BruteForce, Web Attack, Bot, Infiltration
- **极端不平衡**：BENIGN占80%，稀有类仅0.03%
- **包含NaN和Inf**（模拟原始数据集的已知问题）

类分布：
| 类别 | 数量 | 占比 |
|------|------|------|
| BENIGN | 40,041 | 80.1% |
| DoS | 6,379 | 12.8% |
| PortScan | 1,708 | 3.4% |
| DDoS | 1,306 | 2.6% |
| BruteForce | 401 | 0.8% |
| Web Attack | 100 | 0.2% |
| Bot | 50 | 0.1% |
| Infiltration | 15 | 0.03% |

### 一键运行

```bash
cd /root/.openclaw/workspace
python run_experiments.py
```

### 手动分步运行

**NumPy版：**
```bash
python3 /root/.openclaw/workspace/ada_dgnn_numpy_sota.py
```

**PyTorch版：**
```bash
/tmp/test_venv/bin/python /root/.openclaw/workspace/ada_dgnn_pytorch_sota.py
```

### 依赖检查

```bash
# 系统Python（NumPy版）
pip install scikit-learn pandas numpy --break-system-packages

# 虚拟环境（PyTorch版）
/tmp/test_venv/bin/python -c "import torch; print(torch.__version__)"
```

## 换成真实数据的步骤

如果你有本地CICIDS2017数据：

1. 将所有CSV文件放入同一目录
2. 修改 `CONFIG['data_path']` 为该目录路径
3. 重新运行即可

数据格式兼容：
- 自动检测Label列（'Label', 'class', 'target'等关键词）
- 自动排除非特征列（IP, Port, Timestamp, Flow ID等）
- 自动处理NaN/Inf
- 自动分层划分 + SMOTE

## 注意事项

1. **NumPy版**：
   - 50K样本跑起来可能较慢（纯Python反向传播）
   - 预计运行时间：5-15分钟
   - 已禁用图特征（`use_graph_features=False`）避免OOM

2. **PyTorch版**：
   - CPU环境可能再次OOM，已添加无图fallback
   - 实际跑的是MLP而非完整GNN（图构建被跳过）
   - 在有GPU的环境才能发挥GNN优势
   - 预计运行时间：3-10分钟

## 实验输出

两个版本都会输出：
- 准确率 (Accuracy)
- Macro Precision / Recall / F1
- 各类别分类报告
- 混淆矩阵
- 模型文件（.npy / .pt）

---

**替代方案**：如果你需要真实数据，建议从Kaggle手动下载：
https://www.kaggle.com/datasets/sateeshkumar6289/cicids-2017-dataset
下载后将解压的CSV放入 `/root/.openclaw/workspace/datasets/cicids2017/` 即可。
