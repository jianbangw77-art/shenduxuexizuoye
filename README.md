# 深度学习期末材料：缺陷/异常检测

本项目完成“计算机视觉中的缺陷检测/异常检测”方向作业：从 0 使用 PyTorch 复现一个 DRAEM-Lite 风格 baseline，并加入改进策略进行对比实验。

## 目录

- `anomaly_detection/`: 完整 PyTorch 代码包
- `run_experiment.py`: 单次训练/测试入口
- `run_comparison.py`: baseline 与 improved 一键对比入口
- `report/report.md`: 论文格式实验报告
- `requirements.txt`: 运行依赖

## 方法

- **Baseline**: DRAEM-Lite。仅用正常样本训练，通过合成异常构造伪标签；模型包含重建网络和判别分割网络。
- **Improved**: 在 baseline 上加入更丰富的缺陷合成、多尺度分割监督、Focal loss 与 Dice loss，提高小缺陷和不规则缺陷的召回。

## 快速运行

```powershell
pip install -r requirements.txt
python run_comparison.py --epochs 5 --image-size 128 --batch-size 16 --device cpu
```

运行结果会保存到 `outputs/comparison/summary.json`，训练日志和模型权重会分别保存在 `outputs/baseline/` 与 `outputs/improved/`。

## 单独训练

```powershell
python run_experiment.py --variant baseline --epochs 5 --image-size 128 --batch-size 16
python run_experiment.py --variant improved --epochs 5 --image-size 128 --batch-size 16
```

## 使用 MVTec AD 文件夹

如果已有 MVTec AD 数据，可按类别目录传入：

```powershell
python run_experiment.py --dataset mvtec --data-root D:\datasets\mvtec\bottle --variant improved
```

期望结构：

```text
bottle/
  train/good/*.png
  test/good/*.png
  test/broken_large/*.png
  ground_truth/broken_large/*.png
```

没有真实数据时，默认使用代码内置的 `SyntheticSurfaceDataset` 生成正常纹理、划痕、斑点、凹坑和污染等缺陷，保证作业可直接复现流程。

## 本地说明

当前机器 Python 环境未预装 `torch`，因此我已完成代码与报告，并做了语法级检查。安装依赖后即可运行完整训练。
