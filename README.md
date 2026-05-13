# M&Ms Cardiac MRI Segmentation

**围绕 M&Ms 心脏 MRI 分割任务，分析跨厂商 domain shift，并用分割 baseline 与数据增强方法做初步验证。**

---

## 当前研究线

| 方向 | 内容 | 状态 |
|------|------|------|
| U-Net | 手写 MONAI 2D U-Net baseline、跨厂商实验、强度增强 | ✅ 初步完成 |
| nnU-Net | 用强 baseline 重启一轮，重新建立更规范的分割基线 | 🏗️ 准备中 |

现有手写 U-Net 实验都已整理到根目录 `unet/`；下一轮 nnU-Net 相关代码会放到根目录 `nnunet/`。

---

## U-Net 实验进度

| 实验 | 内容 | 状态 |
|------|------|------|
| Exp001 | 数据读取与初步分析 | ✅ 完成 |
| Exp002 | MONAI 预处理 Pipeline | ✅ 完成 |
| Exp003 | 2D U-Net Baseline 训练 | ✅ 完成 |
| Exp004 | Leave-One-Vendor-Out 跨厂商实验 | ✅ 完成 |
| Exp005 | 数据增强缓解方法（强度增强） | ✅ 初步完成 |

### Exp001 — 数据读取与分析

- 读取 Training/Labeled NIfTI + CSV 元数据
- 确认标签值：`0=背景, 1=LV, 2=MYO, 3=RV`
- **关键发现**：Training 主要用于 Siemens 和 Philips，GE 和 Canon 可作为 unseen vendor
- Siemens 与 Philips 图像尺寸差异明显，domain shift 真实存在
- Vendor 和 Centre 高度耦合

### Exp002 — 预处理 Pipeline

```
Load NIfTI → 选有标注 frame → EnsureChannelFirst
           → Spacing(1.5mm) → NormalizeIntensity(z-score)
           → CropForeground → SpatialPad → 提取有标签 2D slice
```

### Exp003 — U-Net Baseline

- 模型：MONAI BasicUNet，4 分类 Dice Loss
- 输入：预处理后的 2D short-axis slice
- 输出：背景 + LV + MYO + RV
- 作用：跑通数据读取、预处理、训练、预测可视化的完整分割链路
- 观察：baseline 能识别主要结构，但有过拟合趋势

### Exp004 — Vendor Shift 实验

| Split | 训练 | 测试 | LV | RV* | MYO* | Avg Dice |
|---|---|---|---:|---:|---:|---:|
| 1 | Siemens | Philips | 0.7646 | 0.6514 | 0.5467 | 0.6542 |
| 2 | Philips | Siemens | 0.7286 | 0.4800 | 0.5253 | 0.5780 |
| 3 | Siemens+Philips | GE+Canon | 0.6896 | 0.5069 | 0.5268 | 0.5744 |
| Mixed | Siemens+Philips | Siemens+Philips | 0.8717 | 0.7986 | 0.7240 | 0.7981 |

**结论：** Split 3 Avg Dice=0.5744，明显低于 Mixed Avg Dice=0.7981，跨未见厂商下降约 0.224，证明跨厂商 domain shift 确实导致性能下降。

> 注：早期脚本的单类表头沿用了 `LV/RV/MYO`，而 M&Ms/ACDC 标签语义通常是 `1=LV, 2=MYO, 3=RV`。Avg Dice 不受影响，后续 nnU-Net 阶段会统一类别命名。

### Exp005 — 强度增强

| 方法 | LV | RV* | MYO* | Avg Dice |
|---|---:|---:|---:|---:|
| Split3 baseline | 0.7194 | 0.4953 | 0.5459 | 0.5869 |
| Split3 + intensity aug | 0.6785 | 0.4190 | 0.5432 | 0.5469 |

**初步结论：** 当前简单强度增强没有提升 unseen vendor 表现，Avg Dice 反而下降 0.0400。说明简单随机灰度扰动不足以模拟 GE/Canon 的真实域差异，后续需要更强 baseline 或更有针对性的域泛化方法。

---

## nnU-Net 实验进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| nnU-Net-001 | 准备目录结构与实验入口 | ✅ 已创建 |
| nnU-Net-002 | 转换 M&Ms 到 nnU-Net raw dataset 格式 | 📋 待做 |
| nnU-Net-003 | 训练 nnU-Net baseline | 📋 待做 |
| nnU-Net-004 | 复现跨厂商评估 | 📋 待做 |
| nnU-Net-005 | 与手写 U-Net 结果对比 | 📋 待做 |

下一轮目标是用 nnU-Net 重新建立更强、更规范的分割 baseline，再在此基础上继续分析跨厂商泛化问题。

---

## 运行方式

### U-Net 线

```bash
# Exp003: U-Net Baseline 训练
python unet/exp003_train_unet.py

# Exp004: Vendor Shift 实验
python unet/exp004_vendor_shift.py

# Exp005: 强度增强实验
python unet/exp005_augmentation.py
```

### nnU-Net 线

```bash
# 下一轮将在 nnunet/ 中补充数据转换、训练、评估脚本
```

---

## 项目结构

```
M&Ms/
├── dataset/                    # 数据集（不纳入版本控制）
│   ├── Training/Labeled/
│   ├── Validation/
│   └── Testing/
├── unet/                       # 手写 MONAI U-Net 实验线
│   ├── mnms_first_step.py
│   ├── mnms_analyze_all.py
│   ├── exp002_preprocessing.py
│   ├── exp003_train_unet.py
│   ├── exp004_vendor_shift.py
│   └── exp005_augmentation.py
├── nnunet/                     # 下一轮 nnU-Net 实验线
│   └── README.md
├── outputs/                    # 本地训练输出（不纳入版本控制）
│   ├── unet/
│   │   ├── exp001/
│   │   ├── exp002/
│   │   ├── exp003/
│   │   ├── exp004/
│   │   └── exp005/
│   └── nnunet/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 或使用 uv
uv sync
```

---

## 后续计划

1. 用 nnU-Net 重启一轮 baseline，先获得更可靠的分割性能上限。
2. 将 M&Ms 数据整理成 nnU-Net 所需的 raw dataset 格式。
3. 在 nnU-Net 上复现 Siemens/Philips 到 GE/Canon 的跨厂商评估。
4. 再决定后续是继续做强度/风格增强，还是转向不确定性、弱监督、标注效率等问题。
