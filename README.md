# M&Ms Cardiac MRI Segmentation

**用数据增强方法缓解跨厂商心脏 MRI 分割中的 domain shift**

---

## 实验进度

| 实验 | 内容 | 状态 |
|------|------|------|
| Exp001 | 数据读取与初步分析 | ✅ 完成 |
| Exp002 | MONAI 预处理 Pipeline | ✅ 完成 |
| Exp003 | U-Net Baseline 训练 | ✅ 完成 |
| Exp004 | Leave-One-Vendor-Out 实验 | ✅ 完成 |
| Exp005 | 数据增强缓解方法（强度增强） | 🏗️ 进行中 |

## 运行

```bash
# Exp004: Vendor Shift 实验
python scripts/exp004_vendor_shift.py

# Exp005: 强度增强实验
python scripts/exp005_augmentation.py --skip-baseline --epochs 6
```

---

### Exp001 — 数据读取与分析

- 读取全部 150 例 Training/Labeled NIfTI + CSV 元数据
- **关键发现**：Training 只有 Siemens（75）和 Philips（75），GE 和 Canon 仅出现在 Validation/Testing
- Siemens 矩形图像，Philips 正方形图像 —— domain shift 真实存在
- Vendor 和 Centre 完全耦合

### Exp002 — 预处理 Pipeline

```
EnsureChannelFirst → Spacing(1.5mm isotropic) → NormalizeIntensity(z-score)
                  → CropForeground → SpatialPad(256×256)
```

### Exp003 — U-Net Baseline

- 模型：MONAI BasicUNet（features: 32,32,64,128,256,512），4 分类 Dice Loss
- 训练：120 例 / 验证：30 例（按 vendor 分层采样），10 epochs
- 支持：CUDA / MPS / CPU
- **结果：** train_loss=0.10, val_loss=0.20（第 4 轮开始过拟合），三种结构均能识别
- **结论：** baseline 已建立，过拟合表明后续需数据增强

### Exp004 — Vendor Shift 实验

| Split | 训练 | 测试 | LV | RV | MYO | Avg Dice |
|---|---|---|---:|---:|---:|---:|
| 1 | Siemens | Philips | 0.7646 | 0.6514 | 0.5467 | 0.6542 |
| 2 | Philips | Siemens | 0.7286 | 0.4800 | 0.5253 | 0.5780 |
| 3 | Siemens+Philips | GE+Canon | 0.6896 | 0.5069 | 0.5268 | 0.5744 |
| Mixed | Siemens+Philips | Siemens+Philips | 0.8717 | 0.7986 | 0.7240 | 0.7981 |

**结论：** Split 3 Avg Dice=0.5744，明显低于 Mixed Avg Dice=0.7981，跨未见厂商下降约 0.224，证明跨厂商 domain shift 确实导致性能下降。

### Exp005 — 强度增强缓解方法

- 方法：训练阶段对非零 MRI 像素做随机强度缩放、偏移和 Gaussian noise
- 基准：Exp004 Split 3（Train Siemens+Philips → Test GE+Canon）
- 输出：`outputs/exp005/exp005_intensity_dice_table.txt`、`outputs/exp005/exp005_intensity_dice_barplot.png`

---

## 项目结构

```
M&Ms/
├── dataset/                 # 数据集（不纳入版本控制）
│   ├── Training/Labeled/    # 150 例（Siemens 75 + Philips 75）
│   ├── Validation/          # 34 例
│   └── Testing/             # 136 例
├── scripts/
│   ├── mnms_first_step.py     # Exp001：数据读取
│   ├── mnms_analyze_all.py    # Exp001：统计分析
│   ├── exp002_preprocessing.py # Exp002：预处理验证
│   ├── exp003_train_unet.py   # Exp003：U-Net 训练
│   ├── exp004_vendor_shift.py # Exp004：Vendor Shift 实验
│   └── exp005_augmentation.py # Exp005：强度增强实验
├── tests/
│   └── test_exp005_augmentation.py
├── outputs/                 # 训练输出（不纳入版本控制）
├── requirements.txt
└── pyproject.toml
```

---

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 或使用 uv
uv sync
```

---

## 运行

```bash
# Exp003: U-Net Baseline 训练
python scripts/exp003_train_unet.py

# Exp004: Vendor Shift 实验
python scripts/exp004_vendor_shift.py

# Exp005: 强度增强实验
python scripts/exp005_augmentation.py --skip-baseline --epochs 6
```

---

## 后续计划

### Exp005 — 数据增强

当前实现：强度增强。后续可继续尝试 Histogram Matching、风格扰动 / FDA。
