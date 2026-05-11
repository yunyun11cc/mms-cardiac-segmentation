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
| Exp005 | 数据增强缓解方法 | 📋 计划中 |

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

- 模型：MONAI BasicUNet（features: 32,32,64,128,256,512），4.6M 参数
- 训练/验证：120/30 例，按 vendor 分层采样，10 epoch，AMP 混合精度
- **结果**：train_loss → 0.1，val_loss → 0.2（Dice ≈ 0.80），epoch 4 后开始过拟合

### Exp004 — Leave-One-Vendor-Out

- 设计：三个 cross-vendor split + 一个同域对照
  - Split 1: 训练 Siemens → 测试 Philips
  - Split 2: 训练 Philips → 测试 Siemens
  - Split 3: 训练 Siemens+Philips → 测试 GE+Canon
  - Mixed: Siemens+Philips 80/20 同域对照
- 输出：per-class Dice 表 + 柱状图
- **结论**：baseline U-Net 跨厂商泛化能力有限，印证 domain shift 问题是后续增强方法要攻克的核心

---

## 项目结构

```
M&Ms/
├── dataset/                      # 数据集（不纳入版本控制）
│   ├── Training/Labeled/         # 150 例（Siemens 75 + Philips 75）
│   ├── Validation/               # 34 例（含 GE、Canon）
│   └── Testing/                  # 136 例（含 GE、Canon）
├── scripts/
│   ├── mnms_first_step.py        # Exp001：数据读取与可视化
│   ├── mnms_analyze_all.py       # Exp001：Vendor 统计分析
│   ├── exp002_preprocessing.py   # Exp002：预处理 Pipeline 验证
│   ├── exp003_train_unet.py      # Exp003：U-Net Baseline 训练
│   └── exp004_vendor_shift.py    # Exp004：Leave-One-Vendor-Out 实验
├── outputs/                      # 训练输出（不纳入版本控制）
│   ├── exp001/                   # 数据探索可视化
│   ├── exp002/                   # 预处理对比图
│   ├── exp003/                   # 训练曲线 + 预测图 + 模型权重
│   └── exp004/                   # Dice 结果表 + 柱状图
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
```

---

## 运行

```bash
python scripts/exp003_train_unet.py    # U-Net Baseline 训练
python scripts/exp004_vendor_shift.py  # Leave-One-Vendor-Out 实验
```

---

## 后续计划

### Exp005 — 数据增强缓解 Domain Shift

待选方法：强度增强（亮度/对比度/噪声）→ Histogram Matching → 风格扰动 / FDA
