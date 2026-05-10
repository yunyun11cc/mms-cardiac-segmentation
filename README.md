# M&Ms Cardiac MRI Segmentation

**用数据增强方法缓解跨厂商心脏 MRI 分割中的 domain shift**

---

## 实验进度

| 实验 | 内容 | 状态 |
|------|------|------|
| Exp001 | 数据读取与初步分析 | ✅ 完成 |
| Exp002 | MONAI 预处理 Pipeline | ✅ 完成 |
| Exp003 | U-Net Baseline 训练 | 🏗️ 进行中 |
| Exp004 | Leave-One-Vendor-Out 实验 | 📋 计划中 |
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

- 模型：MONAI BasicUNet（features: 32,32,64,128,256,512），4 分类 Dice Loss
- 训练：120 例 / 验证：30 例（按 vendor 分层采样）
- 支持：CUDA（AMP 混合精度）/ MPS / CPU

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
│   └── exp003_train_unet.py   # Exp003：U-Net 训练
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
```

---

## 后续计划

### Exp004 — Leave-One-Vendor-Out

| Split | 训练 | 测试 |
|-------|------|------|
| 1 | Siemens | Philips |
| 2 | Philips | Siemens |
| 3 | Siemens + Philips | GE + Canon |

### Exp005 — 数据增强

待选：强度增强 → Histogram Matching → 风格扰动
