# nnU-Net 实验线

用 nnU-Net v2 重建更强、更规范的分割 baseline，并在同一批 M&Ms 数据上复现跨厂商评估。

---

## 实验阶段

| 阶段 | 做什么 | 为什么 |
|---|---|---|
| nnU-Net-001 | M&Ms → nnU-Net raw 格式转换 | 把数据弄进 nnU-Net 能吃的格式 |
| nnU-Net-002 | 混合域训练 + 内部验证 | 拿到"这个任务在 nnU-Net 下能做到多好" |
| nnU-Net-003 | Leave-One-Vendor-Out | 复现 U-Net 的核心结论 |
| nnU-Net-004 | U-Net vs nnU-Net 对比分析 | 量化差距，决定后续方法方向 |
| nnU-Net-005 | （条件）数据增强 | 如果 nnU-Net 跨厂商仍有 gap，在这里尝试 |

---

### nnU-Net-001：数据格式转换

**目标：** 把 M&Ms 数据转成 nnU-Net v2 能直接读的 raw 格式。

**nnU-Net 要求的目录结构：**
```
nnUNet_raw/Dataset301_MMs/
├── dataset.json
├── imagesTr/          # 训练图像
├── labelsTr/          # 训练标签
├── imagesTs/          # 测试图像（unseen vendor）
└── labelsTs/          # 测试标签（用于评估）
```

**要做的：**
1. 每个病例选 ED 帧，存为 2D `.nii.gz`
2. 标签值保持 `0=背景, 1=LV, 2=MYO, 3=RV`
3. 写 `dataset.json`：指定类别名、模态、训练/测试划分
4. 以 Split 3 配置为例（imagesTr=S+P, imagesTs=GE+Canon），其他 split 可通过改 json 或符号链接切换

**验收标准：**
- `nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity` 不报错

---

### nnU-Net-002：混合域训练

**目标：** 拿到 nnU-Net 在 M&Ms 上的"天花板"，作为跨厂商实验的参照。

**操作：**
```bash
nnUNetv2_plan_and_preprocess -d 301
nnUNetv2_train 301 2d 0
```

**要记录：**
- 五折交叉验证 Dice
- 和 U-Net Mixed 域 (0.80) 的差距

---

### nnU-Net-003：Leave-One-Vendor-Out

**目标：** 用 nnU-Net 复现 U-Net Exp004 的所有 split。

**Split 设计（和 U-Net Exp004 完全对齐）：**

| Split | 训练 | 测试 | 对应 nnU-Net 数据集 ID |
|---|---|---|---|
| 1 | Siemens | Philips | 302 |
| 2 | Philips | Siemens | 303 |
| 3 | Siemens+Philips | GE+Canon | 304（可复用 001 的数据） |
| Mixed | 全部混合 | 五折 CV | 301 |

> 每个 Split 都是独立的数据集，有各自的 `dataset.json` 和 images/labels 目录。脚本批量生成，不改手工切。

**要记录的：**
- 每个 Split 的 LV/MYO/RV Dice + Avg Dice
- 和 U-Net Exp004 结果对应行对比

---

### nnU-Net-004：对比分析

**核心问题：**
1. nnU-Net 比手写 U-Net 高多少 Mixed Dice？
2. nnU-Net 的跨厂商 gap（Mixed - Unseen）比 U-Net 大还是小？
3. 哪个结构（LV/MYO/RV）在两个 baseline 下都最难跨域？

**输出：** 一张对比表 + 一张对比柱状图。

| Split | U-Net Avg Dice | nnU-Net Avg Dice |
|---|---|---|
| Mixed | 0.80 | ? |
| Siem→Philips | 0.65 | ? |
| Philips→Siem | 0.58 | ? |
| S+P→GE+Canon | 0.57 | ? |

---

### nnU-Net-005（条件触发）：数据增强

**触发条件：** 003 中跨厂商 gap 仍然 > 0.15

**为什么放在 nnU-Net 上做而不是 U-Net 上继续：**
- U-Net Exp005 的强度增强没效果，可能是 baseline 本身太弱
- nnU-Net 自带更好的预处理和数据增强
- 在更强的 baseline 上，增强方法的收益更容易被观察到

**可选方向：**
- nnU-Net 自身的增强策略（mirroring、旋转等）是否已经足够？
- 添加额外的域泛化增强（Histogram Matching、FDA）

---

## 脚本规划

```
nnunet/
├── README.md                          # 这个文件
├── nnunet_001_convert_mnms.py         # M&Ms → nnU-Net raw 格式
├── nnunet_003_lovo.py                 # Leave-One-Vendor-Out 批量转换
└── nnunet_004_compare.py              # U-Net vs nnU-Net 对比图
```

训练和推理用 nnU-Net v2 本身的 CLI，不需要额外脚本。只写数据转换和结果对比的脚本。
