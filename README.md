# 8 通道 EEG + tES 建模工具

面向 8 通道 **EEG + tES** 设备的电极配置与头部电场建模界面。  
EEG 采集与电刺激共用同一组电极。界面用 PySide6 完成 10-10 电极分配和 tES 角色勾选，再调用本机 [SimNIBS](https://simnibs.github.io/simnibs/) 做有限元仿真。

## 功能

- 为硬件 **Ch1–Ch8** 指定 EEG 10-10 位置（不可重复）
- 用复选框选择刺激 / 回流通道，头图只着色显示，不在图上点选
- 设置总电流、电极形状与尺寸、头模型、输出场量
- 一键调用 SimNIBS Python 开始建模，日志显示在窗口底部

## 通道约定

EEG 与 tES 共用 8 个电极：

| 硬件通道 | tES 角色 | 说明 |
|---|---|---|
| Ch1、Ch8 | 回流（source） | 可选 1 个或 2 个都选 |
| Ch2–Ch7 | 刺激（sink） | 至少选 1 个 |

未勾选的通道仍作为 EEG 电极，但不参与刺激。

默认 10-10 位置：

| 通道 | 位置 | 默认 tES |
|---|---|---|
| Ch1 | POz | 回流 |
| Ch2 | P1 | 刺激 |
| Ch3 | P2 | 刺激 |
| Ch4 | O2 | 刺激 |
| Ch5 | O1 | 刺激 |
| Ch6 | Fp1 | 仅 EEG |
| Ch7 | Fp2 | 仅 EEG |
| Ch8 | Cz | 仅 EEG |

电流分配：

- 总电流在已勾选的 **刺激电极** 之间均分（正电流）
- 等量回流在已勾选的 **回流电极** 之间均分（负电流）
- 电流代数和为 0

头图颜色：红 = 刺激，蓝 = 回流，青绿 = 仅 EEG，角标数字 = 硬件通道号。

## 环境要求

本项目使用 **两套 Python**，不要混用：

1. **界面环境**（项目 `.venv`）  
   需要：Python 3.10+、PySide6、MNE、NumPy
2. **SimNIBS 环境**（SimNIBS 安装目录下的 `simnibs_env`）  
   界面通过该解释器运行 `simnibs_run_tdcs.py`  
   常见路径：`C:\Users\<用户名>\SimNIBS-4.x\simnibs_env\python.exe`

还需要 SimNIBS 示例头模型（如 `m2m_ernie`），默认工作目录为 `simnibs4_examples/`。头模型体积较大，未纳入 Git。可从 SimNIBS 安装包或官方 example dataset 获取后放到该目录。

## 安装与启动

在项目根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tdcs_app.py
```

启动后请确认：

- **工作目录** 指向含 `m2m_ernie` 的文件夹（默认为 `simnibs4_examples`）
- **SimNIBS Python** 指向 `simnibs_env\python.exe`（界面会尝试自动查找）

## 使用步骤

1. 为 Ch1–Ch8 选择 10-10 电极。若选了已被占用的位置，会与原通道自动对调。
2. 勾选 Ch1/Ch8 作为回流，勾选 Ch2–Ch7 作为刺激。
3. 设置总电流、电极直径/厚度等参数。
4. 确认头模型与输出文件夹。
5. 点击 **开始建模**。建模过程中界面不会卡住，可随时 **停止**。
6. 完成后可用 **打开结果** 查看输出目录；若勾选了 Gmsh，会自动打开结果。

## 主要文件

| 文件 | 说明 |
|---|---|
| `tdcs_app.py` | 主界面 |
| `simnibs_run_tdcs.py` | 由 SimNIBS Python 执行的建模脚本 |
| `logo.jpg` | 界面顶栏公司 logo |
| `simnibs4_examples/` | 工作目录：头模型、示例脚本、建模输出 |
| `eeg_montage_selector_demo.py` | 早期 matplotlib 电极选择原型 |
| `simnibs4_examples/tdcs_demo*.py` | SimNIBS 官方风格的示例脚本 |

点击「开始建模」时，界面会把当前配置写入：

`simnibs4_examples/_tdcs_gui_last_config.json`

然后执行：

```text
"<SimNIBS python.exe>" simnibs_run_tdcs.py _tdcs_gui_last_config.json
```

## 输出

结果保存在工作目录下你指定的输出文件夹中（例如 `tdcs_P1_P2_O1_O2_1return`）。  
默认输出场量 `veEjJ`：电位 `v`、电场强度 `e`、电场矢量 `E`、电流密度 `j`、电流密度矢量 `J`。

## 许可与数据

`simnibs4_examples` 中的 Ernie / MNI152 等头模型来自 SimNIBS example dataset，许可为 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。商业使用请自行确认 SimNIBS 与数据集条款。
