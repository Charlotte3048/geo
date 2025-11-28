# 品牌AI认知排行榜分析引擎 (GEO)

**GEO (GenAI Equity Observer)** 是一个强大的、可配置的分析框架，旨在量化和追踪品牌在生成式AI（GenAI）模型认知中的**品牌AI认知排行榜**。

与传统的基于销量或市值的排行榜不同，本项目通过模拟真实用户的提问，自动化地调用大语言模型（LLM），并对返回结果进行多维度计分，从而构建一个反映品牌在AI知识库中影响力的动态排行榜。

---

## 核心理念

在AI时代，当用户越来越多地通过AI获取信息和购买建议时，一个品牌能否被AI频繁、优先、正面地提及，将直接影响其市场地位。本项目旨在衡量这一全新的、无形的品牌资产——**品牌AI认知**。

---

## 项目亮点

- **🚀 自动化工作流**: 从问题生成、数据采集到分析报告，实现端到端的自动化。
- **🧩 可配置分析**: 通过独立的配置文件，轻松定义分析目标、品牌词典和计分权重，无需修改代码即可复用于任何行业。
- **🤖 智能品牌发现**: 内置独立的探索脚本，可利用LLM从海量文本中智能发现新的、未知的候选品牌，极大提升词典构建效率。
- **📊 六维计分**: 采用先进的**六维计分模型**，从**品牌可见度、引用率、AI认知排行指数、正文引用率、AI认知份额、竞争力指数**等角度全面评估品牌表现。
- **📋 详尽报告**: 自动生成包含**总榜单**和**各子品类分榜单**的Markdown报告。
- **🌐 模型兼容性**: 灵活支持**国际主流LLM**（如GPT、Gemini）和**国内主流LLM**（如文心一言、通义千问、Kimi等）。

---

## 项目架构与双榜单结构

本项目采用**双榜单结构**，将国际市场和国内市场的分析逻辑、数据和配置完全隔离，确保项目的高模块化和可维护性。

| 目录/文件 | 描述 | 目标市场 | 核心模型 |
| :--- | :--- | :--- | :--- |
| `oversea/` | **出海榜单**：包含所有国际市场分析所需的文件。 | 国际市场 | GPT, Gemini, Perplexity (OpenRouter) |
| `domestic/` | **国内榜单**：包含所有国内市场分析所需的文件。 | 国内市场 | 文心一言, Kimi, 智谱, 腾讯混元等 10+ 国内LLM |
| `analyze_results.py` | **核心分析脚本**：复用于两个榜单的最终排名计算。 | 通用 | N/A |
| `merge_results.py` | **结果合并脚本**：用于将多模型结果合并为最终分析文件。 | 通用 | N/A |

### **工作流**

本项目遵循一个清晰、严谨的四步闭环工作流，**每个榜单（`oversea` 和 `domestic`）独立运行**：

1.  **数据采集 (`run_analysis.py` / `domestic/run_analysis_domestic.py`)**:
    *   **输入**: `questions.json` 或 `domestic/questions_domestic.json`。
    *   **过程**: 自动化调用LLM API，对每个问题进行提问。
    *   **输出**: 原始结果 JSON 文件。

2.  **品牌探索 (`explore_brands.py` / `domestic/explore_brands_domestic.py`)**:
    *   **输入**: 合并后的结果文件（如 `results_nev_merged.json`）。
    *   **过程**: 调用LLM从原始回答中智能提取所有可能的候选品牌。
    *   **输出**: 品牌词典模板（如 `brand_dictionary_nev.yaml`）。

3.  **人工决策 (配置文件)**:
    *   **输入**: 品牌词典模板。
    *   **过程**: **(关键人工环节)** 完善品牌词典 (`brand_dictionary`) 和品牌白名单 (`chinese_brands_whitelist`)。
    *   **输出**: 最终配置文件（如 `config_home_appliances.yaml` 或 `domestic/config_nev.yaml`）。

4.  **分析与报告 (`analyze_results.py`)**:
    *   **输入**: 原始数据和最终配置文件。
    *   **过程**: 严格按照六维计分模型，对数据进行分组、计分和排名。
    *   **输出**: 最终的分析报告（如 `ranking_report_home_appliances.md` 或 `domestic/ranking_report_nev.md`）。

---

## 如何使用

### 环境准备

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/Charlotte3048/geo.git
    cd geo
    ```

2.  **安装依赖**:
    ```bash
    pip install openai httpx pyyaml python-dotenv tenacity
    ```

3.  **设置API密钥**:
    创建项目根目录下的 `.env` 文件，并填入所有国际和国内模型的 API Key 和 Base URL。

### 执行分析 (以国内榜单-新能源汽车为例)

1.  **第一步：采集数据**
    运行国内数据采集脚本。
    ```bash
    python domestic/run_analysis_domestic.py
    ```

2.  **第二步：探索品牌**
    运行智能品牌探索脚本，生成候选品牌词典模板。
    ```bash
    python domestic/explore_brands_domestic.py --task nev --results_file results_nev_merged.json
    ```

3.  **第三步：创建配置文件**
    手动编辑生成的 `brand_dictionary_nev.yaml`，并将其内容整合到最终的 `domestic/config_nev.yaml` 中。

4.  **第四步：生成最终报告**
    运行核心分析引擎，传入国内榜单的配置文件。
    ```bash
    python analyze_results.py --config domestic/config_nev.yaml
    ```

    完成后，您将在 `domestic/` 目录下找到最终的分析报告 `ranking_report_nev.md`。

---

## 未来展望

- [ ] **前端可视化**: 开发一个基于Web的交互式仪表盘，以更直观的方式展示和筛选排行榜数据。
- [ ] **数据库集成**: 将 JSON 文件替换为SQLite或更专业级的数据库，以支持更大规模、更长时间的数据存储和趋势分析。
- [ ] **自动化调度**: 结合GitHub Actions或云函数，实现每日/每周的自动化分析，持续追踪品牌AI认知排行榜的变化。
- [ ] **情感分析**: 在计分模型中加入情感分析维度，评估AI在提及品牌时的“语气”是积极、中立还是消极。
