# 出海品牌GenAI认知指数分析引擎 (GEO)

**GEO (GenAI Equity Observer)** 是一个强大的、可配置的分析框架，旨在量化和追踪品牌在生成式AI（GenAI）模型认知中的“心智占有率”。

与传统的基于销量或市值的排行榜不同，本项目通过模拟真实用户的提问，自动化地调用大语言模型（LLM），并对返回结果进行多维度计分，从而构建一个反映品牌在AI知识库中影响力的动态排行榜。

---

## 核心理念

在AI时代，当用户越来越多地通过AI获取信息和购买建议时，一个品牌能否被AI频繁、优先、正面地提及，将直接影响其市场地位。本项目旨在衡量这一全新的、无形的品牌资产——**GenAI心智占有率**。

---

## 项目亮点

- **🚀 自动化工作流**: 从问题生成、数据采集到分析报告，实现端到端的自动化。
- **🧩 可配置分析**: 通过独立的 `config.yaml` 文件，轻松定义分析目标、品牌词典和计分权重，无需修改代码即可复用于任何行业。
- **🤖 智能品牌发现**: 内置独立的探索脚本，可利用LLM从海量文本中智能发现新的、未知的候选品牌，极大提升词典构建效率。
- **📊 多维度计分**: 采用先进的六维度计分模型（v8.0 绝对权重版），从**可见度、引用率、推荐度、引用深度、认知份额、竞争力**等角度全面评估品牌表现。
- **📋 详尽报告**: 自动生成包含**总榜单**和**各子品类分榜单**的Markdown报告，洞察品牌在不同领域的具体表现。
- **🌐 模型兼容性**: 基于OpenRouter平台，可灵活切换使用包括Google Gemini, OpenAI GPT, Anthropic Claude在内的多种先进大语言模型。

---

## 项目架构与工作流

本项目遵循一个清晰、严谨的四步闭环工作流：

![工作流示意图](https://i.imgur.com/your-workflow-diagram.png )  <!-- 建议您后续可以创建一个流程图替换此链接 -->

1.  **数据采集 (`run_analysis.py`)**:
    *   **输入**: `questions.json` (预设的问题列表)。
    *   **过程**: 自动化调用LLM API，对每个问题进行提问。
    *   **输出**: `results.json` (包含AI原始回答和引用链接的数据库)。

2.  **品牌探索 (`explore_brands_pure.py`)**:
    *   **输入**: `results.json`。
    *   **过程**: 调用高性价比LLM（如 `google/gemini-2.5-flash`），从原始回答中智能提取所有可能的候选品牌。
    *   **输出**: `candidate_brands_pure_llm.txt` (一个干净的、按频率排序的候选品牌列表)。

3.  **人工决策 (`config_*.yaml`)**:
    *   **输入**: `candidate_brands_pure_llm.txt`。
    *   **过程**: **(关键人工环节)** 基于候选列表，创建或完善YAML配置文件。在此文件中，您需要：
        *   构建 `brand_dictionary` (品牌词典及其所有别名)。
        *   定义 `chinese_brands_whitelist` (哪些品牌需要被计分)。
    *   **输出**: 一个完整的配置文件，如 `config_home_appliances.yaml`。

4.  **分析与报告 (`analyze_results_final.py`)**:
    *   **输入**: `results.json` 和 `config_home_appliances.yaml`。
    *   **过程**: 加载配置，严格按照品牌词典和计分模型，对原始数据进行分组、计分和排名。
    *   **输出**: `ranking_report_home_appliances.md` (最终的、包含总榜单和子榜单的分析报告)。

---

## 如何使用

### 环境准备

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/Charlotte3048/geo.git
    cd geo
    ```

2.  **创建并激活Python虚拟环境**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**:
    ```bash
    pip install openai httpx pyyaml
    ```

4.  **设置API密钥**:
    本项目通过 [OpenRouter.ai](https://openrouter.ai/ ) 调用LLM。请在您的终端中设置API密钥：
    ```bash
    export OPENROUTER_API_KEY="sk-or-v1-..."
    ```

### 执行分析 (以家用电器品类为例)

1.  **准备问题列表**:
    确保 `questions.json` 文件存在，并包含您想要提问的问题。

2.  **第一步：采集数据**
    运行数据采集脚本。这将遍历所有问题并生成 `results.json`。
    ```bash
    python run_analysis.py
    ```

3.  **第二步：探索品牌**
    运行智能品牌探索脚本，生成候选品牌列表。
    ```bash
    python explore_brands_pure.py
    ```

4.  **第三步：创建配置文件**
    这是唯一需要手动编辑的步骤。
    - 打开 `candidate_brands_pure_llm.txt`。
    - 创建一个名为 `config_home_appliances.yaml` 的新文件。
    - 参考候选列表，仔细填写 `brand_dictionary` 和 `chinese_brands_whitelist`。

5.  **第四步：生成最终报告**
    运行最终的分析引擎，传入您刚刚创建的配置文件。
    ```bash
    python analyze_results_final.py --config config_home_appliances.yaml
    ```

    完成后，您将在项目根目录下找到最终的分析报告 `ranking_report_home_appliances.md`。

---

## 未来展望

- [ ] **前端可视化**: 开发一个基于Web的交互式仪表盘，以更直观的方式展示和筛选排行榜数据。
- [ ] **数据库集成**: 将 `results.json` 替换为SQLite或更专业级的数据库，以支持更大规模、更长时间的数据存储和趋势分析。
- [ ] **自动化调度**: 结合GitHub Actions或云函数，实现每日/每周的自动化分析，持续追踪品牌AI心智占有率的变化。
- [ ] **情感分析**: 在计分模型中加入情感分析维度，评估AI在提及品牌时的“语气”是积极、中立还是消极。

