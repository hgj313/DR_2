# PRD & 原型图审查系统设计

**日期：** 2026-04-30
**版本：** v1.0
**状态：** 待实现

---

## 1. 概述

构建一个智能的 PRD 和原型图审查 Agentic 系统，基于 LangGraph + LangChain Agent，调用 GLM-4.6V-Flash 模型，对 PRD 文档和原型图进行审查，并对照设计标准进行合规性检查。

### 核心能力

1. **PRD 文档审查** - 检查需求文档的完整性、一致性、术语规范性
2. **原型图审查** - 检查原型图与 PRD 描述的一致性
3. **设计标准合规审查** - 验证 PRD/原型是否符合已存储的设计标准

### 输入方式

- 文件上传：PRD 文档（Markdown/Word）+ 原型图截图（PNG/JPG）

---

## 2. 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | >=3.12 | 运行时 |
| uv | latest | 包管理器 |
| langchain-core | 最新版 | Agent 核心抽象 |
| langgraph | 最新版 | Agent 编排 |
| langchain-community | 最新版 | 工具集成 |
| chromadb | 0.5.0+ | 向量数据库 |
| zai-sdk / zhipuai | 最新版 | GLM-4.6V-Flash API (智谱AI) |
| openai | 最新版 | MiniMax API (备选模型) |

### 可用模型

| 模型 | API | 用途 |
|------|-----|------|
| GLM-4.6V-Flash | `https://open.bigmodel.cn/api/paas/v4` | 视觉理解（原型图分析） |
| MiniMax-M2.7 | `https://api.minimaxi.com/v1` | Agent 主体、文本处理、报告生成 |

**双模型分工策略：**
- **MiniMax-M2.7** → Agent 主体（编排流程、文本分析、报告生成、复杂任务编排）
- **GLM-4.6V-Flash** → 视觉工具（`analyze_prototype` 专用，分析原型图）

**API Keys:**
- GLM: `2f84b8c9874e45c295e110c43952afce.MmcJgxFdZE5Haqml`
- MiniMax: `sk-cp-HbqEw_in0Ak-rs-fsLEvknbmp6mHtZUPx8O2KEG5tBnwUrIMSRIQzAt88vpowSXBholobeKgTuK_dyRGQMaou5hi02wgaKx0yaw3fs1qcHvxAgHEvCp1Hlk`

---

## 3. 系统架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      ReviewAgent                            │
│              (LangGraph + GLM-4.6V-Flash)                   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   System Prompt                      │   │
│   │   - 角色定义                                         │   │
│   │   - 工具使用指南                                     │   │
│   │   - 输出格式规范                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    Tools                              │   │
│   │   ├── retrieve_standards()  → ChromaDB              │   │
│   │   ├── retrieve_prototypes() → ChromaDB              │   │
│   │   ├── analyze_prd()         → GLM-4.6V             │   │
│   │   ├── analyze_prototype()   → GLM-4.6V (视觉)     │   │
│   │   └── generate_report()     → 结构化报告            │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │        ChromaDB               │
         │  ┌─────────────────────────┐ │
         │  │  design_reference        │ │ ← 已存储的设计标准
         │  │  (已有 collection)       │ │
         │  └─────────────────────────┘ │
         │  ┌─────────────────────────┐ │
         │  │  prototype_descriptions  │ │ ← 新建：原型图描述
         │  │  (新建 collection)       │ │
         │  └─────────────────────────┘ │
         └───────────────────────────────┘
```

### 3.2 核心流程

```
用户上传 PRD + 原型图
        ↓
[retrieve_standards] → 检索相关设计标准
        ↓
[analyze_prd] → LLM 分析 PRD 内容
        ↓
[analyze_prototype] → GLM-4.6V 视觉分析原型图
        ↓
[retrieve_prototypes] → 检索历史原型描述（若有）
        ↓
[generate_report] → 生成审查报告
        ↓
存储原型描述到 prototype_descriptions collection
        ↓
输出：结构化审查报告
```

---

## 4. 上下文管理策略

**方案：RAG 检索 + 按需获取**

### 4.1 设计标准检索

- `design_reference` collection 已存有设计标准
- 检索时使用向量相似度召回最相关片段
- 仅将相关片段注入上下文，不全量塞入

### 4.2 PRD 处理

- 首次分析时建立整体理解
- 后续按需检索相关章节/段落
- 关键信息存入上下文变量

### 4.3 原型图处理

- 首次分析时调用 `analyze_prototype`
- 生成结构化描述存入 `prototype_descriptions` collection
- 后续审查时通过 `retrieve_prototypes` 检索历史描述

### 4.4 对话历史

- 保留最近 N 轮完整对话
- 超过阈值时自动摘要或截断

---

## 5. Agent 实现

### 5.1 创建方式

使用 LangChain Agent 抽象层，而非简单单次 LLM 调用：

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=chat_model,
    tools=tools,
    state_modifier=system_prompt
)
executor = AgentExecutor(agent=agent, tools=tools)
```

### 5.2 Tools 定义

| Tool | 输入 | 输出 | ChromaDB 操作 |
|------|------|------|---------------|
| `retrieve_standards` | 查询文本 | 相关标准片段列表 | similarity_search |
| `retrieve_prototypes` | 查询文本 | 历史原型描述列表 | similarity_search |
| `analyze_prd` | PRD 文本 | 结构化分析结果 | 无 |
| `analyze_prototype` | 图片文件 | 结构化描述 | insert |
| `generate_report` | 审查结果 | Markdown 报告 | 无 |

### 5.3 System Prompt

定义 Agent 角色和输出格式：

```
你是一个专业的 PRD 和原型图审查专家。
用户将上传 PRD 文档和原型图，你需要：
1. 分析 PRD 的完整性和一致性
2. 分析原型图的布局和元素
3. 对照设计标准检查合规性
4. 输出结构化的审查报告

输出格式：
## 审查结果

### PRD 分析
...

### 原型图分析
...

### 合规性检查
...

### 问题列表
1. ...
2. ...

### 建议
...
```

---

## 6. 数据模型

### 6.1 prototype_descriptions Collection

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 唯一标识 (UUID)，内部使用 |
| name | str | 用户可读名称，如"登录页"、"主页导航" |
| document_id | str | 关联的 PRD 文档 ID |
| page_name | str | 页面名称 |
| layout | str | 布局描述 |
| components | list[str] | 组件列表 |
| interactions | str | 交互描述 |
| query_text | str | 用于检索的文本（聚合 name + page_name + layout） |
| embedding | list[float] | 向量 (1024维 BGE-M3) |

**防重复策略：**
- 基于 `document_id + page_name` 作为唯一键
- 存储前检查是否已存在，存在则更新而非插入

**用户查询方式：**
- 通过 `name` 字段（用户可读名称）进行语义检索
- 通过 `query_text` 进行向量相似度检索

### 6.2 ChromaDB 配置

```python
ChromaStore(
    collection_name="prototype_descriptions",
    persist_directory="./chroma_data",
    use_remote=False
)
```

---

## 7. 文件结构

```
dr-2/
├── agent/
│   ├── __init__.py
│   ├── review_agent.py      # Agent 创建和执行
│   ├── tools.py             # Tool 定义
│   ├── prompts.py           # System prompts
│   └── schemas.py           # 数据模型
├── vec_db/
│   └── store/
│       └── prototype_store.py  # 原型描述存储
├── pyproject.toml
└── main.py
```

---

## 8. 实现计划

### Phase 1: 基础架构
- [ ] 安装依赖 (langchain, langgraph, langchain-community, zai-sdk)
- [ ] 创建 `prototype_descriptions` collection
- [ ] 实现 `ReviewAgent` 基础结构

### Phase 2: 工具实现
- [ ] 实现 `retrieve_standards` tool
- [ ] 实现 `retrieve_prototypes` tool
- [ ] 实现 `analyze_prd` tool
- [ ] 实现 `analyze_prototype` tool (视觉)
- [ ] 实现 `generate_report` tool

### Phase 3: Agent 集成
- [ ] 创建 LangGraph ReAct Agent
- [ ] 绑定 Tools 到 Agent
- [ ] 配置 System Prompt
- [ ] 测试 Agent 执行流程

### Phase 4: 端到端测试
- [ ] 上传 PRD 文档测试
- [ ] 上传原型图测试
- [ ] 生成审查报告测试

---

## 9. 待确认事项

- [ ] 是否需要支持流式输出？
- [ ] 审查报告的输出格式（Markdown/JSON/PDF）？
- [ ] 是否需要持久化对话历史？

---

*设计完成，等待实现*