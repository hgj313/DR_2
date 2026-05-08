# PRD & 原型图审查系统 技术文档

## 1. 系统概述

### 1.1 项目简介

**PRD & 原型图审查系统** 是一个基于 AI 的产品需求文档（PRD）和原型图合规性审查工具。系统通过向量数据库检索、视觉分析和多模型协同，自动化执行设计标准合规性检查。

### 1.2 核心功能

- **PRD 文档处理**：支持 Markdown 格式文档的读取、分块（Chunk）和向量化存储
- **原型图分析**：基于 GLM-4.6V-Flash 模型进行视觉理解，提取页面布局、组件和交互元素
- **设计标准检索**：从 ChromaDB 向量数据库检索相关设计标准
- **合规性审查**：对比 PRD、原型图与设计标准的一致性
- **报告生成**：输出结构化的 Markdown 审查报告

### 1.3 技术栈

| 层级 | 技术/库 | 用途 |
|------|---------|------|
| Agent 框架 | LangGraph, LangChain | ReAct 代理实现 |
| 向量数据库 | ChromaDB | 文档和原型描述存储 |
| Embedding | BGE-M3 (Sentence-Transformers) | 文本向量化 |
| LLM (主模型) | MiniMax-M2.7 | 文本理解和生成 |
| LLM (视觉) | GLM-4.6V-Flash | 原型图视觉分析 |
| 数据验证 | Pydantic | 数据模型定义 |

---

## 2. 系统架构

### 2.1 模块结构

```
dr-2/
├── main.py                      # 系统入口，CLI 交互界面
├── agent/                        # Agent 层
│   ├── review_agent.py          # LangGraph 版本的审查 Agent
│   ├── session.py               # 会话管理器
│   ├── tools.py                 # Agent 可用工具集
│   ├── prompts.py               # System Prompt 模板
│   └── schemas.py               # 数据模型定义
├── domain/                       # 领域实体层
│   ├── document.py              # DesignDocument 实体
│   ├── chunk.py                 # Chunk 和 ChunkMetadata
│   └── value_objects.py         # 值对象
├── vec_db/                       # 向量数据库层
│   ├── embeddings/
│   │   ├── bge_m3_embedding.py  # BGE-M3 向量化实现
│   │   └── chunker.py           # 文档分块器
│   └── store/
│       ├── chroma_store.py       # ChromaDB 通用存储
│       └── prototype_store.py    # 原型描述存储
├── models/                       # 模型封装层
│   ├── minimax.py               # MiniMax ChatOpenAI 兼容接口
│   └── glm.py                   # GLM-4.6V 视觉分析接口
└── tests/                        # 测试文件
```

### 2.2 数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              ReviewSessionManager (session.py)                  │
│  - create_session()                                             │
│  - upload_session()                                              │
│  - bind_prototype_to_document()                                 │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────────┐    ┌─────────────────┐
│  Document   │      │   Prototype     │    │    Design      │
│  Register   │      │   Register      │    │    Standards   │
└─────────────┘      └─────────────────┘    └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────────┐    ┌─────────────────┐
│  BgeM3      │      │   BgeM3        │    │   ChromaDB      │
│  Embedding  │      │   Embedding    │    │   Collection    │
└─────────────┘      └─────────────────┘    └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ChromaDB (chroma_data/)                    │
│  Collections: prd_documents, prototype_descriptions,            │
│                design_references                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ReviewAgent (Agent Layer)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Tools:                                                 │  │
│  │  - retrieve_prd_chunks() 检索 PRD 文档块                │  │
│  │  - retrieve_prototypes()  检索原型描述                   │  │
│  │  - retrieve_standards()   检索设计标准                  │  │
│  │  - analyze_prototype()     分析原型图                    │  │
│  │  - generate_report()       生成审查报告                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Models:                                                │  │
│  │  - MiniMaxModel (主对话)                                │  │
│  │  - GLMModel (图像分析)                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 Agent 层

#### 3.1.1 ReviewAgent ([review_agent.py](file:///c:/AA_GSE_DP/DR-2/agent/review_agent.py))

基于 LangGraph 实现的 ReAct 风格审查 Agent。

**主要类**：`ReviewAgent`

**核心方法**：
| 方法 | 签名 | 功能 |
|------|------|------|
| `invoke()` | `(input: str, thread_id: str) -> dict` | 同步调用 Agent |
| `astream()` | `(input: str, thread_id: str)` | 异步流式调用 |
| `stream()` | `(input: str, thread_id: str)` | 同步流式调用，区分输出类型 |

**流式输出类型** (`StreamChunkType`)：
- `THINKING`：AI 推理思考过程
- `TOOL_CALL`：工具调用
- `TOOL_RESULT`：工具返回结果
- `FINAL`：最终回复
- `STATUS`：状态信息

#### 3.1.2 ReviewSessionManager ([session.py](file:///c:/AA_GSE_DP/DR-2/agent/session.py))

审查会话管理器，负责 PRD 和原型图的关联、绑定/解绑，支持 SQLite 持久化。

**核心功能**：
- 创建审查会话
- 上传/注册 PRD 文档（带自动切分）
- 上传/注册原型图（支持绑定 document_id）
- 绑定/解绑 PRD 与原型图的关联
- SQLite 持久化 session 元数据

**初始化参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chroma_persist_dir` | str | "./chroma_data" | 持久化目录（包含 ChromaDB 和 SQLite） |

**持久化实现**：
- 数据库路径：`{chroma_persist_dir}/sessions.db`
- 启动时自动调用 `_load_sessions()` 从数据库加载所有会话
- 每次会话变更（创建、更新、绑定/解绑）自动调用 `_save_session()` 和 `_save_prototypes()`

**核心方法**：
| 方法 | 功能 |
|------|------|
| `create_session()` | 创建新会话，返回 session_id |
| `register_prd()` | 注册 PRD 文档（自动切分 + 存储到 ChromaDB） |
| `register_prototype()` | 注册原型图（自动调用 GLM 分析） |
| `bind_prototype_to_document()` | 绑定原型图到 PRD |
| `unbind_prototype()` | 解除绑定 |
| `upload_session()` | 同时上传 PRD 和原型图，自动建立关联 |

**私有方法**：
| 方法 | 功能 |
|------|------|
| `_init_db()` | 初始化 SQLite 数据库表结构 |
| `_load_sessions()` | 从数据库加载所有会话 |
| `_save_session()` | 保存单个会话到数据库 |
| `_save_prototypes()` | 保存会话关联的原型图列表到数据库 |
| `_parse_prototype_analysis()` | 解析 GLM 返回的原型图分析结果 |

#### 3.1.4 Agent Tools ([tools.py](file:///c:/AA_GSE_DP/DR-2/agent/tools.py))

Agent 可调用工具集：

| 工具 | 功能 |
|------|------|
| `retrieve_prd_chunks()` | 从 PRD 文档库检索相关内容片段 |
| `retrieve_prototypes()` | 从原型描述库检索相关内容 |
| `retrieve_standards()` | 从设计标准知识库检索相关内容 |
| `analyze_prototype()` | 分析原型图（需要图片路径） |
| `generate_report()` | 生成结构化审查报告 |

### 3.2 领域层

#### 3.2.1 数据模型 ([schemas.py](file:///c:/AA_GSE_DP/DR-2/agent/schemas.py))

```python
# 核心数据模型
PrototypeDescription  # 原型图描述
DocumentInfo          # PRD 文档信息
ReviewSession         # 审查会话
ReviewResult          # 审查结果
ReviewReport          # 审查报告
```

#### 3.2.2 领域实体 ([domain/](file:///c:/AA_GSE_DP/DR-2/domain/))

| 文件 | 类 | 说明 |
|------|-----|------|
| `document.py` | `DesignDocument` | 设计文档实体，继承 langchain Document |
| `chunk.py` | `Chunk`, `ChunkMetadata` | 文档分块及其元数据 |
| `value_objects.py` | `ChunkId`, `DocumentId` | 值对象 |

**ChunkMetadata 包含字段**：
- 身份标识：`chunk_id`, `document_id`, `content_hash`
- 层级结构：`header_path`, `heading`, `heading_level`
- 位置信息：`chunk_index`, `chunk_start_index`, `section_start_index`, `total_chunks`
- 内容类型：`chunk_type`
- 来源信息：`source`, `created_at`

### 3.3 向量数据库层

#### 3.3.1 ChromaStore ([chroma_store.py](file:///c:/AA_GSE_DP/DR-2/vec_db/store/chroma_store.py))

ChromaDB 通用存储封装。

**初始化参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_name` | str | "design_references" | 集合名称 |
| `persist_directory` | str | "./chroma_db" | 本地持久化路径 |
| `use_remote` | bool | True | 是否使用远程服务器 |
| `remote_host` | str | "localhost" | 远程服务器地址 |
| `remote_port` | int | 8000 | 远程服务器端口 |

**核心方法**：
| 方法 | 功能 |
|------|------|
| `store()` | 通用存储（ids, documents, embeddings, metadatas） |
| `store_chunk()` | 存储单个文档分块（带完整 ChunkMetadata） |
| `similarity_search()` | 向量相似度检索 |
| `check_by_content_hash()` | 通过内容哈希检查是否存在 |

**集合命名约定**：
- `prd_documents`：PRD 文档块
- `prototype_descriptions`：原型图描述
- `design_references`：设计标准参考

#### 3.3.2 PrototypeStore ([prototype_store.py](file:///c:/AA_GSE_DP/DR-2/vec_db/store/prototype_store.py))

原型图描述专用存储，基于 ChromaStore。

**核心方法**：
| 方法 | 功能 |
|------|------|
| `store()` | 存储原型描述（防重复：基于 id 或 document_id + page_name） |
| `retrieve()` | 检索原型描述 |
| `update_document_id()` | 更新原型图的关联文档 ID |

#### 3.3.3 BgeM3Embeddings ([bge_m3_embedding.py](file:///c:/AA_GSE_DP/DR-2/vec_db/embeddings/bge_m3_embedding.py))

BGE-M3 向量化实现，继承 langchain `Embeddings`。

**初始化参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | "BAAI/bge-m3" | 模型名称 |
| `device` | str | None | 设备（自动检测 CUDA） |
| `normalize_embeddings` | bool | True | 是否归一化 |

**核心方法**：
| 方法 | 功能 |
|------|------|
| `embed_query()` | 嵌入单条查询文本 |
| `embed_documents()` | 批量嵌入文档列表 |

#### 3.3.4 文档分块器 ([chunker.py](file:///c:/AA_GSE_DP/DR-2/vec_db/embeddings/chunker.py))

**ProductStandardChunker**：产品设计标准文档专用切分器

**初始化参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | int | 600 | 目标 chunk 大小（tokens） |
| `overlap` | int | 90 | chunk 之间的重叠大小（tokens） |
| `min_section_level` | int | 1 | 最小保留的标题级别 |
| `separate_tables` | bool | True | 是否将表格单独作为 chunk |

**切分策略**：
1. 按 Markdown 标题层级分割（`\n## `, `\n# `）
2. 按段落分割（`\n\n`）
3. 按句子分割（`。！？；，`）
4. 按字符分割（` `）

### 3.4 模型层

#### 3.4.1 MiniMaxModel ([minimax.py](file:///c:/AA_GSE_DP/DR-2/models/minimax.py))

MiniMax ChatOpenAI 兼容接口，用于 Agent 主体交互。

**API 配置**：
- Base URL: `https://api.minimaxi.com/v1`
- 默认模型: `MiniMax-M2.7`

#### 3.4.2 GLMModel ([glm.py](file:///c:/AA_GSE_DP/DR-2/models/glm.py))

GLM-4.6V-Flash 视觉分析接口，用于原型图分析。

**API 配置**：
- Base URL: `https://open.bigmodel.cn/api/paas/v4`
- 默认模型: `glm-4.6v-flash`

**核心方法**：
| 方法 | 功能 |
|------|------|
| `analyze_image()` | 同步分析图片 |
| `analyze_image_async()` | 异步分析图片 |

---

## 4. 系统使用

### 4.1 CLI 命令

```
命令说明:
  new                          - 创建新会话
  upload <session_id> <prd.md> [proto1.png] [proto2.png]...
                                - 上传 PRD 和原型图，自动绑定
  bind <session_id> <proto_id> <doc_id>
                                - 绑定原型图到 PRD
  unbind <session_id> <proto_id>
                                - 解除绑定
  session <session_id>         - 查看会话状态
  review <session_id>          - 开始审查指定会话（流式）
  review <session_id> --sync  - 开始审查指定会话（同步，非流式）

  直接输入审查请求，Agent 将自动处理
```

### 4.2 输出模式

- **流式模式**（默认）：实时显示 AI 思考过程、工具调用和结果
- **同步模式**：`review <session_id> --sync`，等待完整结果后一次性输出

---

## 5. 数据存储

### 5.1 SQLite Session 持久化

系统使用 SQLite 数据库持久化审查会话元数据。

**数据库路径**：`{chroma_persist_dir}/sessions.db`（默认 `./chroma_data/sessions.db`）

**数据库表结构**：

#### sessions 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | TEXT PRIMARY KEY | 会话唯一标识 |
| `status` | TEXT NOT NULL | 会话状态 |
| `created_at` | TEXT NOT NULL | 创建时间 (ISO 格式) |
| `updated_at` | TEXT NOT NULL | 最后更新时间 (ISO 格式) |
| `document_id` | TEXT | 关联的 PRD 文档 ID |
| `document_file_name` | TEXT | PRD 文件名 |
| `document_file_path` | TEXT | PRD 文件路径 |
| `document_chunk_ids` | TEXT | Chunk IDs 列表 (JSON) |
| `document_metadata` | TEXT | 文档元数据 (JSON) |

#### prototypes 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | TEXT PRIMARY KEY | 原型图唯一标识 |
| `session_id` | TEXT NOT NULL | 所属会话 ID (外键) |
| `name` | TEXT NOT NULL | 原型图名称 |
| `document_id` | TEXT | 关联的 PRD 文档 ID |
| `page_name` | TEXT | 页面名称 |
| `layout` | TEXT | 布局描述 |
| `components` | TEXT | 组件列表 (JSON) |
| `interactions` | TEXT | 交互描述 |
| `query_text` | TEXT | 检索用文本 |
| `image_path` | TEXT | 原型图文件路径 |

**持久化机制**：
- 启动时自动调用 `_load_sessions()` 加载所有会话到内存
- 所有会话操作（create、register_prd、register_prototype、bind、unbind）自动同步到数据库
- 使用 `INSERT OR REPLACE` 实现自动更新

### 5.2 ChromaDB Collections

| Collection | 用途 | 主要字段 |
|------------|------|----------|
| `prd_documents` | PRD 文档块 | content, document_id, header_path, chunk_index |
| `prototype_descriptions` | 原型图描述 | query_text, document_id, layout, components |
| `design_references` | 设计标准 | content, heading |

### 5.3 本地持久化

- 路径：`./chroma_data/`
- 使用 `chromadb.PersistentClient` 进行本地存储
- SQLite 数据库与 ChromaDB 共用同一持久化目录

---

## 6. 依赖配置

### 6.1 Python 版本

要求：Python >= 3.12

### 6.2 核心依赖

```
chromadb>=0.5.0
langchain>=1.2.15
langgraph>=0.0.20
sentence-transformers>=5.4.1
transformers>=4.50.0
pydantic>=2.0
```

### 6.3 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINIMAX_API_KEY` | *(内置默认值)* | MiniMax API 密钥 |
| `GLM_API_KEY` | *(内置默认值)* | GLM API 密钥 |

---

## 7. 关键实现细节

### 7.1 PRD 文档处理流程

```
1. 读取 .md 文件内容
2. 生成 document_id (UUID 前 16 位)
3. 使用 ProductStandardChunker 切分文档
4. 使用 BgeM3Embeddings 生成向量
5. 存储到 ChromaDB prd_documents collection
```

### 7.2 原型图分析流程

```
1. 调用 GLMModel.analyze_image()
2. 解析返回的结构化分析结果
3. 提取 layout, components, interactions
4. 生成 query_text 用于检索
5. 使用 BgeM3Embeddings 生成向量
6. 存储到 PrototypeStore
```

### 7.3 审查工作流

```
1. retrieve_standards() - 检索设计标准
2. retrieve_prd_chunks() - 检索 PRD 内容
3. analyze_prototype()   - 分析原型图
4. retrieve_prototypes() - 检索历史原型
5. generate_report()     - 生成审查报告
```

### 7.4 流式输出实现

**LangGraph 版本**：通过 `_parse_stream_event()` 解析 LangGraph 流事件，识别 THINKING、TOOL_RESULT、FINAL 等类型。

---

## 8. 注意事项

### 8.1 编码问题

原型图分析结果可能存在编码问题（如 GBK 乱码），代码中包含 `_parse_prototype_analysis()` 方法进行自动修复。

### 8.2 性能考虑

当前实现在循环内逐个插入 ChromaDB（见 session.py 注释），后续版本可优化为批量插入。

### 8.3 API 限流

GLM API 返回 429 错误时，analyze_prototype 工具会返回 rate_limited 状态提示用户稍后重试。
