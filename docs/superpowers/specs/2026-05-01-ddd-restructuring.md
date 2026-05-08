# DDD 重构设计文档

## 架构概览

```
src/
├── domain/          # 核心业务逻辑，零外部依赖
├── application/     # 用例编排，依赖 Domain
├── infrastructure/  # 外部依赖实现（ChromaDB、LLM）
└── interfaces/      # 外部适配器（CLI、API）
```

## 文件迁移映射

| 当前 | 目标 |
|------|------|
| domain/document.py | src/domain/entities/document.py |
| domain/chunk.py | src/domain/entities/chunk.py |
| domain/value_objects.py | src/domain/value_objects/ids.py |
| agent/session.py | src/application/services/review_session_service.py |
| agent/schemas.py | src/application/schemas.py |
| agent/tools.py | src/infrastructure/agent/tools.py |
| agent/review_agent.py | src/interfaces/adapters/review_agent.py |
| vec_db/store/chroma_store.py | src/infrastructure/persistence/chroma_store.py |
| vec_db/store/prototype_store.py | src/infrastructure/persistence/prototype_store.py |
| models/minimax.py | src/infrastructure/llm/minimax.py |
| models/glm.py | src/infrastructure/llm/glm.py |
| vec_db/embeddings/* | src/infrastructure/embeddings/* |

## 依赖规则

interfaces → application → domain

## 状态

执行中...