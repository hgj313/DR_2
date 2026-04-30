"""PRD & 原型图审查系统入口"""

from pathlib import Path

from agent.review_agent import create_review_agent
from agent.session import get_session_manager


def main():
    print("=" * 60)
    print("PRD & 原型图审查系统")
    print("=" * 60)
    print()
    print("命令说明:")
    print("  new                          - 创建新会话")
    print("  upload <session_id> <prd.md> [proto1.png] [proto2.png]...")
    print("                                - 上传 PRD 和原型图，自动绑定")
    print("  bind <session_id> <proto_id> <doc_id>")
    print("                                - 绑定原型图到 PRD")
    print("  unbind <session_id> <proto_id>")
    print("                                - 解除绑定")
    print("  session <session_id>         - 查看会话状态")
    print("  review <session_id>          - 开始审查指定会话")
    print()
    print("  直接输入审查请求，Agent 将自动处理")
    print()

    agent = create_review_agent()
    session_mgr = get_session_manager()

    current_session_id = None

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("再见!")
                break

            parts = user_input.split()
            cmd = parts[0].lower()

            # 创建新会话
            if cmd == "new":
                session_id = session_mgr.create_session()
                current_session_id = session_id
                print(f"新会话: {session_id}")
                print(session_mgr.get_session_summary(session_id))
                continue

            # 上传文件
            if cmd == "upload":
                if len(parts) < 2:
                    print("用法: upload <session_id> [prd.md] [proto1.png] [proto2.png] ...")
                    continue

                session_id = parts[1]
                current_session_id = session_id

                # 解析文件列表
                prd_file = None
                prototype_files = []

                for f in parts[2:]:
                    p = Path(f)
                    if p.suffix.lower() in [".md", ".docx", ".doc"]:
                        prd_file = f
                    elif p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        prototype_files.append({"path": f, "name": p.stem})

                result = session_mgr.upload_session(
                    session_id,
                    prd_file=prd_file,
                    prototype_files=prototype_files,
                )

                doc_info = result["document"]
                protos = result["prototypes"]
                bindings = result["bindings"]

                if doc_info:
                    print(f"PRD 文档: {doc_info.file_name}")
                    print(f"  Document ID: {doc_info.document_id}")
                    print(f"  Chunks: {len(doc_info.chunk_ids)}")
                else:
                    print("PRD 文档: (未上传)")

                print(f"原型图: {len(protos)} 张")
                for p in protos:
                    bind_status = f"已绑定到 {p.document_id}" if p.document_id else "(未绑定)"
                    print(f"  - {p.name} [{p.id[:8]}] {bind_status}")

                print(f"自动绑定: {len(bindings)} 对")
                continue

            # 绑定
            if cmd == "bind":
                if len(parts) < 4:
                    print("用法: bind <session_id> <proto_id> <doc_id>")
                    continue

                session_id, proto_id, doc_id = parts[1], parts[2], parts[3]
                success = session_mgr.bind_prototype_to_document(session_id, proto_id, doc_id)
                print("绑定成功" if success else "绑定失败")
                continue

            # 解绑
            if cmd == "unbind":
                if len(parts) < 3:
                    print("用法: unbind <session_id> <proto_id>")
                    continue

                session_id, proto_id = parts[1], parts[2]
                success = session_mgr.unbind_prototype(session_id, proto_id)
                print("已解绑" if success else "解绑失败")
                continue

            # 查看会话
            if cmd == "session":
                if len(parts) < 2:
                    print("用法: session <session_id>")
                    continue

                print(session_mgr.get_session_summary(parts[1]))
                continue

            # 审查
            if cmd == "review":
                if len(parts) < 2:
                    print("用法: review <session_id>")
                    continue

                session_id = parts[1]
                session = session_mgr.get_session(session_id)

                if not session:
                    print(f"会话 {session_id} 不存在")
                    continue

                # 构建审查上下文
                context_parts = ["请审查以下内容:"]

                if session.document:
                    context_parts.append(f"\nPRD 文档: {session.document.file_name}")
                    context_parts.append(f"Document ID: {session.document.document_id}")

                if session.prototypes:
                    context_parts.append(f"\n原型图 ({len(session.prototypes)} 张):")
                    for p in session.prototypes:
                        bind_info = f"→ {p.document_id}" if p.document_id else "(未绑定)"
                        context_parts.append(f"  - {p.name} [{bind_info}]")

                context_parts.append("\n请执行合规性审查并生成报告。")

                user_msg = "\n".join(context_parts)
                result = agent.invoke(user_msg, thread_id=session_id)

                print("\n--- 审查结果 ---")
                for message in result["messages"]:
                    if hasattr(message, "content") and message.content:
                        print(message.content)
                continue

            # 其他直接发给 Agent
            result = agent.invoke(user_input, thread_id=current_session_id)

            print("\n--- 回复 ---")
            for message in result["messages"]:
                if hasattr(message, "content") and message.content:
                    print(message.content)

            print()

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
