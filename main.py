"""PRD & 原型图审查系统入口"""

from pathlib import Path

from agent.review_agent import (
    create_review_agent,
)
from agent.session import get_session_manager
from agent.schemas import ReviewSession


# ANSI 颜色码
class Colors:
    THINKING = "\033[96m"    # 青色 - 思考
    TOOL = "\033[93m"        # 黄色 - 工具
    RESULT = "\033[94m"      # 蓝色 - 结果
    FINAL = "\033[92m"       # 绿色 - 最终
    STATUS = "\033[90m"      # 灰色 - 状态
    TEXT = "\033[95m"        # 洋红 - 文本
    RESET = "\033[0m"


def print_chunk(chunk, use_color: bool = True):
    """根据 chunk 类型格式化输出（统一支持新旧两种 StreamChunk）"""
    color = ""
    prefix = ""
    end = "\n"
    flush = False

    # 获取 chunk 类型
    if hasattr(chunk, 'is_split') and chunk.is_split and hasattr(chunk, 'chunk_type'):
        chunk_type = chunk.chunk_type
    elif hasattr(chunk.type, 'value'):
        chunk_type = chunk.type.value
    else:
        chunk_type = str(chunk.type)

    if chunk_type == "text":
        if use_color:
            color = Colors.TEXT
            prefix = "📝 文本"
        else:
            prefix = "[文本]"
        end = ""
        flush = True
    elif chunk_type == "thinking":
        if use_color:
            color = Colors.THINKING
            prefix = "🤔 思考"
        else:
            prefix = "[思考]"
    elif chunk_type == "responding":
        if use_color:
            color = Colors.FINAL
            prefix = "✨ 回复"
        else:
            prefix = "[回复]"
    elif chunk_type == "tool_call":
        if use_color:
            color = Colors.TOOL
            prefix = "🔧 工具调用"
        else:
            prefix = "[工具调用]"
    elif chunk_type == "tool_result":
        if use_color:
            color = Colors.RESULT
            prefix = "📋 工具结果"
        else:
            prefix = "[工具结果]"
    elif chunk_type == "final":
        if use_color:
            color = Colors.FINAL
            prefix = "✨ 最终回复"
        else:
            prefix = "[最终]"
    elif chunk_type == "status":
        if use_color:
            color = Colors.STATUS
            prefix = "⏳ 状态"
        else:
            prefix = "[状态]"

    reset = Colors.RESET if use_color else ""
    content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)

    print(f"{color}{prefix}: {content}{reset}", end=end, flush=flush)


def stream_review(agent, session_id: str, session: ReviewSession, use_color: bool = True):
    """流式审查会话"""
    context_parts = ["请审查以下内容:"]

    if session.document:
        context_parts.append(f"\nPRD 文档: {session.document.file_name}")
        context_parts.append(f"Document ID: {session.document.document_id}")

    if session.prototypes:
        context_parts.append(f"\n原型图 ({len(session.prototypes)} 张):")
        for p in session.prototypes:
            bind_info = f"→ {p.document_id}" if p.document_id else "(未绑定)"
            # 传递 image_path 让 Agent 能调用 analyze_prototype
            image_path = getattr(p, 'image_path', '') or ''
            context_parts.append(f"  - {p.name} [{bind_info}]")
            if image_path:
                context_parts.append(f"    图片路径: {image_path}")

    context_parts.append("\n请执行合规性审查并生成报告。")
    user_msg = "\n".join(context_parts)

    print("\n" + "=" * 60)
    print("开始流式审查...")
    print("=" * 60 + "\n")

    for chunk in agent.stream(user_msg, thread_id=session_id):
        # 获取 chunk 类型，支持 is_split
        if hasattr(chunk, 'is_split') and chunk.is_split and hasattr(chunk, 'chunk_type'):
            chunk_type = chunk.chunk_type
        elif hasattr(chunk.type, 'value'):
            chunk_type = chunk.type.value
        else:
            chunk_type = str(chunk.type)

        if chunk_type == "thinking":
            print_chunk(chunk, use_color)
            print()
        elif chunk_type == "responding":
            print_chunk(chunk, use_color)
            print()
        elif chunk_type == "tool_result":
            tool_name = chunk.messages[0].name if chunk.messages else "unknown"
            result = chunk.content
            if len(result) > 200:
                result = result[:200] + "..."
            print(f"{Colors.TOOL if use_color else ''}🔧 调用工具: {tool_name}{Colors.RESET if use_color else ''}")
            print(f"{Colors.RESULT if use_color else ''}📋 结果: {result}{Colors.RESET if use_color else ''}")
            print()
        elif chunk_type == "tool_call":
            print_chunk(chunk, use_color)
            print()
        elif chunk_type == "final":
            print(f"{Colors.FINAL if use_color else ''}{'=' * 60}{Colors.RESET if use_color else ''}")
            print(f"{Colors.FINAL if use_color else ''}审查报告:{Colors.RESET if use_color else ''}")
            print(f"{Colors.FINAL if use_color else ''}{'=' * 60}{Colors.RESET if use_color else ''}")
            print()
            print_chunk(chunk, use_color)
            print()
        else:
            print_chunk(chunk, use_color)


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
    print("  review <session_id>          - 开始审查指定会话（流式）")
    print("  review <session_id> --sync  - 开始审查指定会话（同步，非流式）")
    print()
    print()
    print("  直接输入审查请求，Agent 将自动处理")
    print()

    agent = create_review_agent()
    print(f"当前 Agent: langgraph")
    print()
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
                    print("用法: review <session_id> [--sync]")
                    continue

                session_id = parts[1]
                session = session_mgr.get_session(session_id)

                if not session:
                    print(f"会话 {session_id} 不存在")
                    continue

                # 检查是否使用同步模式
                use_sync = "--sync" in parts
                use_color = "--no-color" not in parts

                if use_sync:
                    # 同步模式（非流式）
                    context_parts = ["请审查以下内容:"]

                    if session.document:
                        context_parts.append(f"\nPRD 文档: {session.document.file_name}")
                        context_parts.append(f"Document ID: {session.document.document_id}")

                    if session.prototypes:
                        context_parts.append(f"\n原型图 ({len(session.prototypes)} 张):")
                        for p in session.prototypes:
                            bind_info = f"→ {p.document_id}" if p.document_id else "(未绑定)"
                            image_path = getattr(p, 'image_path', '') or ''
                            context_parts.append(f"  - {p.name} [{bind_info}]")
                            if image_path:
                                context_parts.append(f"    图片路径: {image_path}")

                    context_parts.append("\n请执行合规性审查并生成报告。")

                    user_msg = "\n".join(context_parts)
                    result = agent.invoke(user_msg, thread_id=session_id)

                    print("\n--- 审查结果 ---")
                    for message in result["messages"]:
                        if hasattr(message, "content") and message.content:
                            print(message.content)
                else:
                    # 流式模式
                    stream_review(agent, session_id, session, use_color)
                continue

            # 其他直接发给 Agent（支持流式）
            if current_session_id:
                # 检查是否使用流式
                use_stream = "--stream" in parts
                parts_no_flags = [p for p in parts if not p.startswith("--")]

                if use_stream and len(parts_no_flags) == 1:
                    # 全流式模式
                    print("\n--- 流式回复 ---")
                    for chunk in agent.stream(user_input, thread_id=current_session_id):
                        print_chunk(chunk)
                    print()
                else:
                    # 同步模式
                    msg = " ".join(parts_no_flags) if parts_no_flags else user_input
                    result = agent.invoke(msg, thread_id=current_session_id)

                    print("\n--- 回复 ---")
                    for message in result["messages"]:
                        if hasattr(message, "content") and message.content:
                            print(message.content)
                    print()
            else:
                # 非命令输入：自动创建会话并发送
                session_id = session_mgr.create_session()
                current_session_id = session_id
                use_color = "--no-color" not in parts
                print(f"[自动创建会话: {session_id}]")
                print(f"[发送消息: {user_input[:50]}{'...' if len(user_input) > 50 else ''}]")
                print()
                for chunk in agent.stream(user_input, thread_id=session_id):
                    print_chunk(chunk, use_color)
                print()

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
