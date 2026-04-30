"""PRD & 原型图审查系统入口"""

from agent.review_agent import create_review_agent


def main():
    print("=" * 60)
    print("PRD & 原型图审查系统")
    print("=" * 60)
    print()

    # 创建 Agent
    agent = create_review_agent()

    print("Agent 已就绪，请输入审查请求...")
    print("示例：")
    print("  请审查 PRD 文档 at ref-doc/产品设计标准文档V2.０--25年持续更新.md")
    print("  和原型图 at tests/temp/prototype.png")
    print()

    while True:
        try:
            user_input = input("> ")

            if user_input.lower() in ["exit", "quit", "q"]:
                print("再见！")
                break

            if not user_input.strip():
                continue

            # 调用 Agent
            result = agent.invoke(user_input)

            # 输出结果
            print("\n--- 审查结果 ---\n")
            for message in result["messages"]:
                if hasattr(message, "content") and message.content:
                    print(message.content)

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


if __name__ == "__main__":
    main()
