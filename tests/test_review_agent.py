"""ReviewAgent 端到端测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from agent.review_agent import create_review_agent
from agent.tools import retrieve_standards, analyze_prd


def test_agent_creation():
    """测试 Agent 创建"""
    agent = create_review_agent()
    assert agent is not None
    assert agent.model is not None


def test_retrieve_standards():
    """测试设计标准检索工具"""
    result = retrieve_standards.invoke("交互设计规范")
    assert isinstance(result, str)


def test_analyze_prd():
    """测试 PRD 分析工具"""
    test_prd = """
    # 产品需求文档

    ## 功能需求

    ### 用户管理
    1. 用户可以注册账号
    2. 用户可以登录
    3. 用户可以修改密码

    ### 权限管理
    - 管理员可以查看所有用户
    - 普通用户只能查看自己的信息
    """

    result = analyze_prd.invoke(test_prd)
    assert isinstance(result, str)
    assert len(result) > 0


def test_agent_invoke():
    """测试 Agent 调用（需要 API key）"""
    agent = create_review_agent()

    # 简单测试
    result = agent.invoke("你好，请介绍一下你自己")

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])