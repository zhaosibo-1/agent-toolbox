"""Agent 工具箱 —— 让大模型真正「会用工具」的最小可用实现。

模块划分::

    config      配置读取（.env / 环境变量 + 厂商预设）
    schemas     数据结构定义
    tools       工具子系统
      base        工具抽象 + 由函数签名自动生成 JSON Schema
      builtin     内置工具（计算器 / 时间 / 文本 / HTTP / 文件 / JSON / 单位换算）
      registry    工具注册与安全执行
    llm         大模型客户端（Function Calling 与文本协议双通道）
    agent       推理-行动循环（ReAct / Function Calling）
    main        FastAPI 接口层
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
