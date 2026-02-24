import logging
import ast
import operator
from typing import Dict, Any
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

# 定义安全的数学操作符映射
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,        # 加 +
    ast.Sub: operator.sub,        # 减 -
    ast.Mult: operator.mul,       # 乘 *
    ast.Div: operator.truediv,    # 除 /
    ast.FloorDiv: operator.floordiv, # 整除 //
    ast.Mod: operator.mod,        # 取余/取模 %
    ast.Pow: operator.pow,        # 幂运算 **
    ast.USub: operator.neg,       # 负号 -
    ast.UAdd: operator.pos,       # 正号 +
}

def _safe_eval(node: ast.AST) -> float:
    """递归且安全地计算 AST 节点"""
    if isinstance(node, ast.Constant):  # Python 3.8+ 支持的常量节点
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")
    elif isinstance(node, ast.Num):     # 兼容 Python 3.7 及以下
        return node.n
    elif isinstance(node, ast.BinOp):   # 二元运算，如 1 + 2
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if not op_func:
            raise ValueError(f"不支持的操作符: {type(node.op)}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp): # 一元运算，如 -5
        operand = _safe_eval(node.operand)
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if not op_func:
            raise ValueError(f"不支持的一元操作符: {type(node.op)}")
        return op_func(operand)
    else:
        raise ValueError(f"不支持的表达式节点: {type(node)}")


@tool_metadata(
    name="数学计算器",
    description="高精度的数学计算工具，支持加减乘除、取余、幂运算及复杂组合表达式。",
    emoji="🧮",
    category="计算",
)
async def calculate_math_expression(
    expression: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    一个用于执行精确数学计算的专用工具。
    
    [调用指南 - 最高优先级]
    - **强制调用**: 当用户的请求中涉及任何形式的数学计算（无论是简单的加减乘除、取余取模，还是复杂的组合运算）时，你 **必须、绝对要** 调用此工具！
    - **严禁心算**: 绝对不允许使用你自身的内部模型直接生成计算结果！LLM在数学计算上容易出现幻觉，为了保证对用户的绝对准确，任何数字计算都必须将表达式传入此工具。
    - **支持的操作符**: 加(+), 减(-), 乘(*), 除(/), 整除(//), 取余/取模(%), 幂运算(**)。
    - **复杂表达式**: 支持传入包含多个数字和操作符的复杂式子，支持使用括号 `()` 来控制运算优先级。例如: "(150.5 + 49.5) * 3 ** 2 / 5 - 8 % 3"。
    - **自然回复**: 获得工具返回的精确结果后，请将该数值自然地融入到你的最终文本回复中。

    Args:
        expression (str): 需要计算的数学表达式字符串。必须只包含数字、受支持的数学运算符和括号。例如: "123 * (45 + 55) % 7"。

    Returns:
        一个包含计算状态和结果的字典。成功时包含 'result'，失败时包含 'error'。
    """
    log.info(f"--- [工具执行]: calculate_math_expression, expression='{expression}' ---")

    result_data = {
        "expression_received": expression,
        "result": None,
        "error": None,
    }

    if not expression or not expression.strip():
        result_data["error"] = "表达式不能为空。"
        return result_data

    try:
        # 去除首尾空格，并将中文括号替换为英文括号，提升用户容错率
        clean_expr = expression.strip().replace("（", "(").replace("）", ")")
        
        # 解析为抽象语法树 (AST)
        tree = ast.parse(clean_expr, mode='eval')
        
        # 执行安全计算
        calc_result = _safe_eval(tree.body)
        
        # 处理浮点数精度问题（例如 1.2 + 2.2 = 3.4000000000000004）
        if isinstance(calc_result, float) and calc_result.is_integer():
            calc_result = int(calc_result)
        elif isinstance(calc_result, float):
            calc_result = round(calc_result, 6) # 保留适当小数位
            
        result_data["result"] = calc_result
        log.info(f"计算成功: {clean_expr} = {calc_result}")

    except ZeroDivisionError:
        error_msg = "除数不能为零。"
        result_data["error"] = error_msg
        log.warning(f"计算错误 (除零): {expression}")
    except ValueError as e:
        error_msg = f"表达式包含不支持的符号或结构: {str(e)}"
        result_data["error"] = error_msg
        log.warning(f"计算错误 (语法不支持): {expression} - {str(e)}")
    except SyntaxError:
        error_msg = "数学表达式语法错误，请检查括号和运算符是否匹配。"
        result_data["error"] = error_msg
        log.warning(f"计算错误 (语法错误): {expression}")
    except Exception as e:
        error_msg = f"计算期间发生未知错误: {str(e)}"
        result_data["error"] = error_msg
        log.error(f"计算未知错误: {expression}", exc_info=True)

    return result_data


# Metadata for the tool (供底层调用大模型接口时注册 schema 使用)
CALCULATE_MATH_EXPRESSION_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_math_expression",
        "description": "执行精确的数学计算，支持加减乘除、取余(%)、幂(**)及括号组合。AI 在遇到任何计算请求时必须使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string", 
                    "description": "要计算的数学表达式，例如: '25.5 * 4 + (10 % 3)'。"
                }
            },
            "required": ["expression"],
        },
    },
}