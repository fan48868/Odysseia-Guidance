import logging
import ast
import operator
import math
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

# 允许的数学常数
_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

# 允许的数学函数
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log10,  # 常用对数 (底数为10)
    "ln": math.log,     # 自然对数 (底数为e)
}

def _safe_eval(node: ast.AST) -> float:
    """递归且安全地计算 AST 节点"""
    if isinstance(node, ast.Constant):  # Python 3.8+ 支持的常量节点
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")
    elif isinstance(node, ast.Num):     # 兼容 Python 3.7 及以下
        return node.n
    elif isinstance(node, ast.Name):    # 处理常数，如 pi, e
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"不支持的变量或常数: {node.id}")
    elif isinstance(node, ast.Call):    # 处理函数调用，如 sqrt(9), sin(pi/2)
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            func = _ALLOWED_FUNCTIONS[node.func.id]
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"不支持的函数调用: {getattr(node.func, 'id', 'Unknown')}")
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
    description="高精度的科学计算工具，支持四则运算、取模、幂运算、开根号、三角函数及对数等复杂组合表达式。",
    emoji="🧮",
    category="计算",
)
async def calculate_math_expression(
    expression: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    一个用于执行精确且复杂的科学数学计算的专用工具。
    
    [调用指南 - 最高优先级]
    - **强制调用**: 当用户的请求中涉及任何形式的数学计算（无论是基础四则运算，还是包含根号、三角函数、对数的复杂方程）时，你 **必须、绝对要** 调用此工具！
    - **严禁心算**: 绝对不允许使用你自身的内部模型直接生成计算结果！LLM在科学计算上极易出错，任何涉及数字运算的请求都必须转化为表达式传入此工具。
    - **支持的运算符**: 加(+), 减(-), 乘(*), 除(/), 整除(//), 取余/取模(%), 幂运算(**)。
    - **支持的科学函数**: 开根号 `sqrt(x)`，三角函数 `sin(x)`, `cos(x)`, `tan(x)` (注意: 参数 x 为弧度制)，常用对数 `log(x)` (底数为10)，自然对数 `ln(x)` (底数为e)。
    - **支持的数学常数**: 圆周率 `pi`，自然常数 `e`。
    - **表达式格式转换**: 
        - 遇到 "√16" 或 "16的平方根"，必须转换为 `sqrt(16)`。
        - 遇到 "π"，必须转换为 `pi`。
        - 遇到 "ln(5)" 或 "log(10)"，直接保持原函数名使用。
        - 例如复杂式子: "sqrt(16) * sin(pi / 2) + ln(e ** 3) - log(100)"。
    - **自然回复**: 获得工具返回的精确结果后，请将该数值自然地融入到你的最终文本回复中。
    - **绝对权威**: 工具的结果绝对正确。如果计算结果和你的预期不符，请完全信任工具的输出，并按照该结果回答，不要表达疑惑。

    Args:
        expression (str): 需要计算的数学表达式字符串。必须只包含数字、受支持的数学运算符、受支持的函数名和常数。

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
        
        # 将一些常见的非常规输入做预处理转换 (防御性编程)
        clean_expr = clean_expr.replace("π", "pi")
        clean_expr = clean_expr.replace("√", "sqrt") # 注意：√16 会变成 sqrt16 会报错，但这里为了演示，主要依赖 AI 正确传参
        
        # 解析为抽象语法树 (AST)
        tree = ast.parse(clean_expr, mode='eval')
        
        # 执行安全计算
        calc_result = _safe_eval(tree.body)
        
        # 处理浮点数精度问题（例如 1.2 + 2.2 = 3.4000000000000004）
        if isinstance(calc_result, float) and calc_result.is_integer():
            calc_result = int(calc_result)
        elif isinstance(calc_result, float):
            calc_result = round(calc_result, 10) # 科学计算保留到小数点后10位
            
        result_data["result"] = calc_result
        log.info(f"计算成功: {clean_expr} = {calc_result}")

    except ZeroDivisionError:
        error_msg = "除数不能为零。"
        result_data["error"] = error_msg
        log.warning(f"计算错误 (除零): {expression}")
    except ValueError as e:
        error_msg = f"表达式计算错误 (可能包含不支持的符号/负数开根号等): {str(e)}"
        result_data["error"] = error_msg
        log.warning(f"计算错误 (值/语法不支持): {expression} - {str(e)}")
    except SyntaxError:
        error_msg = "数学表达式语法错误，请检查括号、运算符、函数格式是否正确。"
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
        "description": "执行精确的科学数学计算。支持四则运算、%、**、sqrt()、sin()、cos()、tan()、log()、ln()、pi、e及括号。AI 必须用此工具进行计算。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string", 
                    "description": "要计算的科学数学表达式，例如: 'sqrt(16) + sin(pi/2) * ln(e)'。"
                }
            },
            "required": ["expression"],
        },
    },
}