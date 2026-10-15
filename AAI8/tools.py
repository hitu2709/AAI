import ast
import operator as op
from langchain_core.tools import tool

# Sample data for this experiment. Replace with your own test data if needed.
STUDENTS = {
    "S069": {
        "name": "Hitanshu Shah",
        "roll_number": "S069",
        "department": "Computer Engineering",
        "sap_id": "60018240010",
    },
    "S055": {
        "name": "Varun Gupta",
        "roll_number": "S055",
        "department": "Computer Engineering",
        "sap_id": "60018240055",
    },
}

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}

def _safe_eval(node):
    """Evaluate only numeric arithmetic AST nodes; never use eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large for this demo.")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression. Use numbers and +, -, *, /, %, or **.")

@tool
def calculator(expression: str) -> str:
    """Calculate a numeric arithmetic expression such as '12 + 8' or '144 / 12'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: division by zero is not allowed."
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"Error: invalid arithmetic expression ({exc})."

@tool
def student_info(student_id: str) -> str:
    """Look up sample student information using the student roll ID, e.g. S069."""
    key = student_id.strip().upper()
    student = STUDENTS.get(key)
    if not student:
        return f"No student record found for ID {student_id}."
    return (
        f"Name: {student['name']}; Roll number: {student['roll_number']}; "
        f"Department: {student['department']}; SAP ID: {student['sap_id']}"
    )

TOOLS = [calculator, student_info]
