"""Safe ``when`` expressions over event metrics.

A deliberately tiny language: comparisons, ``and``/``or``/``not``, arithmetic,
``in`` against literal lists, and metric names. No calls, attributes,
subscripts, or comprehensions, so rule text can never run code.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Iterable, Mapping
from typing import Any

MAX_EXPRESSION_CHARS = 300

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_LITERAL_ALIASES = {"true": True, "false": False, "null": None}


class ConditionError(ValueError):
    """The expression is invalid or uses something outside the whitelist."""


class _Missing(Exception):
    """A referenced metric is absent on this event."""


def compile_condition(
    expression: str, allowed_names: Iterable[str] | None = None
) -> ast.Expression:
    """Parse and validate ``expression``; raise ConditionError if unsafe."""
    text = (expression or "").strip()
    if not text:
        raise ConditionError("condition is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ConditionError(
            f"condition is longer than {MAX_EXPRESSION_CHARS} characters"
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"invalid condition syntax: {exc.msg}") from None
    allowed = set(allowed_names) if allowed_names is not None else None
    _validate(tree.body, allowed)
    return tree


def condition_names(tree: ast.Expression) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in _LITERAL_ALIASES
    }


def evaluate_condition(tree: ast.Expression, metrics: Mapping[str, Any]) -> bool:
    """True if the condition holds; missing metrics or type errors are False."""
    try:
        return bool(_eval(tree.body, metrics))
    except (_Missing, TypeError, ZeroDivisionError, ValueError):
        return False


def _validate(node: ast.AST, allowed: set[str] | None) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ConditionError("unsupported boolean operator")
        for value in node.values:
            _validate(value, allowed)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise ConditionError("unsupported unary operator")
        _validate(node.operand, allowed)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise ConditionError("unsupported arithmetic operator")
        _validate(node.left, allowed)
        _validate(node.right, allowed)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise ConditionError("unsupported comparison")
        _validate(node.left, allowed)
        for comparator in node.comparators:
            _validate(comparator, allowed)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            if not isinstance(element, ast.Constant):
                raise ConditionError("lists may only contain literal values")
            _validate(element, allowed)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ConditionError("unsupported literal")
    elif isinstance(node, ast.Name):
        if node.id in _LITERAL_ALIASES:
            return
        if node.id.startswith("_"):
            raise ConditionError(f"invalid metric name: {node.id}")
        if allowed is not None and node.id not in allowed:
            known = ", ".join(sorted(allowed)) or "none"
            raise ConditionError(
                f"unknown metric '{node.id}' (known metrics: {known})"
            )
    else:
        raise ConditionError(
            f"unsupported expression element: {type(node).__name__}"
        )


def _eval(node: ast.AST, metrics: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = _eval(value, metrics)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            try:
                result = _eval(value, metrics)
            except _Missing:
                result = False
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand, metrics))
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](
            _eval(node.left, metrics), _eval(node.right, metrics)
        )
    if isinstance(node, ast.Compare):
        left = _eval(node.left, metrics)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, metrics)
            if left is None or right is None:
                if type(op) not in (ast.Eq, ast.NotEq):
                    return False
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(element, metrics) for element in node.elts]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _LITERAL_ALIASES:
            return _LITERAL_ALIASES[node.id]
        if node.id not in metrics:
            raise _Missing(node.id)
        return metrics[node.id]
    raise ConditionError(f"unsupported expression element: {type(node).__name__}")
