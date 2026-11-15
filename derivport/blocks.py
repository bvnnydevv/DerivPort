"""Compile finite, explicitly ordered score blocks into the safe formula language."""
from __future__ import annotations

from copy import deepcopy
import math

from .formula import FormulaError, evaluate_formula

FACTORS = {
    'predicted_return': 'Model return',
    'momentum': 'Momentum',
    'uncertainty': 'Forecast error',
    'volatility': 'Volatility',
}
OPERATORS = ('+', '-', '*', '/')
MAX_BLOCKS = 6
PRESETS = {
    'Balanced': [
        {'id': 'return', 'factor': 'predicted_return', 'weight': .6, 'operator': '+', 'use_k': False},
        {'id': 'momentum', 'factor': 'momentum', 'weight': .4, 'operator': '+', 'use_k': False},
        {'id': 'error', 'factor': 'uncertainty', 'weight': .5, 'operator': '-', 'use_k': True},
        {'id': 'volatility', 'factor': 'volatility', 'weight': .25, 'operator': '-', 'use_k': True},
    ],
    'Momentum': [{'id': 'momentum', 'factor': 'momentum', 'weight': 1., 'operator': '+', 'use_k': False}],
    'Defensive': [
        {'id': 'return', 'factor': 'predicted_return', 'weight': 1., 'operator': '+', 'use_k': False},
        {'id': 'error', 'factor': 'uncertainty', 'weight': 1., 'operator': '-', 'use_k': True},
        {'id': 'volatility', 'factor': 'volatility', 'weight': .5, 'operator': '-', 'use_k': True},
    ],
}


def preset_blocks(name='Balanced'):
    if name not in PRESETS:
        raise FormulaError('Choose a known block starting point.')
    return deepcopy(PRESETS[name])


def normalize_blocks(blocks):
    """Treat browser packets as untrusted structured data, never source code."""
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= MAX_BLOCKS:
        raise FormulaError(f'Use between one and {MAX_BLOCKS} score blocks.')
    result, seen = [], set()
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {'id', 'factor', 'weight', 'operator', 'use_k'}:
            raise FormulaError('Each block needs a factor, weight, operation and k setting.')
        identity = block['id']
        if not isinstance(identity, str) or not identity.isascii() or not identity.replace('-', '').replace('_', '').isalnum() or len(identity) > 40 or identity in seen:
            raise FormulaError('Block identifiers must be unique simple names.')
        seen.add(identity)
        if type(block['factor']) is not str or block['factor'] not in FACTORS or type(block['operator']) is not str or block['operator'] not in OPERATORS or type(block['use_k']) is not bool:
            raise FormulaError('Use the listed factors and arithmetic operators.')
        weight = block['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 10:
            raise FormulaError('Block weights must be finite numbers from 0 to 10.')
        result.append(dict(block, weight=float(weight)))
    result[0]['operator'] = '+'
    return result


def compile_blocks(blocks):
    """Evaluate from top to bottom with explicit left-to-right grouping.

    Multiplication/division between blocks change a score's dimensional meaning;
    no result is relabeled as an expected return.
    """
    expression = ''
    for block in normalize_blocks(blocks):
        term = f"{block['weight']:.8g} * " + ('k * ' if block['use_k'] else '') + block['factor']
        expression = term if not expression else f"({expression} {block['operator']} ({term}))"
    # Exercise the existing restricted parser as well as its length/node limits.
    evaluate_formula(expression, {factor: 1. for factor in FACTORS}, k=1.)
    return expression


def move_block(blocks, identity, direction):
    ordered = normalize_blocks(blocks)
    if direction not in (-1, 1):
        raise FormulaError('Move a block one place up or down.')
    positions = [i for i, row in enumerate(ordered) if row['id'] == identity]
    if not positions:
        raise FormulaError('That block is no longer in the draft.')
    source = positions[0]
    target = min(max(source + direction, 0), len(ordered) - 1)
    ordered[source], ordered[target] = ordered[target], ordered[source]
    return normalize_blocks(ordered)


def python_algorithm(blocks):
    expression = compile_blocks(blocks)
    return ('"""A DerivPort block score. Inputs are percentage points; this is a ranking rule."""\n\n'
        'def score(predicted_return, momentum, uncertainty, volatility, k=1.0):\n'
        f'    return {expression}\n')
