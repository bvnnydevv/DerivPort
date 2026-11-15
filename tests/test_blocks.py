"""A block stack is explicit bounded arithmetic, never executable input."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest

from derivport.blocks import (MAX_BLOCKS, compile_blocks, move_block,
    normalize_blocks, preset_blocks, python_algorithm)
from derivport.formula import FormulaError, evaluate_formula


def test_balanced_blocks_exactly_match_original_formula_on_multiple_assets():
    values = pd.DataFrame({'predicted_return': [4., -2., 8.], 'momentum': [2., 3., -4.],
        'uncertainty': [1., 2., 0.], 'volatility': [3., 4., 5.]})
    expression = compile_blocks(preset_blocks())
    for k in [0., 1., 3.5]:
        np.testing.assert_allclose(evaluate_formula(expression, values, k),
            .6 * values.predicted_return + .4 * values.momentum - k * (.5 * values.uncertainty + .25 * values.volatility))


def test_reordering_and_noncommutative_operators_follow_explicit_stack_order():
    items = preset_blocks('Momentum')
    items.extend([dict(id='return', factor='predicted_return', weight=2., operator='/', use_k=False),
        dict(id='vol', factor='volatility', weight=1., operator='*', use_k=False)])
    inputs = dict(momentum=12., predicted_return=2., volatility=3., uncertainty=1.)
    assert evaluate_formula(compile_blocks(items), inputs, 1.) == pytest.approx(9.)
    moved = move_block(items, 'vol', -1)
    assert evaluate_formula(compile_blocks(moved), inputs, 1.) == pytest.approx(9.)
    # Moving a leading division term changes the explicit start rather than
    # permitting an invalid leading operator or hidden expression precedence.
    moved = move_block(items, 'return', -1)
    assert moved[0]['operator'] == '+'
    assert evaluate_formula(compile_blocks(moved), inputs, 1.) == pytest.approx(48.)
    assert items[0]['id'] == 'momentum'


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), -1., 11., True, '1; import os'])
def test_nonfinite_or_unbounded_weights_are_rejected(weight):
    items = preset_blocks('Momentum');items[0]['weight'] = weight
    with pytest.raises(FormulaError):
        compile_blocks(items)


@pytest.mark.parametrize('change', [{'factor': '__import__("os")'}, {'factor': []}, {'operator': '**'},
    {'use_k': 'true'}, {'id': '<script>'}, {'unexpected': 1}])
def test_browser_packets_cannot_inject_arithmetic_or_code(change):
    items = preset_blocks('Momentum');items[0].update(change)
    with pytest.raises(FormulaError):
        normalize_blocks(items)


def test_empty_oversized_and_duplicate_stacks_rejected_and_presets_are_isolated():
    for items in [[], {}, [preset_blocks('Momentum')[0]] * (MAX_BLOCKS+1),
        [preset_blocks('Momentum')[0], preset_blocks('Momentum')[0]]]:
        with pytest.raises(FormulaError):
            normalize_blocks(items)
    first = preset_blocks();first[0]['weight'] = 4.
    assert preset_blocks()[0]['weight'] == .6


def test_zero_weight_divisor_and_current_zero_values_do_not_become_silent_scores():
    items = preset_blocks('Defensive');items[1].update(operator='/', weight=0.)
    with pytest.raises(FormulaError, match='zero'):
        compile_blocks(items)
    items[1].update(weight=1.)
    expression = compile_blocks(items)
    with pytest.raises(FormulaError, match='zero'):
        evaluate_formula(expression, dict(predicted_return=4., momentum=2., uncertainty=0., volatility=1.), 1.)


def test_exported_python_is_the_same_constrained_expression():
    blocks = preset_blocks('Balanced')
    # Compile the trusted generated artifact in a test to verify export parity;
    # browser input never becomes Python source without constrained compilation.
    namespace = {}
    exec(python_algorithm(blocks), namespace)
    values = dict(predicted_return=4., momentum=2., uncertainty=1., volatility=3., k=2.)
    assert namespace['score'](**values) == pytest.approx(evaluate_formula(compile_blocks(blocks), values))
