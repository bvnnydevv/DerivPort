"""Judge-facing explanations agree with the real ranking arithmetic."""
import pytest

from derivport.blocks import compile_blocks, preset_blocks
from derivport.builder_guide import contribution_rows, describe_blocks, guided_blocks
from derivport.formula import FormulaError, evaluate_formula

INPUTS = dict(predicted_return=4., momentum=3., uncertainty=1., volatility=2.)


def test_guided_controls_produce_bounded_additive_scores_and_explain_risk_direction():
    blocks = guided_blocks(80, 'Strong')
    score = evaluate_formula(compile_blocks(blocks), INPUTS, 1.)
    assert score == pytest.approx(1.8)
    rows = contribution_rows(blocks, INPUTS, 1.)
    assert sum(row['Score contribution'] for row in rows) == pytest.approx(score)
    assert rows[2]['Score contribution'] == -1.
    assert rows[3]['Score contribution'] == -1.
    explanation = describe_blocks(blocks, 1.)
    assert '0.8 × model outlook' in explanation
    assert 'subtract 1 × past forecast error' in explanation
    assert 'subtract 0.5 × price swings' in explanation


def test_advanced_nonadditive_stack_explains_running_score_without_false_contributions():
    blocks = preset_blocks('Momentum')
    blocks.append(dict(id='return', factor='predicted_return', weight=2., operator='/', use_k=False))
    rows = contribution_rows(blocks, INPUTS, 2.)
    assert rows[-1]['Running score'] == pytest.approx(3/8)
    assert 'Score contribution' not in rows[-1]
    assert 'divide the running score by 2 × model outlook' in describe_blocks(blocks, 2.)


@pytest.mark.parametrize('share,risk', [(-1,'Medium'), (101,'Medium'), (True,'Medium'), (50,'Unknown')])
def test_bad_guided_controls_are_rejected(share, risk):
    with pytest.raises(FormulaError):
        guided_blocks(share,risk)
