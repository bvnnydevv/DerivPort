"""Plain-language views of the same bounded block arithmetic."""
from __future__ import annotations

from .blocks import compile_blocks, normalize_blocks, preset_blocks
from .formula import FormulaError, evaluate_formula

FACTOR_GUIDE = {
    'predicted_return': ('Model outlook', 'The model’s estimate for the next 21 trading sessions. It can be wrong.'),
    'momentum': ('Recent strength', 'How much the stock rose or fell over the last 21 trading sessions.'),
    'uncertainty': ('Past forecast error', 'Typical past forecast error. Subtracting it penalizes less reliable estimates.'),
    'volatility': ('Price swings', 'Recent price variation over a 21-session horizon. Subtracting it penalizes bigger swings.'),
}
GUIDED_STARTS = {'Balanced idea': (60, 'Medium'), 'Follow recent strength': (20, 'Light'), 'Emphasize steadier prices': (60, 'Strong')}
RISK_WEIGHTS = {'Light': .25, 'Medium': .5, 'Strong': 1.}


def guided_blocks(model_share=60, risk='Medium'):
    """Produce an additive draft without changing the active rule or shared k."""
    if type(model_share) is not int or not 0 <= model_share <= 100 or risk not in RISK_WEIGHTS:
        raise FormulaError('Choose a model share from 0 to 100 and a listed risk guardrail.')
    blocks = preset_blocks('Balanced')
    blocks[0]['weight'] = model_share / 100
    blocks[1]['weight'] = 1 - model_share / 100
    blocks[2]['weight'] = RISK_WEIGHTS[risk]
    blocks[3]['weight'] = RISK_WEIGHTS[risk] / 2
    return normalize_blocks(blocks)


def describe_blocks(items, k):
    """Describe literal score operations; no return or performance promise."""
    blocks = normalize_blocks(items)
    operations = {'+': 'add', '-': 'subtract', '*': 'multiply the running score by', '/': 'divide the running score by'}
    parts = []
    for index, block in enumerate(blocks):
        label = FACTOR_GUIDE[block['factor']][0].lower()
        weight = block['weight'] * (k if block['use_k'] else 1)
        parts.append(f'{"Start with" if index == 0 else operations[block["operator"]]} {weight:g} × {label}')
    return '; '.join(parts) + '.'


def contribution_rows(items, row, k):
    """Expose actual arithmetic, including non-additive advanced stacks."""
    blocks = normalize_blocks(items)
    result = []
    additive = all(block['operator'] in ('+', '-') for block in blocks)
    for index, block in enumerate(blocks):
        raw = float(row[block['factor']])
        multiplier = float(f"{block['weight']:.8g}") * (k if block['use_k'] else 1)
        term = raw * multiplier
        running = float(evaluate_formula(compile_blocks(blocks[:index + 1]), row, k))
        item = {'Ingredient': FACTOR_GUIDE[block['factor']][0], 'Input (pp)': raw,
            'Multiplier': multiplier, 'Operation': 'Start' if index == 0 else block['operator'],
            'Running score': running}
        if additive:
            item['Score contribution'] = -term if block['operator'] == '-' else term
        result.append(item)
    return result
