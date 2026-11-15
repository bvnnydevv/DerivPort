"""Normal Streamlit block interactions preserve the active rule until apply."""
import json
from io import BytesIO

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from derivport.blocks import compile_blocks, preset_blocks
from derivport.formula import evaluate_formula
import derivport.blocks_ui as blocks_ui


SCRIPT = '''
import pandas as pd
import streamlit as st
from derivport.blocks_ui import render_blocks
from derivport.analysis import apply_rule
base = pd.DataFrame({'predicted_return':[4.,2.], 'momentum':[3.,5.], 'uncertainty':[1.,0.], 'volatility':[2.,3.]}, index=['AAPL','MSFT'])
st.session_state.setdefault('valid_formula','predicted_return - k * uncertainty')
st.session_state.setdefault('valid_k',1.)
render_blocks(base, 'AAPL')
formula = st.text_input('Ranking formula', value=st.session_state.valid_formula, key='formula')
try:
    st.session_state.active_scores = apply_rule(base, formula, st.session_state.valid_k)
except Exception:
    pass
'''


def test_native_block_changes_compile_preview_then_apply_and_export(monkeypatch):
    exports = {}
    real_download = st.download_button
    def capture_download(*args, **kwargs):
        if kwargs.get('key', '').startswith('block_download'):
            exports[kwargs['key']] = args[1]
        return real_download(*args, **kwargs)
    monkeypatch.setattr(st, 'download_button', capture_download)
    app = AppTest.from_string(SCRIPT, default_timeout=30).run()
    assert not app.exception
    initial = app.text_input(key='formula').value
    original = app.session_state['active_scores'].copy()
    app.number_input(key='block_native_return_weight').set_value(.9).run()
    assert not app.exception
    assert app.text_input(key='formula').value == initial
    pd.testing.assert_frame_equal(app.session_state['active_scores'], original)
    app.button(key='block_apply').click().run()
    assert not app.exception
    expected = compile_blocks(app.session_state['block_items'])
    assert app.text_input(key='formula').value == expected
    assert app.session_state['valid_formula'] == expected
    assert app.session_state['active_scores'].loc['AAPL', 'score'] != original.loc['AAPL', 'score']
    saved = json.loads(exports['block_download_json'])
    assert saved['formula'] == expected
    assert compile_blocks(saved['blocks']) == expected
    assert f'return {expected}' in exports['block_download_python']
    app.button(key='block_move_error_-1').click().run()
    assert not app.exception
    assert [item['id'] for item in app.session_state['block_items']] == ['return','error','momentum','volatility']
    app.button(key='block_remove_momentum').click().run()
    assert not app.exception
    assert len(app.session_state['block_items']) == 3
    app.button(key='block_add_momentum').click().run()
    assert not app.exception
    assert len(app.session_state['block_items']) == 4
    app.selectbox(key='block_preset').set_value('Momentum').run()
    app.button(key='block_reset').click().run()
    assert not app.exception
    assert len(app.session_state['block_items']) == 1
    assert app.button(key='block_remove_momentum').disabled


def test_invalid_current_divisor_cannot_replace_active_algorithm():
    app = AppTest.from_string(SCRIPT, default_timeout=30).run()
    initial = app.text_input(key='formula').value
    app.selectbox(key='block_native_error_operator').set_value('/').run()
    assert not app.exception
    assert app.error
    assert app.text_input(key='formula').value == initial
    assert not any(button.key == 'block_apply' for button in app.button)


@pytest.mark.parametrize('packet_kind', ['valid', 'stale', 'invalid'])
def test_browser_board_packets_are_revision_checked_and_validated(monkeypatch, packet_kind):
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    initial = preset_blocks()
    changed = preset_blocks('Momentum')
    if packet_kind == 'invalid':
        changed[0]['factor'] = '__import__("os")'
    state = State(block_items=initial, block_revision=7,
        blocks_canvas={'edit': {'blocks': changed, 'revision': 6 if packet_kind == 'stale' else 7}})
    monkeypatch.setattr(blocks_ui.st, 'session_state', state)
    blocks_ui._receive_blocks()
    if packet_kind == 'valid':
        assert state['block_items'] == changed
        assert state['block_revision'] == 8
    else:
        assert state['block_items'] == initial
        assert state['block_revision'] == 7
        assert ('block_error' in state) == (packet_kind == 'invalid')
