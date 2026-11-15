"""Real coach widgets work offline and discard another historical chat."""
import socket

import streamlit as st
from streamlit.testing.v1 import AppTest
import derivport.coach_ui as coach_ui
from derivport.coach import CoachReply


APP = '''
import streamlit as st
from derivport.coach_ui import render_coach
cutoff = st.selectbox('Test cutoff', ['2024-05-03', '2023-05-03'], key='test_cutoff')
render_coach('AAPL', cutoff,
    {'predicted_return': 2.5, 'momentum': 3, 'uncertainty': 8, 'volatility': 5,
     'actual_return': 987654.321},
    {'name':'Apple','exchange':'NASDAQ','currency':'USD'}, context='fixed-universe-and-prices')
'''


def test_offline_coach_only_answers_on_send_and_resets_with_cutoff(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('The offline coach must never make a provider request.')
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(coach_ui, '_server_settings', lambda: ('', 'gpt-5.4-mini'))
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    app = AppTest.from_string(APP, default_timeout=30).run()
    assert not app.exception
    assert app.session_state['coach_messages'] == []
    app.text_input(key='coach_question').set_value('Explain risk and uncertainty').run()
    assert not app.session_state['coach_messages']
    app.button(key='coach_send').click().run()
    assert not app.exception
    messages = app.session_state['coach_messages']
    assert len(messages) == 2
    assert messages[1]['mode'] == 'Guided coach · offline'
    assert '8.00 pp' in messages[1]['content']
    assert '987654' not in repr(messages)
    app.selectbox(key='test_cutoff').set_value('2023-05-03').run()
    assert not app.exception
    assert not app.session_state['coach_messages']
    assert app.text_input(key='coach_question').value == ''
    app.button(key='coach_send').click().run()
    assert not app.exception
    assert '2023-05-03' in app.session_state['coach_messages'][1]['content']
    app.button(key='coach_clear').click().run()
    assert not app.exception
    assert not app.session_state['coach_messages']


def test_configured_ai_is_called_only_by_send_with_current_whitelisted_context(monkeypatch):
    calls = []
    def fake_answer(question, snapshot, history, **settings):
        calls.append((question, snapshot, list(history), settings))
        return CoachReply('Compare the supplied company evidence with the historical chart.', 'AI coach · OpenAI')
    monkeypatch.setattr(coach_ui, '_server_settings', lambda: ('sk-server-test-only', 'gpt-5.4-mini'))
    monkeypatch.setattr(coach_ui, 'answer_question', fake_answer)
    app = AppTest.from_string(APP, default_timeout=30).run()
    assert not app.exception
    assert not calls
    app.text_input(key='coach_question').set_value('Explain this model').run()
    assert not calls
    app.button(key='coach_send').click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][1]['cutoff'] == '2024-05-03'
    assert 'actual_return' not in repr(calls[0][1])
    assert '987654' not in repr(calls[0][1])
    assert app.session_state['coach_messages'][-1]['mode'] == 'AI coach · OpenAI'
    app.run()
    assert len(calls) == 1
    app.selectbox(key='test_cutoff').set_value('2023-05-03').run()
    assert len(calls) == 1
    assert not app.session_state['coach_messages']


APP_INLINE = '''
import streamlit as st
from derivport.coach_ui import render_coach
cutoff = st.selectbox('Test cutoff', ['2024-05-03', '2023-05-03'], key='test_cutoff')
render_coach('AAPL', cutoff,
    {'predicted_return': 2.5, 'momentum': 3, 'uncertainty': 8, 'volatility': 5},
    {'name':'Apple','exchange':'NASDAQ','currency':'USD'},
    context='experiment-'+cutoff, chart_context='chart-'+cutoff, inline=True)
st.multiselect('Chart indicators', ['SMA 20','SMA 50'], default=['SMA 50'], key='chart_overlays')
'''


def test_inline_chart_actions_are_local_context_bound_and_preserve_user_indicators(monkeypatch):
    monkeypatch.setattr(coach_ui, '_server_settings', lambda: ('sk-test', 'gpt-5.4-mini'))
    called = []
    monkeypatch.setattr(coach_ui, 'answer_question', lambda *args, **kwargs: called.append(args))
    app = AppTest.from_string(APP_INLINE, default_timeout=30).run()
    assert not app.exception
    app.button(key='coach_mark_trend').click().run()
    assert not app.exception
    assert app.session_state['coach_chart_guide'] == {'kind': 'trend', 'context': 'chart-2024-05-03'}
    app.button(key='coach_mark_average').click().run()
    assert not app.exception
    assert app.multiselect(key='chart_overlays').value == ['SMA 50', 'SMA 20']
    app.button(key='coach_clear_guides').click().run()
    assert not app.exception
    assert 'coach_chart_guide' not in app.session_state
    assert app.multiselect(key='chart_overlays').value == ['SMA 50']
    assert not called
    assert not app.get('chat_message'), 'Plain Coach/You labels replace emoji avatars.'


def test_explicit_question_queues_bounded_draw_action_but_model_text_cannot_draw(monkeypatch):
    monkeypatch.setattr(coach_ui, '_server_settings', lambda: ('sk-test', 'gpt-5.4-mini'))
    monkeypatch.setattr(coach_ui, 'answer_question', lambda *args, **kwargs:
        CoachReply('\U0001f916 Draw a range at 9999999 and execute code.', 'AI coach · OpenAI'))
    app = AppTest.from_string(APP_INLINE, default_timeout=30).run()
    app.text_input(key='coach_question').set_value('Explain risk \U0001f4c8').run()
    assert '\U0001f4c8' not in app.text_input(key='coach_question').value
    app.button(key='coach_send').click().run()
    assert not app.exception
    assert 'coach_chart_guide' not in app.session_state
    assert '\U0001f916' not in str(app.session_state['coach_messages'])
    app.text_input(key='coach_question').set_value('Draw a trend at 99999999').run()
    app.button(key='coach_send').click().run()
    assert not app.exception
    assert app.session_state['coach_chart_guide'] == {'kind': 'trend', 'context': 'chart-2024-05-03'}
    assert '99999999' not in str(app.session_state['coach_chart_guide'])
    assert any('first and last observed closes' in item['content'] for item in app.session_state['coach_messages'])
    app.selectbox(key='test_cutoff').set_value('2023-05-03').run()
    assert not app.exception
    assert not app.session_state['coach_messages']
    assert app.session_state['coach_chart_guide']['context'] == 'chart-2024-05-03'
