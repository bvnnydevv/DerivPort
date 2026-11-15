"""A compact, explicit-send research coach for the selected historical view."""
from __future__ import annotations

import os
import streamlit as st

from .coach import DEFAULT_MODEL, answer_question, build_context, context_key, conversation_for, strip_emojis
from .chart_guides import GUIDE_EXPLANATIONS, guide_intent, queue_chart_guide

LESSONS = {
    'Stock analysis checklist': 'How do I analyze this stock step by step?',
    'Candles and trend': 'How do I read this stock chart, candles and momentum?',
    'Risk and uncertainty': 'How should I understand risk and model uncertainty?',
    'Company and valuation': 'How do I analyze the company and its valuation?',
    'Backtests and algorithms': 'How do I test an algorithm without overfitting?',
    'Explain the model': 'How should I interpret the model forecast and rank?',
}


def _server_settings():
    key, model = os.environ.get('OPENAI_API_KEY', ''), os.environ.get('OPENAI_MODEL', DEFAULT_MODEL)
    # Secrets are server settings only: no UI field, log line, or exported file.
    if not key:
        try:
            key = str(st.secrets.get('OPENAI_API_KEY', ''))
            model = str(st.secrets.get('OPENAI_MODEL', model))
        except (FileNotFoundError, KeyError, RuntimeError):
            pass
    return key, model


def _guide_message(messages, kind):
    if kind == 'clear':
        explanation = 'Coach guides cleared. Existing chart-toolbar drawings and user-selected indicators are managed separately.'
    else:
        explanation = GUIDE_EXPLANATIONS[kind]
    messages.append({'role': 'assistant', 'content': explanation, 'mode': 'Chart guide · local'})
    del messages[:-16]


def render_coach(ticker, cutoff, row, metadata, note=None, context='', inline=False, chart_context=None):
    snapshot = build_context(ticker, cutoff, row, metadata, note)
    key = context_key(snapshot, context)
    messages = conversation_for(st.session_state, key)
    api_key, model = _server_settings()
    chart_context = context if chart_context is None else chart_context
    panel = st.container(border=True, key='coach_panel') if inline else st.popover('Ask the research coach', width='stretch')
    with panel:
        st.markdown(f'**{ticker} coach** · {snapshot["cutoff"]}')
        if api_key:
            st.caption('AI · OpenAI. Send shares your question, public stock evidence and chat. Keep personal details out.')
        else:
            st.caption('Offline guide · prepared lessons, not AI-generated.')
        tools, suggestions = st.columns(2, gap='small')
        with tools, st.popover('Chart tools', width='stretch'):
            left, right = st.columns(2, gap='small')
            with left:
                trend = st.button('Mark trend', key='coach_mark_trend', width='stretch')
                moving = st.button('Moving average', key='coach_mark_average', width='stretch')
            with right:
                window_range = st.button('Mark range', key='coach_mark_range', width='stretch')
                clear_guides = st.button('Clear guides', key='coach_clear_guides', width='stretch')
            for clicked, kind in ((trend, 'trend'), (window_range, 'range'),
                    (moving, 'moving_average'), (clear_guides, 'clear')):
                if clicked:
                    queue_chart_guide(st.session_state, kind, chart_context)
                    _guide_message(messages, kind)
            st.caption('Chart guides use observed prices only. Range marks are window extremes, not validated support or resistance.')
        with suggestions, st.popover('Questions', width='stretch'):
            lesson = st.selectbox('Choose a lesson', list(LESSONS), key='coach_lesson')
            lesson_submitted = st.button('Ask this question', key='coach_lesson_send', width='stretch')
            cleared = st.button('Clear conversation', key='coach_clear', width='stretch')
        # Fill this earlier slot after submission, keeping history above the
        # composer and sizing the first reply correctly in the same rerun.
        history_slot = st.empty()
        with st.container(key='coach_composer'):
            question = st.chat_input(f'Ask about {ticker}…', key='coach_question', max_chars=2000)

        if cleared:
            messages.clear()
        submitted = bool(question) or lesson_submitted
        if submitted:
            prompt = strip_emojis(question or '').strip() or LESSONS[lesson]
            intent = guide_intent(prompt)
            if intent:
                queue_chart_guide(st.session_state, intent, chart_context)
            with st.spinner('Preparing your explanation…'):
                reply = answer_question(prompt, snapshot, messages,
                    api_key=api_key, model=model)
            messages.extend([{'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': strip_emojis(reply.text),
                 'mode': strip_emojis(reply.mode), 'notice': strip_emojis(reply.notice) if reply.notice else None}])
            if intent:
                _guide_message(messages, intent)
            del messages[:-16]
        history_height = (280 if inline else 360) if messages else 170
        with history_slot.container(height=history_height, border=False, key='coach_history'):
            if messages:
                for message in messages:
                    with st.container(border=False):
                        st.markdown('**You**' if message['role'] == 'user' else '**Coach**')
                        if message['role'] == 'assistant':
                            st.caption(strip_emojis(message.get('mode', 'Guided coach · offline')))
                            if message.get('notice'):
                                st.caption(strip_emojis(message['notice']))
                        st.markdown(strip_emojis(message['content']))
            else:
                st.markdown('**Ask about this stock.**')
                st.markdown('Try “Explain this chart” or choose a question above.')
