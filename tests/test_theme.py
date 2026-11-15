import pandas as pd
import plotly.graph_objects as go
import pytest

from derivport.theme import (DISPLAY_MODES, FONT, _light, theme_from_state,
    theme_css, style_plot, style_stock_plot, sync_chart_colors)


def test_custom_palette_contrasts_with_background_and_rejects_css_injection():
    editorial = theme_from_state({})
    assert editorial['name'] == 'Day'
    assert editorial['background'] == '#eeefec'
    dark = theme_from_state({'theme_name':'Midnight'})
    light = theme_from_state({'theme_name':'Custom','theme_background':'#ffffff','theme_accent':'#234567'})
    assert dark['text'] != light['text']
    assert light['accent_text'] == '#ffffff'
    assert 'Helvetica Neue' in theme_css(dark)
    for value in ('red', '#fff', '#fff;}body{display:none}', None):
        with pytest.raises(ValueError):
            theme_from_state({'theme_name':'Custom', 'theme_accent':value})


def test_display_modes_keep_text_and_actions_readable():
    def contrast(a, b):
        return (max(_light(a), _light(b)) + .05) / (min(_light(a), _light(b)) + .05)
    for name in DISPLAY_MODES:
        palette = theme_from_state({'theme_name': name})
        for surface in ('background', 'panel', 'panel2', 'feature'):
            assert contrast(palette['text'], palette[surface]) >= 4.5
            assert contrast(palette['muted'], palette[surface]) >= 4.5
        assert contrast(palette['accent'], palette['accent_text']) >= 4.5


def test_mode_colors_follow_defaults_and_preserve_customized_candles():
    state = {}
    sync_chart_colors(state, theme_from_state({'theme_name':'Day'}))
    state['chart_up'] = '#3355aa'  # Explicit user customization.
    sync_chart_colors(state, theme_from_state({'theme_name':'Night'}))
    assert state['chart_up'] == '#3355aa'
    assert state['chart_down'] == theme_from_state({'theme_name':'Night'})['down']
    sync_chart_colors(state, theme_from_state({'theme_name':'Pause'}))
    assert state['chart_up'] == '#3355aa'
    assert state['chart_down'] == theme_from_state({'theme_name':'Pause'})['down']


def test_stock_furniture_and_price_badges_follow_mode_without_changing_candles():
    for name in DISPLAY_MODES:
        fig = go.Figure(go.Candlestick(x=[1,2], open=[10,11], high=[12,13],
            low=[9,10], close=[11,12], increasing_line_color='#3355aa'))
        fig.add_annotation(text='12.00', bgcolor='#3355aa')
        fig.add_annotation(text='Observed window trend', name='coach-trend-label', bgcolor='#15242c')
        palette = theme_from_state({'theme_name':name})
        style_stock_plot(fig, palette)
        assert list(fig.data[0].close) == [11,12]
        assert fig.data[0].increasing.line.color == '#3355aa'
        assert fig.layout.modebar.bgcolor == palette['panel']
        assert fig.layout.xaxis.tickfont.color == palette['muted']
        assert fig.layout.annotations[0].font.color == '#ffffff'
        assert fig.layout.annotations[1].bgcolor == palette['panel2']


def test_theme_does_not_change_observed_candles_or_indicator_data():
    dates = pd.date_range('2024-01-01',periods=3)
    fig = go.Figure([go.Candlestick(x=dates,open=[10,11,12],high=[12,13,14],low=[9,10,11],close=[11,12,13]),
        go.Bar(x=dates,y=[100,200,300],marker_color=['#34d399']*3),
        go.Scatter(x=dates,y=[10,11,12],line_color='#101820')])
    before = list(fig.data[0].close)
    palette = theme_from_state({'theme_name':'Graphite'})
    style_plot(fig,palette)
    assert list(fig.data[0].close) == before
    assert list(fig.data[1].y) == [100,200,300]
    assert fig.layout.font.family == FONT
    assert fig.data[-1].line.color == palette['text']
