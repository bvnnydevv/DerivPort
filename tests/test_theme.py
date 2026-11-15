import pandas as pd
import plotly.graph_objects as go
import pytest

from derivport.theme import FONT, theme_from_state, theme_css, style_plot


def test_custom_palette_contrasts_with_background_and_rejects_css_injection():
    dark = theme_from_state({})
    light = theme_from_state({'theme_name':'Custom','theme_background':'#ffffff','theme_accent':'#234567'})
    assert dark['text'] != light['text']
    assert light['accent_text'] == '#ffffff'
    assert 'Helvetica Neue' in theme_css(dark)
    for value in ('red', '#fff', '#fff;}body{display:none}', None):
        with pytest.raises(ValueError):
            theme_from_state({'theme_accent':value})


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
