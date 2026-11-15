"""Shared Day, Night and green Pause palettes for the workspace and charts."""
from __future__ import annotations

import re

FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'
DISPLAY_MODES = ('Day', 'Night', 'Pause')
MODE_PALETTES = {
    'Day': dict(background='#eeefec', panel='#f6f7f4', panel2='#e9ece5',
        feature='#dce3d0', text='#1b2d22', muted='#586451', line='#cdd4c8',
        accent='#28543a', accent_text='#ffffff', up='#267044', down='#b23c55'),
    'Night': dict(background='#0b100e', panel='#121b16', panel2='#1b2720',
        feature='#203629', text='#eef4ed', muted='#a7b9ad', line='#34473b',
        accent='#89dfaa', accent_text='#10291b', up='#56d894', down='#f0808b'),
    'Pause': dict(background='#62ffa0', panel='#73ffa9', panel2='#54ed91',
        feature='#4fd283', text='#102f20', muted='#234d33', line='#3d9b62',
        accent='#123622', accent_text='#75ffac', up='#123e29', down='#9e3452'),
}
# Retain aliases for previously saved settings and standalone chart callers.
THEMES = {'Editorial': '#eeefec', 'Midnight': '#050505', 'Graphite': '#121416', 'Deep ocean': '#071b25', 'Custom': '#0c1425'}


def _hex(value):
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        raise ValueError('Use a six-digit color.')
    return value.lower()


def _mix(first, second, amount):
    a, b = _hex(first)[1:], _hex(second)[1:]
    return '#' + ''.join(f'{round(int(a[i:i+2], 16) * (1-amount) + int(b[i:i+2],16) * amount):02x}' for i in (0,2,4))


def _light(value):
    c = [int(_hex(value)[i:i+2], 16) / 255 for i in (1,3,5)]
    c = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in c]
    return .2126*c[0] + .7152*c[1] + .0722*c[2]


def theme_from_state(state):
    name = state.get('theme_name', 'Day')
    if name in MODE_PALETTES:
        return dict(name=name, **MODE_PALETTES[name], compact=False)
    bg = _hex(state.get('theme_background', THEMES['Custom'])) if name == 'Custom' else THEMES.get(name, THEMES['Editorial'])
    accent = _hex(state.get('theme_accent', '#28543a'))
    light = _light(bg) > .35
    feature = '#dce3d0' if light else _mix(bg, '#ffffff', .09)
    text, muted = ('#1b2d22', '#626c60') if light else ('#f3f2ed', '#aaa9a1')
    panel = _mix(bg, '#ffffff', .5) if light else _mix(bg, '#ffffff', .045)
    return dict(name=name, feature=feature, background=bg, panel=panel, panel2=_mix(panel,text,.045),
        text=text, muted=muted, line=_mix(panel,text,.16), accent=accent,
        accent_text='#071018' if _light(accent) > .35 else '#ffffff',
        compact=state.get('theme_density','Comfortable') == 'Compact')


def current_theme():
    import streamlit as st
    return theme_from_state(st.session_state)


def theme_css(palette):
    mapping = {'workspace':'background','panel':'panel','panel2':'panel2','ink':'text',
        'muted':'muted','feature':'feature','line':'line','accent':'accent','accent-text':'accent_text'}
    values = ';'.join(f'--{css}:{_hex(palette[key])}' for css,key in mapping.items())
    gap, pad = ('.55rem','14px') if palette['compact'] else ('.9rem','22px')
    scheme = 'light' if _light(palette['background']) > .35 else 'dark'
    return f':root{{{values};color-scheme:{scheme};--sans:"Helvetica Neue",Helvetica,Arial,sans-serif;--mono:var(--sans);--gap:{gap};--pad:{pad}}}'


def style_plot(fig, palette=None):
    p = palette or current_theme()
    fig.update_layout(font=dict(family=FONT, color=p['muted']),
        paper_bgcolor=p['panel'], plot_bgcolor=p['panel'],
        legend=dict(font_color=p['text']),
        hoverlabel=dict(bgcolor=p['panel2'],font_color=p['text'],bordercolor=p['line']))
    fig.update_xaxes(color=p['muted'], tickfont_color=p['muted'], gridcolor=p['line'], linecolor=p['line'], spikecolor=p['muted'])
    fig.update_yaxes(color=p['muted'], tickfont_color=p['muted'], gridcolor=p['line'], linecolor=p['line'], spikecolor=p['muted'])
    fig.update_layout(modebar=dict(bgcolor=p['panel'], color=p['muted'], activecolor=p['accent']))
    light_colors = {'#bc9bff':'#735195', '#b9a4e8':'#735195', '#ffbd69':'#9b5517', '#dce7e4':'#43694e',
        '#d9e5d2':'#43694e', '#f2cc60':'#876122', '#69d9ff':'#17618b',
        '#ffc966':'#876122', '#8db6ff':'#315b9c', '#d0a8ff':'#6f4595'} if _light(p['background']) > .35 else {
        '#2454ff':'#84a9ff', '#2454FF':'#84a9ff', '#A48645':'#dfbd72'}
    for trace in fig.data:
        if hasattr(trace,'line') and getattr(trace.line, 'color', None) == '#101820':
            trace.line.color = p['text']
        if hasattr(trace,'marker') and isinstance(getattr(trace.marker, 'color', None), str) and trace.marker.color == '#101820':
            trace.marker.color = p['text']
        if hasattr(trace, 'line') and isinstance(getattr(trace.line, 'color', None), str) and trace.line.color in light_colors:
            trace.line.color = light_colors[trace.line.color]
        if hasattr(trace, 'marker') and isinstance(getattr(trace.marker, 'color', None), str) and trace.marker.color in light_colors:
            trace.marker.color = light_colors[trace.marker.color]
    return fig


def style_stock_plot(fig, palette=None):
    """Theme chart furniture without replacing observations or custom candles."""
    p = palette or current_theme()
    style_plot(fig, p)
    fig.update_layout(newshape=dict(line_color=p['accent'], line_width=1.5))
    for shape in fig.layout.shapes:
        if str(shape.name or '').startswith('coach-'):
            shape.line.color = p['accent']
    for annotation in fig.layout.annotations:
        if str(annotation.name or '').startswith('coach-'):
            annotation.update(bgcolor=p['panel2'], bordercolor=p['accent'], font_color=p['text'])
        elif annotation.bgcolor:
            # Price badges use the user's candle color; choose readable text.
            annotation.font.color = '#10291b' if _light(annotation.bgcolor) > .35 else '#ffffff'
        else:
            annotation.font.color = p['muted']
    return fig


def sync_chart_colors(state, palette):
    """Follow mode defaults while preserving colors explicitly customized by a user."""
    previous = state.get('_chart_mode_colors', {})
    defaults = {'chart_up': palette['up'], 'chart_down': palette['down']}
    for key, color in defaults.items():
        if key not in state or state[key] == previous.get(key):
            state[key] = color
    state['_chart_mode_colors'] = defaults
