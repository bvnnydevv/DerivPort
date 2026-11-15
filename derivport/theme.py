"""Session theme controls and shared chart styling."""
from __future__ import annotations

import re

FONT = 'Trebuchet MS, Trebuchet, sans-serif'
THEMES = {'Midnight': '#090f19', 'Graphite': '#121416', 'Deep ocean': '#071b25', 'Custom': '#0c1425'}


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
    name = state.get('theme_name', 'Midnight')
    bg = _hex(state.get('theme_background', THEMES['Custom'])) if name == 'Custom' else THEMES.get(name, THEMES['Midnight'])
    accent = _hex(state.get('theme_accent', '#41d6b3'))
    light = _light(bg) > .35
    text, muted = ('#102132', '#405366') if light else ('#e7eef7', '#a1b2c6')
    panel = _mix(bg, '#000000' if light else '#ffffff', .045)
    return dict(name=name, background=bg, panel=panel, panel2=_mix(panel,text,.045),
        text=text, muted=muted, line=_mix(panel,text,.16), accent=accent,
        accent_text='#071018' if _light(accent) > .35 else '#ffffff',
        compact=state.get('theme_density','Compact') == 'Compact')


def current_theme():
    import streamlit as st
    return theme_from_state(st.session_state)


def theme_css(palette):
    mapping = {'workspace':'background','panel':'panel','panel2':'panel2','ink':'text',
        'muted':'muted','line':'line','accent':'accent','accent-text':'accent_text'}
    values = ';'.join(f'--{css}:{_hex(palette[key])}' for css,key in mapping.items())
    gap, pad = ('.55rem','14px') if palette['compact'] else ('.9rem','22px')
    return f':root{{{values};--sans:"Trebuchet MS",Trebuchet,sans-serif;--mono:var(--sans);--gap:{gap};--pad:{pad}}}'


def render_theme_controls():
    import streamlit as st
    with st.popover('Theme', icon=':material/palette:', width='stretch'):
        st.selectbox('Website theme', list(THEMES), key='theme_name')
        st.color_picker('Accent color', '#41d6b3', key='theme_accent')
        if st.session_state.theme_name == 'Custom':
            st.color_picker('Background color', '#0c1425', key='theme_background')
        st.radio('Layout density', ['Compact','Comfortable'], horizontal=True, key='theme_density')
        st.checkbox('Show illustrations', value=True, key='theme_images')
        st.caption('Trebuchet MS throughout. Settings stay in this browser session.')
    return current_theme()


def style_plot(fig, palette=None):
    p = palette or current_theme()
    fig.update_layout(font=dict(family=FONT, color=p['muted']),
        paper_bgcolor=p['panel'], plot_bgcolor=p['panel'],
        legend=dict(font_color=p['text']),
        hoverlabel=dict(bgcolor=p['panel2'],font_color=p['text'],bordercolor=p['line']))
    fig.update_xaxes(color=p['muted'], gridcolor=p['line'], linecolor=p['line'])
    fig.update_yaxes(color=p['muted'], gridcolor=p['line'], linecolor=p['line'])
    for trace in fig.data:
        if hasattr(trace,'line') and getattr(trace.line, 'color', None) == '#101820':
            trace.line.color = p['text']
        if hasattr(trace,'marker') and isinstance(getattr(trace.marker, 'color', None), str) and trace.marker.color == '#101820':
            trace.marker.color = p['text']
    return fig
