"""A real drag/reorder score board with an equivalent keyboard editor."""
from __future__ import annotations

import json
from uuid import uuid4

import streamlit as st

from .analysis import apply_rule
from .blocks import (FACTORS, MAX_BLOCKS, OPERATORS, PRESETS, compile_blocks,
    move_block, normalize_blocks, preset_blocks, python_algorithm)
from .calculus import formula_to_latex
from .formula import FormulaError

_HTML = '<section class="score-board"><div class="board-top"><span>STACK YOUR SIGNAL</span><span>Drag ⠿ to reorder · Alt + ↑ / ↓ also works</span></div><div class="block-stack"></div><div class="board-bottom"><span>+ Connect another factor</span><div class="block-palette"></div></div><p class="block-hint">The stack runs from top to bottom. Changes stay a draft until you apply them.</p></section>'
_CSS = '''
:host{font-family:"Trebuchet MS",Arial,sans-serif;color:var(--st-text-color,#eef3f8)}
*{box-sizing:border-box;font-family:"Trebuchet MS",Arial,sans-serif}
.score-board{border:1px solid #31424d;border-radius:14px;padding:18px;background:radial-gradient(#293843 1px,transparent 1px) 0 0/14px 14px,#101a23}
.board-top{display:flex;justify-content:space-between;gap:12px;align-items:center;color:#9eb1be;font-size:11px;margin-bottom:18px;letter-spacing:.07em}
.board-top span:first-child{color:#d8e8f1;font-weight:bold}.board-top span:last-child{letter-spacing:0}
.block-stack{display:flex;flex-direction:column;gap:10px;padding:0 10px 6px}
.block{position:relative;border:1px solid var(--tone);border-left:5px solid var(--tone);background:#1a2934;border-radius:8px;padding:12px 12px;display:grid;grid-template-columns:24px 70px minmax(130px,1.5fr) 95px 60px 30px;gap:10px;align-items:center;box-shadow:0 3px 0 #08121b;min-height:62px}
.block::before{content:"";position:absolute;left:32px;top:-7px;width:38px;height:7px;border:1px solid var(--tone);border-bottom:0;border-radius:5px 5px 0 0;background:#1a2934}
.block.dragging{opacity:.45}.block.drop-target{outline:2px dashed #dce9ef;outline-offset:3px}
.grip{color:var(--tone);font-size:26px;cursor:grab;user-select:none;text-align:center}.grip:active{cursor:grabbing}
.block select,.block input[type=number]{width:100%;min-width:0;color:#ecf5fb;background:#101a23;border:1px solid #466071;border-radius:6px;padding:8px;font-size:12px}
.block select.factor{color:var(--tone);font-weight:bold}.block label{display:flex;align-items:center;gap:5px;white-space:nowrap;font-size:12px;color:#d9e4eb}.block input[type=checkbox]{accent-color:var(--tone)}
button{border:1px solid #456071;border-radius:6px;padding:7px;background:#1e303e;color:#d8e7f0;cursor:pointer;font-size:12px}button:hover{border-color:#d2e6ef;background:#2b4252}button:focus-visible,select:focus-visible,input:focus-visible,.grip:focus-visible{outline:2px solid #fff;outline-offset:2px}button:disabled{opacity:.35;cursor:default}.remove{padding:5px;color:#f3a3b4;font-size:20px;border:0;background:transparent}
.board-bottom{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:14px;color:#d4e1e9;font-size:12px}.block-palette{display:flex;gap:6px;flex-wrap:wrap}.block-palette button{border-color:var(--tone);color:var(--tone)}.block-hint{font-size:11px;line-height:1.5;color:#9eb1be;margin:14px 0 0}
@media(max-width:740px){.block{grid-template-columns:20px 55px minmax(80px,1fr) 58px;gap:7px;padding:10px 8px}.block .grip{grid-row:1 / span 2}.block .operation{grid-column:2;grid-row:1}.block .factor{grid-column:3 / span 2;grid-row:1}.block .weight{grid-column:2 / span 2;grid-row:2}.block label{grid-column:4;grid-row:2}.block .remove{position:absolute;right:-6px;top:-12px;background:#172832;border:1px solid #516b7e;border-radius:50%;width:26px;height:26px;line-height:12px}.score-board{padding:12px}.block select,.block input[type=number]{padding:7px 4px;font-size:11px}.board-top,.board-bottom{align-items:flex-start;flex-direction:column}.block-stack{padding:0 0 4px}}
'''
_JS = '''
export default function(component){
 const {parentElement,data,setStateValue}=component;
 const stack=parentElement.querySelector('.block-stack'),palette=parentElement.querySelector('.block-palette');
 const factors=data.factors,tones={predicted_return:'#51ddb6',momentum:'#b99bff',uncertainty:'#f4bc70',volatility:'#f486a8'};
 let blocks=structuredClone(data.blocks),dragged=null;
 const send=()=>setStateValue('edit',{blocks:structuredClone(blocks),revision:data.revision,nonce:Date.now()});
 const el=(tag,cls,text)=>{let e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
 const reorder=(from,to)=>{if(from===to)return;let item=blocks.splice(from,1)[0];blocks.splice(to,0,item);blocks[0].operator='+';send();};
 stack.replaceChildren();palette.replaceChildren();
 blocks.forEach((block,index)=>{
  const row=el('div','block');row.style.setProperty('--tone',tones[block.factor]);row.dataset.blockId=block.id;
  const grip=el('span','grip','⠿');grip.draggable=true;grip.tabIndex=0;grip.setAttribute('role','button');grip.setAttribute('aria-label','Move block '+(index+1)+' '+factors[block.factor]);
  grip.ondragstart=e=>{dragged=index;e.dataTransfer.setData('text/plain',String(index));e.dataTransfer.effectAllowed='move';row.classList.add('dragging');};
  grip.ondragend=()=>{dragged=null;row.classList.remove('dragging');stack.querySelectorAll('.drop-target').forEach(r=>r.classList.remove('drop-target'));};
  grip.onkeydown=e=>{if(e.altKey&&['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();reorder(index,Math.max(0,Math.min(blocks.length-1,index+(e.key==='ArrowUp'?-1:1))));}};
  row.ondragover=e=>{if(dragged!==null){e.preventDefault();row.classList.add('drop-target');e.dataTransfer.dropEffect='move';}};row.ondragleave=()=>row.classList.remove('drop-target');
  row.ondrop=e=>{e.preventDefault();if(dragged!==null)reorder(dragged,index);};row.append(grip);
  const op=el('select','operation');op.setAttribute('aria-label','Operation for block '+(index+1));
  ['+','-','*','/'].forEach(v=>{let o=el('option','',index===0?'Start':({'*':'×','/':'÷','-':'−','+':'+'}[v]));o.value=v;op.append(o);});op.value=block.operator;op.disabled=index===0;op.onchange=()=>{blocks[index].operator=op.value;send();};row.append(op);
  const factor=el('select','factor');factor.setAttribute('aria-label','Factor for block '+(index+1));Object.entries(factors).forEach(([key,label])=>{let o=el('option','',label);o.value=key;factor.append(o);});factor.value=block.factor;factor.onchange=()=>{blocks[index].factor=factor.value;send();};row.append(factor);
  const weight=el('input','weight');weight.type='number';weight.min='0';weight.max='10';weight.step='any';weight.value=block.weight;weight.setAttribute('aria-label','Weight for block '+(index+1));weight.onchange=()=>{let v=Number(weight.value);if(weight.value.trim()!==''&&Number.isFinite(v)&&v>=0&&v<=10){blocks[index].weight=v;send();}else{weight.value=block.weight;}};row.append(weight);
  const label=el('label','');const check=el('input','');check.type='checkbox';check.checked=block.use_k;check.setAttribute('aria-label','Multiply block '+(index+1)+' by k');check.onchange=()=>{blocks[index].use_k=check.checked;send();};label.append(check,document.createTextNode('× k'));row.append(label);
  const remove=el('button','remove','×');remove.setAttribute('aria-label','Remove block '+(index+1));remove.disabled=blocks.length===1;remove.onclick=()=>{blocks.splice(index,1);blocks[0].operator='+';send();};row.append(remove);stack.append(row);
 });
 Object.entries(factors).forEach(([factor,label])=>{const button=el('button','',label+' +');button.style.setProperty('--tone',tones[factor]);button.disabled=blocks.length>=data.max_blocks;button.onclick=()=>{blocks.push({id:'b'+Date.now().toString(36)+Math.random().toString(36).slice(2,5),factor,weight:1,operator:'+',use_k:false});send();};palette.append(button);});
 return ()=>{};
}
'''


@st.cache_resource
def _board_component():
    return st.components.v2.component('derivport_score_blocks', html=_HTML, css=_CSS, js=_JS)


def _native_key(identity, field):
    return f'block_native_{identity}_{field}'


def _set_blocks(items):
    normalized = normalize_blocks(items)
    st.session_state.block_items = normalized
    st.session_state.block_revision = st.session_state.get('block_revision', 0) + 1
    for block in normalized:
        for field in ('factor', 'weight', 'operator', 'use_k'):
            st.session_state[_native_key(block['id'], field)] = block[field]
    st.session_state.pop('block_error', None)


def _receive_blocks():
    packet = st.session_state.get('blocks_canvas', {}).get('edit')
    if not isinstance(packet, dict) or packet.get('revision') != st.session_state.block_revision:
        return
    try:
        _set_blocks(packet.get('blocks'))
    except FormulaError as exc:
        st.session_state.block_error = str(exc)


def _native_changed():
    items = []
    for block in st.session_state.block_items:
        items.append(dict(block, **{field: st.session_state[_native_key(block['id'], field)]
            for field in ('factor', 'weight', 'operator', 'use_k')}))
    _set_blocks(items)


def _move(identity, direction):
    _set_blocks(move_block(st.session_state.block_items, identity, direction))


def _remove(identity):
    _set_blocks([row for row in st.session_state.block_items if row['id'] != identity])


def _add(factor):
    _set_blocks([*st.session_state.block_items, dict(id='b' + uuid4().hex[:12],
        factor=factor, weight=1., operator='+', use_k=False)])


def _load_preset():
    _set_blocks(preset_blocks(st.session_state.block_preset))


def _apply_blocks(base):
    try:
        formula = compile_blocks(st.session_state.block_items)
        apply_rule(base, formula, st.session_state.get('valid_k', 1.))
        st.session_state.formula = formula
        st.session_state.valid_formula = formula
        st.session_state.block_applied = True
        st.session_state.pop('block_error', None)
    except FormulaError as exc:
        st.session_state.block_error = str(exc)


def render_blocks(base, ticker):
    """Call before the existing formula editor; draft blocks do not alter scores."""
    st.session_state.setdefault('block_items', preset_blocks())
    st.session_state.setdefault('block_revision', 0)
    with st.container(key='blocks_lab', border=True):
        st.markdown('### Build with blocks')
        controls, explanation = st.columns([1, 2], vertical_alignment='bottom')
        with controls:
            st.selectbox('Block starting point', list(PRESETS), key='block_preset')
            st.button('Load starting blocks', key='block_reset', on_click=_load_preset)
        with explanation:
            st.caption('Connect factors, weights and arithmetic. Drag the colored handles to change the order, then apply your stack to the algorithm. The text editor and backtest stay available below.')
        _board_component()(data={'blocks': st.session_state.block_items,
            'revision': st.session_state.block_revision, 'factors': FACTORS, 'max_blocks': MAX_BLOCKS},
            default={'edit': None}, key='blocks_canvas', on_edit_change=_receive_blocks)
        if st.session_state.get('block_error'):
            st.error(st.session_state.block_error)
        with st.expander('Keyboard block editor'):
            for position, block in enumerate(st.session_state.block_items):
                cols = st.columns([.8, 2, 1.2, .7, .5, .5, .5], vertical_alignment='bottom')
                identity = block['id']
                with cols[0]:
                    st.selectbox(f'Operation {position+1}', OPERATORS, index=OPERATORS.index(block['operator']),
                        disabled=position == 0, key=_native_key(identity, 'operator'), on_change=_native_changed)
                with cols[1]:
                    st.selectbox(f'Factor {position+1}', list(FACTORS), index=list(FACTORS).index(block['factor']),
                        format_func=FACTORS.get, key=_native_key(identity, 'factor'), on_change=_native_changed)
                with cols[2]:
                    st.number_input(f'Weight {position+1}', 0., 10., float(block['weight']), .05,
                        key=_native_key(identity, 'weight'), on_change=_native_changed)
                with cols[3]:
                    st.checkbox(f'× k {position+1}', value=block['use_k'], key=_native_key(identity, 'use_k'), on_change=_native_changed)
                for column, label, direction in [(cols[4], '↑', -1), (cols[5], '↓', 1)]:
                    with column:
                        st.button(label, key=f'block_move_{identity}_{direction}',
                            disabled=position == (0 if direction == -1 else len(st.session_state.block_items)-1),
                            on_click=_move, args=(identity, direction), help='Move block')
                with cols[6]:
                    st.button('×', key=f'block_remove_{identity}', disabled=len(st.session_state.block_items) == 1,
                        on_click=_remove, args=(identity,), help='Remove block')
            cols = st.columns(4)
            for col, factor in zip(cols, FACTORS):
                with col:
                    st.button('+ ' + FACTORS[factor], key='block_add_' + factor,
                        on_click=_add, args=(factor,), disabled=len(st.session_state.block_items) >= MAX_BLOCKS)
        try:
            formula = compile_blocks(st.session_state.block_items)
            st.session_state.block_formula = formula
            k = st.session_state.get('valid_k', 1.)
            preview = apply_rule(base, formula, k)
            st.latex('S = ' + formula_to_latex(formula))
            row = preview.loc[ticker]
            st.caption(f'Draft at current inputs: {ticker} score {row.score:+.3f}, rank {row.student_rank:g} of {len(preview)} · k = {k:g}. This score is not a return forecast. Multiplication/division can change its units.')
            apply, save_json, save_python = st.columns([1.4, 1, 1])
            with apply:
                st.button('Apply blocks to algorithm', key='block_apply', type='primary',
                    on_click=_apply_blocks, args=(base,), width='stretch')
            payload = {'schema_version': 1, 'blocks': st.session_state.block_items, 'formula': formula, 'k': k}
            with save_json:
                st.download_button('Save blocks', json.dumps(payload, indent=2, allow_nan=False),
                    'derivport_blocks.json', 'application/json', key='block_download_json', width='stretch')
            with save_python:
                st.download_button('Export block algorithm', python_algorithm(st.session_state.block_items),
                    'derivport_block_algorithm.py', 'text/x-python', key='block_download_python', width='stretch')
        except FormulaError as exc:
            st.error(f'Draft cannot run: {exc} Your active algorithm is unchanged.')
        if st.session_state.pop('block_applied', False):
            st.success('Block stack applied. The formula editor and historical backtest now use it.')
