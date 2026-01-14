#!/usr/bin/env python3
"""
m6502_convert_v2.py

v2 improvements:
- Better normalization of instruction-macro mnemonics (LDAI/LDXI/LDYI, STADY, ADCDY, CMPDY, etc.)
- Handles LABEL: BLOCK n  -> LABEL: .res n
- Handles ADR expr and ADR(expr) -> .word expr
- Converts XWD left,right -> .word left,right (placeholder that assembles)
- Expands REPEAT n,<body> (simple form)
- Converts IRPC "STRING",<...> into .byte ASCII bytes for that string (matches how this source emits text)
- Converts EXP <expr> into .byte <expr> (evaluates to literal if possible, else keeps symbol)
- Comments out leftover listing/meta directives and stray text lines

Usage:
  python3 m6502_convert_v2.py m6502.asm -o out.asm
  python3 m6502_convert_v2.py m6502.asm -o out.asm --report unresolved.json
"""

from __future__ import annotations
import re, ast, argparse, json
from pathlib import Path

IDENT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_%$@."

def nl(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def parse_balanced_angle(s: str, start: int):
    assert s[start] == "<"
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return s[start+1:i], i+1
        i += 1
    raise ValueError(f"Unbalanced <...> at {start}")

def parse_conditional_block(s: str, start: int):
    assert s[start] == "<"
    depth = 1
    i = start + 1
    while i < len(s):
        c = s[i]
        if c == "<":
            line_start = s.rfind("\n", 0, i) + 1
            prefix = s[line_start:i]
            if re.match(r'^\s*(?:[A-Z0-9_%$]+:\s*)?(IFE|IFN|IF1|IF2)\b[^<]*,\s*$', prefix, re.IGNORECASE):
                depth += 1
                i += 1
                continue
            _, i = parse_balanced_angle(s, i)
            continue
        if c == ">":
            depth -= 1
            if depth == 0:
                return s[start+1:i], i+1
        i += 1
    raise ValueError(f"Unbalanced conditional <...> at {start}")

def extract_config_overrides(src_text: str):
    config = {}
    sym = {}
    for line in nl(src_text).splitlines():
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        if re.match(r'^(?:[A-Z0-9_%$]+:\s*)?(IFE|IFN|IF1|IF2)\b', code, re.IGNORECASE):
            break
        m = re.match(r'^([A-Z0-9_%$]+)\s*(==|=)\s*(.+)$', code)
        if not m:
            continue
        name = m.group(1)
        if name not in ("REALIO", "EXTIO"):
            continue
        try:
            sym[name] = safe_eval_int(m.group(3), sym)
            config[name] = sym[name]
        except Exception:
            continue
    return config

def preprocess_numbers(expr: str, radix: int = 10):
    def repl_oct(m):
        return str(int(m.group(1), 8))
    def repl_dec(m):
        return m.group(1)
    def repl_hex(m):
        return f"${m.group(1).upper()}"
    expr = re.sub(r"\^O([0-7]+)", repl_oct, expr)
    expr = re.sub(r"\^D([0-9]+)", repl_dec, expr)
    expr = re.sub(r"\^X([0-9A-F]+)", repl_hex, expr)
    if radix == 8:
        def repl_bare_oct(m):
            return str(int(m.group(1), 8))
        expr = re.sub(r"(?<![A-Z0-9_%$])([0-7]+)(?![A-Z0-9_%$])", repl_bare_oct, expr)
    return expr

ANGLE_OP_PAT = re.compile(r"<<([^<>]+?)\s*([&/])\s*([^<>]+?)>")

def preprocess_angle_ops(expr: str):
    out = expr
    for _ in range(500):
        m = ANGLE_OP_PAT.search(out)
        if not m:
            break
        inner, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        rep = f"(({inner}) & ({rhs}))" if op == "&" else f"(({inner}) // ({rhs}))"
        out = out[:m.start()] + rep + out[m.end():]
    out = out.replace("<<", "(").replace(">", ")")
    return out

def safe_eval_int(expr: str, sym: dict):
    expr = preprocess_numbers(expr.strip())
    expr = preprocess_angle_ops(expr)
    expr = re.sub(r"\$([0-9A-F]+)", r"0x\1", expr)
    expr = re.sub(r"(?<!/)/(?!/)", "//", expr)  # integer division
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)): return int(n.value)
            raise ValueError
        if isinstance(n, ast.Constant): return int(n.n)
        if isinstance(n, ast.Name): return int(sym.get(n.id, 0))
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.USub): return -v
            if isinstance(n.op, ast.UAdd): return v
            raise ValueError
        if isinstance(n, ast.BinOp):
            a = _eval(n.left); b = _eval(n.right); op = n.op
            if isinstance(op, ast.Add): return a + b
            if isinstance(op, ast.Sub): return a - b
            if isinstance(op, ast.Mult): return a * b
            if isinstance(op, (ast.Div, ast.FloorDiv)): return a // b
            if isinstance(op, ast.Mod): return a % b
            if isinstance(op, ast.BitAnd): return a & b
            if isinstance(op, ast.BitOr): return a | b
            if isinstance(op, ast.BitXor): return a ^ b
            if isinstance(op, ast.LShift): return a << b
            if isinstance(op, ast.RShift): return a >> b
            raise ValueError
        raise ValueError
    return _eval(node)

def split_args(arg_str: str):
    args, cur = [], []
    depth_angle = 0
    in_quote = False
    for c in arg_str:
        if c == '"':
            in_quote = not in_quote
            cur.append(c)
        elif not in_quote:
            if c == "<":
                depth_angle += 1; cur.append(c)
            elif c == ">":
                depth_angle = max(0, depth_angle - 1); cur.append(c)
            elif c == "," and depth_angle == 0:
                args.append("".join(cur).strip()); cur = []
            else:
                cur.append(c)
        else:
            cur.append(c)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args

def substitute_params(body: str, params, args):
    out = body
    for p, a in zip(params, args):
        out = re.sub(rf'(?<![{IDENT_CHARS}]){re.escape(p)}(?![{IDENT_CHARS}])', a, out)
    return out

def extract_defines(full: str):
    pat = re.compile(r'(?m)^\s*DEFINE\s+([A-Z0-9_%$]+)\s*(\([^)]*\))?\s*,\s*<')
    defines, spans = {}, []
    for m in pat.finditer(full):
        name = m.group(1)
        params = m.group(2)
        params = [p.strip() for p in params.strip()[1:-1].split(",") if p.strip()] if params else []
        body, end_pos = parse_balanced_angle(full, m.end() - 1)
        defines[name] = {"params": params, "body": body, "start": m.start(), "end": end_pos}
        spans.append((m.start(), end_pos))
    return defines, spans

def remove_spans(full: str, spans):
    spans = sorted(spans)
    out, last = [], 0
    for a, b in spans:
        out.append(full[last:a]); last = b
    out.append(full[last:])
    return "".join(out)

def convert_defines_to_macros(text: str):
    pat = re.compile(r'(?m)^\s*DEFINE\s+([A-Z0-9_%$]+)\s*(\([^)]*\))?\s*,\s*<')
    out = []
    last = 0
    defines = {}
    reserved_param_names = {"A", "X", "Y"}
    for m in pat.finditer(text):
        name = m.group(1)
        params = m.group(2)
        params = [p.strip() for p in params.strip()[1:-1].split(",") if p.strip()] if params else []
        body, end_pos = parse_balanced_angle(text, m.end() - 1)
        param_map = {}
        safe_params = []
        used = set()
        for p in params:
            new_p = p
            if p.upper() in reserved_param_names:
                base = f"ARG_{p.upper()}"
                new_p = base
                suffix = 2
                while new_p in used or new_p in params:
                    new_p = f"{base}_{suffix}"
                    suffix += 1
            param_map[p] = new_p
            safe_params.append(new_p)
            used.add(new_p)
        params = safe_params
        orig_text = text[m.start():end_pos]
        defines[name] = {"params": params, "body": body, "start": m.start(), "end": end_pos, "orig": orig_text}

        indent = re.match(r"[ \t]*", text[m.start():]).group(0)
        macro_head = f"{indent}.macro {name}"
        if params:
            if len(params) == 1:
                macro_head += " " + params[0] + ","
            else:
                macro_head += " " + ", ".join(params)

        body_text = body
        if body_text.startswith("\n"):
            body_text = body_text[1:]
        if name in ("DCI", "DCE"):
            body_lines = body_text.splitlines()
            body_lines = [
                ln_ for ln_ in body_lines
                if not re.match(r'^\s*Q\s*=\s*Q\s*\+\s*\d+\s*$', ln_)
            ]
            body_text = "\n".join(body_lines)
            body_text = re.sub(r'\bDC\(([^)]+)\)', r'DC \1', body_text)
        for old, new in param_map.items():
            if old != new:
                body_text = re.sub(
                    rf'(?<![A-Za-z0-9_%$]){re.escape(old)}(?![A-Za-z0-9_%$])',
                    new,
                    body_text,
                )
        if "%Q" in body_text:
            body_text = body_text.replace("%Q", "__Q")
            body_lines = body_text.splitlines()
            body_indent = ""
            for ln_ in body_lines:
                if ln_.strip():
                    body_indent = re.match(r"[ \t]*", ln_).group(0)
                    break
            body_lines.insert(0, f"{body_indent}.local __Q")
            body_text = "\n".join(body_lines)
        orig_lines = orig_text.splitlines()
        commented = []
        for ln_ in orig_lines:
            if ln_.strip():
                commented.append(f"{indent}; {ln_.rstrip()}")
            else:
                commented.append(f"{indent};")
        comment_block = "\n".join(commented) + "\n"
        macro_text = f"{comment_block}{macro_head}\n{body_text.rstrip()}\n{indent}.endmacro\n"

        out.append(text[last:m.start()])
        out.append(macro_text)
        last = end_pos
    out.append(text[last:])
    return "".join(out), defines

# ---- conditionals ----
COND_PAT = re.compile(r'(?m)^[ \t]*(?:([A-Z0-9_%$]+):[ \t]*)?(IFE|IFN|IF1|IF2|IFDEF|IFNDEF)\b[ \t]*([^\n]*?)\s*,\s*<', re.IGNORECASE)

def expand_conditionals(text: str, sym: dict, pass_num: int = 2, max_iters: int = 8000):
    out = text
    for _ in range(max_iters):
        m = COND_PAT.search(out)
        if not m:
            break
        label, kind, expr = m.group(1), m.group(2).upper(), m.group(3).strip()
        block, end_pos = parse_conditional_block(out, m.end() - 1)
        if kind in ("IF1", "IF2"):
            include = (pass_num == 1)
        elif kind in ("IFDEF", "IFNDEF"):
            key = expr.strip()
            is_def = key in sym
            include = is_def if kind == "IFDEF" else (not is_def)
        else:
            try:
                val = safe_eval_int(expr, sym)
            except Exception:
                val = 0  # keep block (safer)
            include = (val == 0) if kind == "IFE" else (val != 0)
        if include and label and block.strip():
            bl = block.splitlines()
            for i, ln_ in enumerate(bl):
                if ln_.strip():
                    bl[i] = f"{label}: {ln_.lstrip()}"
                    break
            block = "\n".join(bl)
        out = out[:m.start()] + (block if include else "") + out[end_pos:]
    return out

def strip_conditionals(text: str, max_iters: int = 8000):
    out = text
    for _ in range(max_iters):
        m = COND_PAT.search(out)
        if not m:
            break
        _block, end_pos = parse_conditional_block(out, m.end() - 1)
        out = out[:m.start()] + out[end_pos:]
    return out

def convert_conditionals_to_ca65(text: str, max_iters: int = 8000):
    out = text
    for _ in range(max_iters):
        m = COND_PAT.search(out)
        if not m:
            break
        label, kind, expr = m.group(1), m.group(2).upper(), m.group(3).strip()
        block, end_pos = parse_conditional_block(out, m.end() - 1)
        block = convert_conditionals_to_ca65(block, max_iters=max_iters)
        if label and block.strip():
            bl = block.splitlines()
            for i, ln_ in enumerate(bl):
                if ln_.strip():
                    bl[i] = f"{label}: {ln_.lstrip()}"
                    break
            block = "\n".join(bl)

        line_start = out.rfind("\n", 0, m.start()) + 1
        leading_ws = re.match(r"[ \t]*", out[line_start:]).group(0)

        comment = ""
        if kind == "IFE":
            cond = f"({expr}) = 0"
        elif kind == "IFN":
            cond = f"({expr}) <> 0"
        elif kind == "IFDEF":
            cond = f".defined({expr})"
        elif kind == "IFNDEF":
            cond = f".not .defined({expr})"
        else:  # IF1/IF2
            cond = "1"
            comment = f" ; {kind} (pass1 only in original)"

        parts = [f"{leading_ws}.if {cond}{comment}"]
        if block:
            parts.append(block.rstrip("\n"))
        parts.append(f"{leading_ws}.endif")
        repl = "\n".join(parts) + "\n"
        out = out[:m.start()] + repl + out[end_pos:]
    return out

# ---- REPEAT ----
REPEAT_PAT = re.compile(r'(?m)^\s*(?:([A-Z0-9_%$]+):\s*)?REPEAT\s+([^,]+)\s*,\s*<')

def expand_repeat(text: str, sym: dict, max_iters: int = 2000):
    out = text
    for _ in range(max_iters):
        m = REPEAT_PAT.search(out)
        if not m:
            break
        label = m.group(1)
        count_expr = m.group(2).strip()
        body, end_pos = parse_balanced_angle(out, m.end() - 1)
        try:
            n = safe_eval_int(count_expr, sym)
        except Exception:
            n = 0
        block = "\n".join([body] * max(0, n))
        if label and block.strip():
            bl = block.splitlines()
            for i, ln_ in enumerate(bl):
                if ln_.strip():
                    bl[i] = f"{label}: {ln_.lstrip()}"
                    break
            block = "\n".join(bl)
        out = out[:m.start()] + block + out[end_pos:]
    return out

def convert_repeat_to_ca65(text: str, max_iters: int = 2000):
    out = text
    for _ in range(max_iters):
        m = REPEAT_PAT.search(out)
        if not m:
            break
        label = m.group(1)
        count_expr = m.group(2).strip()
        body, end_pos = parse_balanced_angle(out, m.end() - 1)
        body = convert_repeat_to_ca65(body, max_iters=max_iters)
        line_start = out.rfind("\n", 0, m.start()) + 1
        leading_ws = re.match(r"[ \t]*", out[line_start:]).group(0)
        repeat_line = f"{leading_ws}.repeat {count_expr}"
        if label:
            repeat_line = f"{leading_ws}{label}: {repeat_line.lstrip()}"
        parts = [repeat_line]
        if body.strip():
            parts.append(body.rstrip("\n"))
        parts.append(f"{leading_ws}.endrepeat")
        repl = "\n".join(parts) + "\n"
        out = out[:m.start()] + repl + out[end_pos:]
    return out

# ---- DEFINE expansion ----
def expand_define_macros(text: str, defines: dict, max_depth: int = 120):
    uniq = 0
    lines = nl(text).splitlines()

    def expand_one(line: str):
        nonlocal uniq
        raw = line
        code, *comment = raw.split(";", 1)
        cmt = ";" + comment[0] if comment else ""
        code = code.rstrip()
        if not code.strip():
            return [raw]

        leading_ws = re.match(r"^\s*", code).group(0)
        s = code.lstrip()

        label, rest = "", s
        m = re.match(r'^([A-Z0-9_%$]+):\s*(.*)$', rest)
        if m:
            label = m.group(1) + ":"
            rest = m.group(2)

        m2 = re.match(r'^([A-Z0-9_%$]+)\b(.*)$', rest.strip())
        if not m2:
            return [raw]
        op = m2.group(1)
        tail = m2.group(2).strip()

        if op not in defines:
            return [raw]

        args = split_args(tail) if tail else []
        expanded = substitute_params(defines[op]["body"], defines[op]["params"], args)

        uniq += 1
        expanded = expanded.replace("%Q", f"__{op}_{uniq}")

        out_lines = [ln_ for ln_ in nl(expanded).splitlines()]
        while out_lines and out_lines[0].strip() == "":
            out_lines.pop(0)
        if not out_lines:
            return [raw]

        if label:
            out_lines[0] = f"{leading_ws}{label} {out_lines[0].lstrip()}"
        else:
            out_lines[0] = leading_ws + out_lines[0]
        if cmt:
            out_lines[0] += " " + cmt
        for i in range(1, len(out_lines)):
            out_lines[i] = leading_ws + out_lines[i]
        return out_lines

    work = lines
    for _ in range(max_depth):
        changed = False
        new_work = []
        for ln_ in work:
            ex = expand_one(ln_)
            if len(ex) != 1 or ex[0] != ln_:
                changed = True
            new_work.extend(ex)
        work = new_work
        if not changed:
            break
    return "\n".join(work) + "\n"

# ---- IRPC "STRING",<...> -> .byte ASCII ----
IRPC2_PAT = re.compile(r'(?m)^(\s*(?:[A-Z0-9_%$]+:\s*)?)IRPC\s+"([^"]*)"\s*,\s*<')
IRPC_PAT = re.compile(r'(?m)^\s*(?:([A-Z0-9_%$]+):\s*)?IRPC\s+([A-Z0-9_%$]+)\s*,\s*<')

def expand_irpc_string_only(text: str):
    out = text
    for _ in range(5000):
        m = IRPC2_PAT.search(out)
        if not m:
            break
        prefix = m.group(1)
        s = m.group(2)
        _body, end_pos = parse_balanced_angle(out, m.end() - 1)
        bytes_list = ",".join(f"${ord(ch) & 0xFF:02X}" for ch in s)
        repl = f"{prefix}.byte {bytes_list}"
        out = out[:m.start()] + repl + out[end_pos:]
    return out

def convert_irpc_to_ca65(text: str, max_iters: int = 2000):
    out = text
    uniq = 0
    for _ in range(max_iters):
        m = IRPC_PAT.search(out)
        if not m:
            break
        label = m.group(1)
        var = m.group(2)
        body, end_pos = parse_balanced_angle(out, m.end() - 1)
        body = convert_irpc_to_ca65(body, max_iters=max_iters)
        uniq += 1
        idx = f"__IRPCI_{uniq}"
        new_lines = []
        for ln_ in nl(body).splitlines():
            if re.match(r'^\s*IFDIF\b', ln_, re.IGNORECASE):
                pat = re.compile(
                    rf'^\s*IFDIF\s+<\s*{re.escape(var)}\s*>\s*<\s*"\s*>\s*,\s*<\s*EXP\s+"{re.escape(var)}"\s*>\s*$',
                    re.IGNORECASE,
                )
                if pat.match(ln_.strip()):
                    leading_ws = re.match(r"[ \t]*", ln_).group(0)
                    new_lines.append(f"{leading_ws}.if .not .match (.strat ({var}, {idx}), '\"')")
                    new_lines.append(f"{leading_ws}    .byte .strat ({var}, {idx})")
                    new_lines.append(f"{leading_ws}.endif")
                    continue
            new_lines.append(ln_)
        body = "\n".join(new_lines)
        line_start = out.rfind("\n", 0, m.start()) + 1
        leading_ws = re.match(r"[ \t]*", out[line_start:]).group(0)
        repeat_line = f"{leading_ws}.repeat .strlen ({var}), {idx}"
        if label:
            repeat_line = f"{leading_ws}{label}: {repeat_line.lstrip()}"
        parts = [repeat_line]
        if body.strip():
            parts.append(body.rstrip("\n"))
        parts.append(f"{leading_ws}.endrepeat")
        repl = "\n".join(parts) + "\n"
        out = out[:m.start()] + repl + out[end_pos:]
    return out

# ---- EXP <expr> -> .byte <expr> ----
def expand_exp_lines(text: str, sym: dict):
    out_lines = []
    for ln_ in nl(text).splitlines():
        code, *comment = ln_.split(";", 1)
        cmt = ";" + comment[0] if comment else ""
        m = re.match(r'^(\s*(?:[A-Z0-9_%$]+:\s*)?)EXP\s+(.+?)\s*$', code)
        if m:
            prefix = m.group(1)
            arg = m.group(2).strip()
            try:
                v = safe_eval_int(arg, sym)
                rep = f"{prefix}.byte ${v & 0xFF:02X}"
            except Exception:
                rep = f"{prefix}.byte {preprocess_numbers(arg)}"
            out_lines.append(rep + ((" " + cmt) if cmt else ""))
        else:
            out_lines.append(ln_)
    return "\n".join(out_lines) + "\n"

# ---- instruction macro normalization ----
BASE_OPS = {
    "ADC","AND","ASL","BCC","BCS","BEQ","BIT","BMI","BNE","BPL","BRK","BVC","BVS",
    "CLC","CLD","CLI","CLV","CMP","CPX","CPY","DEC","DEX","DEY","EOR","INC","INX","INY",
    "JMP","JSR","LDA","LDX","LDY","LSR","NOP","ORA","PHA","PHP","PLA","PLP",
    "ROL","ROR","RTI","RTS","SBC","SEC","SED","SEI","STA","STX","STY","TAX","TAY","TSX","TXA","TXS","TYA"
}
SUFFIX_MAP = {
    "IND": "({arg})",
    "IY":  "({arg}),Y",
    "IX":  "({arg},X)",
    "AY":  "{arg},Y",
    "AX":  "{arg},X",
    "ZY":  "{arg},Y",
    "ZX":  "{arg},X",
    "I":   "#{arg}",
    "Z":   "{arg}",
    "A":   "{arg}",
    "DY":  "({arg}),Y",  # appears as LDADY/STADY/etc
}

def normalize_instruction_macros(text: str):
    out = []
    for ln_ in nl(text).splitlines():
        code, *comment = ln_.split(";", 1)
        cmt = ";" + comment[0] if comment else ""
        s = code.rstrip()
        if not s.strip():
            out.append(ln_)
            continue

        leading_ws = re.match(r"^\s*", s).group(0)
        rest = s.strip()

        label = ""
        mlabel = re.match(r'^([A-Z0-9_%$]+):\s*(.*)$', rest)
        if mlabel:
            label = mlabel.group(1) + ":"
            rest = mlabel.group(2).strip()

        m = re.match(r'^([A-Z0-9_%$]+)\b\s*(.*)$', rest)
        if not m:
            out.append(ln_)
            continue

        op0 = m.group(1)
        tail = m.group(2).strip()

        if op0 == "JMPD":
            rebuilt = leading_ws + (f"{label} " if label else "") + f"JMP ({tail})"
            if cmt:
                rebuilt += " " + cmt
            out.append(rebuilt)
            continue

        converted = None
        # Try all bases; op0 is like "LDAI" => base "LDA", suffix "I"
        for base in BASE_OPS:
            if op0.startswith(base):
                suf = op0[len(base):]
                if suf in SUFFIX_MAP:
                    converted = f"{base} {SUFFIX_MAP[suf].format(arg=tail)}".rstrip()
                    break

        if converted is None:
            out.append(ln_)
            continue

        rebuilt = leading_ws + (f"{label} " if label else "") + converted
        if cmt:
            rebuilt += " " + cmt
        out.append(rebuilt)
    return "\n".join(out) + "\n"

# ---- directive normalization ----

def comment_if_plain_text_line(line: str) -> str:
    """
    Some comment paragraphs in the original source are not prefixed by ';'
    after macro expansion. If a line does NOT look like an assembly statement,
    comment it out.

    Heuristic:
      - Ignore lines that already start with ';'
      - If it matches label/opcode/directive/equate-like patterns -> keep
      - Otherwise, if it contains mostly letters/spaces/punctuation and no digits,
        comment it.
    """
    if line.lstrip().startswith(";"):
        return line

    code = line.split(";", 1)[0].rstrip("\n")
    t = code.strip()
    if not t:
        return line

    # assembly-ish patterns we keep
    if re.match(r'^[A-Za-z0-9_%$]+:\s*', t):  # label:
        return line
    if re.match(r'^\.[A-Za-z]\w*', t):     # .directive
        return line
    if re.match(r'^[A-Za-z0-9_%$]+\s*(==|=)\s*', t):  # equate
        if t.rstrip().endswith(".") and re.fullmatch(r"[A-Z0-9\"'\(\)\[\]\-\.,/:^#+=<>$* \t]+", t):
            return "; " + t
        return line

    if t.startswith("DC") or t.startswith("DT"):
        return line

    def has_opcode_punct(text: str) -> bool:
        return bool(re.search(r"[#$()\[\],]", text))

    def looks_like_prose(text: str) -> bool:
        return re.fullmatch(r"[A-Z0-9\"'\(\)\[\]\-\.,/:^#+=<>$* \t]+", text) is not None

    # Indented prose blocks tend to be tabbed and lack opcode/addressing punctuation.
    if re.match(r'^\s+', line) and len(t.split()) >= 3 and not has_opcode_punct(t) and looks_like_prose(t):
        return "; " + t

    # opcode / pseudo that might start a line
    first = t.split()[0]
    if first in BASE_OPS or first in ("ORG","BLOCK","ADR","XWD","DEFINE","IFE","IFN","IF1","IF2","REPEAT","IRPC","EXP","IFDIF","DC","DT","RADIX"):
        if len(t.split()) >= 3 and not has_opcode_punct(t) and looks_like_prose(t):
            return "; " + t
        if first in BASE_OPS and len(t.split()) > 2 and not re.search(r"[#$()]", t):
            return "; " + t
        return line

    # Treat prose-like lines as comments, even if they contain digits (dates, model names).
    if len(t.split()) >= 2 and re.fullmatch(r"[A-Z0-9\"'\(\)\[\]\-\.,/:^#+=<>$* \t]+", t):
        return "; " + t

    if len(t.split()) >= 3 and re.match(r"\d+/\d+/\d+", t):
        return "; " + t

    if re.fullmatch(r"[A-Z][A-Z0-9_$%]*\.?", t) and t.rstrip(".") not in BASE_OPS:
        return "; " + t

    if t == "*":
        return "; " + t
    if t == "%":
        return "; " + t

    if re.search(r"[A-Z]", t):
        return "; " + t

    return line

def normalize_directives(text: str, defines: dict | None = None):
    out = []
    defined_macros = set()
    seen_names = set()
    pending_label = None
    defined_names = set()
    cond_stack = []
    macro_depth = 0
    preamble_emitted = False
    sym_values = {}
    cur_radix = 10
    redefined = set()
    counts = {}
    for ln_ in nl(text).splitlines():
        code = ln_.split(";", 1)[0].strip()
        if not code:
            continue
        m = re.match(r'^\s*\.macro\s+([A-Za-z0-9_%$]+)\b', code, re.IGNORECASE)
        if m:
            defined_macros.add(m.group(1).upper())
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*([A-Za-z0-9_%$]+)\s*(==|=)\s*', code)
        if m:
            counts[m.group(2)] = counts.get(m.group(2), 0) + 1
            continue
        m = re.match(r'^\s*([A-Za-z0-9_%$]+)\s*(==|=)\s*', code)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    redefined = {k for k, v in counts.items() if v > 1}

    def replace_char_literals(expr: str) -> str:
        def repl(m):
            return f"'{m.group(1)}'"
        return re.sub(r'"([^"\\])"', repl, expr)

    def normalize_symbol_tokens(s: str) -> str:
        out = []
        i = 0
        in_quote = None
        while i < len(s):
            c = s[i]
            if c in ("'", '"'):
                if in_quote is None:
                    in_quote = c
                elif in_quote == c:
                    in_quote = None
                out.append(c)
                i += 1
                continue
            if in_quote:
                out.append(c)
                i += 1
                continue
            m = re.match(r"[A-Za-z0-9_%$]+", s[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            tok = m.group(0)
            if tok.startswith("$") and re.fullmatch(r"\$[0-9A-F]+", tok):
                out.append(tok)
            elif "$" in tok:
                out.append(tok.replace("$", "DOLLAR").upper())
            else:
                out.append(tok.upper())
            i += len(tok)
        return "".join(out)

    def replace_bang_or(s: str) -> str:
        out = []
        in_quote = None
        for c in s:
            if c in ("'", '"'):
                if in_quote is None:
                    in_quote = c
                elif in_quote == c:
                    in_quote = None
                out.append(c)
                continue
            if in_quote:
                out.append(c)
                continue
            out.append("|" if c == "!" else c)
        return "".join(out)

    def bytes_for_string(s: str, set_high_bit_last: bool) -> str:
        vals = [ord(ch) & 0xFF for ch in s]
        if set_high_bit_last and vals:
            vals[-1] |= 0x80
        return ",".join(f"${v:02X}" for v in vals) if vals else "$00"

    def is_symbolic_expr(expr: str) -> bool:
        cleaned = re.sub(r"\$[0-9A-F]+", "", expr, flags=re.I)
        cleaned = re.sub(r"0x[0-9A-F]+", "", cleaned, flags=re.I)
        return re.search(r"\b[A-Z][A-Z0-9_%]*\b", cleaned) is not None

    def clamp_byte_expr(expr: str) -> str:
        if is_symbolic_expr(expr):
            return expr
        try:
            val = safe_eval_int(expr, sym_values)
        except Exception:
            return expr
        return f"${val & 0xFF:02X}"

    def emit(line: str, consume_label: bool = True):
        nonlocal pending_label
        if pending_label and consume_label:
            if re.match(r'^\s*\.', line) or re.match(r'^\s*[A-Z0-9_%$]', line):
                line = f"{pending_label}: {line.lstrip()}"
            else:
                line = f"{pending_label}: {line}"
            pending_label = None
        out.append(line)

    def fix_trailing_commas_in_op(line: str) -> str:
        m = re.match(r'^(\s*(?:[A-Z0-9_%$]+:\s*)?)([A-Z]{3})\b(.*)$', line)
        if not m:
            return line
        op = m.group(2)
        if op not in BASE_OPS:
            return line
        rest = re.sub(r',\s*$', '', m.group(3))
        return f"{m.group(1)}{op}{rest}"

    def clamp_immediate_expr(line: str) -> str:
        def repl(m):
            expr = m.group(1)
            if re.fullmatch(r"\$[0-9A-F]+", expr) or re.fullmatch(r"[0-9]+", expr):
                return "#" + expr
            if re.fullmatch(r"[A-Z0-9_%$]+", expr):
                return f"#(({expr}) & $FF)"
            if re.fullmatch(r"'[^']'", expr):
                return "#" + expr
            if not re.search(r"[-+*/&|^()]", expr):
                return "#" + expr
            return f"#(({expr}) & $FF)"
        return re.sub(r"#([A-Z0-9_%$'*/&|^()+-:]+)", repl, line)

    def fix_zero_page_minus_one(line: str) -> str:
        def repl(m):
            name = m.group(1)
            idx = m.group(2)
            if sym_values.get(name) == 0:
                return f"$FF,{idx}"
            return m.group(0)
        return re.sub(r"\b([A-Za-z0-9_%$]+)-1,([XY])\b", repl, line)

    def emit_redefine(name: str, expr_out: str, cmt: str, consume_label: bool = False):
        needs_temp = bool(re.search(rf'(?<![A-Za-z0-9_%$]){re.escape(name)}(?![A-Za-z0-9_%$])', expr_out))
        if name in defined_names:
            if needs_temp:
                tmp = f"__{name}_TMP"
                emit(f"    .define {tmp} {name}", consume_label=consume_label)
                emit(f"    .undef {name}", consume_label=False)
                expr_out = re.sub(
                    rf'(?<![A-Za-z0-9_%$]){re.escape(name)}(?![A-Za-z0-9_%$])',
                    tmp,
                    expr_out,
                )
                emit(f"    .define {name} {expr_out}" + ((" " + cmt) if cmt else ""), consume_label=False)
                emit(f"    .undef {tmp}", consume_label=False)
                defined_names.add(name)
                return
            emit(f"    .undef {name}", consume_label=consume_label)
        emit(f"    .define {name} {expr_out}" + ((" " + cmt) if cmt else ""), consume_label=consume_label)
        defined_names.add(name)

    predefine = [f"    .define {name} 0" for name in sorted(redefined) if name != "Q"]

    allowed_dot = {
        "org","byte","word","res","include","segment","proc","endproc","export","import",
        "scope","endscope","setcpu","ifdef","ifndef","if","else","elseif","endif","repeat","endrepeat",
        "macro","endmacro","local","endlocal","assert","align","asciiz","ascii","bank","define","undef",
        "exitmacro","set",
    }

    for ln_ in nl(text).splitlines():
        if defines:
            code = ln_.split(";", 1)[0]
            rest = code.strip()
            mlabel = re.match(r'^([A-Za-z0-9_%$]+):\s*(.*)$', rest)
            if mlabel:
                rest = mlabel.group(2).strip()
            mcall = re.match(r'^([A-Za-z0-9_%$]+)\b', rest)
            if mcall and mcall.group(1).upper() in defines:
                pass
            else:
                ln_ = comment_if_plain_text_line(ln_)
        else:
            ln_ = comment_if_plain_text_line(ln_)
        code, *comment = ln_.split(";", 1)
        cmt = ";" + comment[0] if comment else ""
        s = code.rstrip()
        if not s.strip():
            out.append(ln_)
            continue
        if defines and macro_depth == 0:
            rest = s.strip()
            mlabel = re.match(r'^([A-Za-z0-9_%$]+):\s*(.*)$', rest)
            if mlabel:
                rest = mlabel.group(2).strip()
            mcall = re.match(r'^([A-Za-z0-9_%$]+)\b', rest)
            if mcall:
                op = mcall.group(1).upper()
                if op in defines:
                    params = defines[op].get("params") or []
                    sig = f"DEFINE {op}"
                    if params:
                        sig += "(" + ",".join(params) + ")"
                    parts = []
                    if cmt:
                        parts.append(cmt.lstrip(";").strip())
                    parts.append(sig)
                    cmt = "; " + " ; ".join(parts)
        if not preamble_emitted:
            for ln_ in predefine:
                if ln_.split()[1] == "Q":
                    continue
                out.append(ln_)
                defined_names.add(ln_.split()[1])
            preamble_emitted = True
        s = replace_char_literals(s)
        s = fix_trailing_commas_in_op(s)
        s = re.sub(r'(^|[ \t:])([A-Z][A-Z0-9_%$]+)"', r'\1\2 "', s)

        # comment out dot directives like .CREF/.XCREF and "..." pseudo
        m_dot = re.match(r'^\s*\.(\w+)', s)
        if m_dot or re.match(r'^\s*\.\.\.', s):
            if m_dot and m_dot.group(1).lower() in allowed_dot:
                pass
            else:
                out.append("; " + s.strip())
                continue

        # listing/meta directives become comments
        if re.match(r'^\s*(TITLE|SUBTTL|PAGE|COMMENT|SALL|SEARCH)\b', s):
            out.append("; " + s.strip())
            continue
        m = re.match(r'^\s*RADIX\b\s*([0-9]+)?', s, re.IGNORECASE)
        if m:
            if m.group(1):
                try:
                    cur_radix = int(m.group(1))
                except Exception:
                    pass
            out.append("; " + s.strip())
            continue

        # end/purge directives are not used by ca65
        if re.match(r'^\s*END\b(?!\s*:)', s) or re.match(r'^\s*PURGE\b', s):
            out.append("; " + s.strip())
            continue

        if macro_depth > 0 and not re.match(r'^\s*\.endmacro\b', s, re.IGNORECASE):
            s = preprocess_numbers(s, cur_radix)
            s = normalize_symbol_tokens(s)
            s = replace_bang_or(s)
            s = s.replace("<>", "__NEQ__")
            s = s.replace("<", "(").replace(">", ")")
            s = s.replace("__NEQ__", "<>")
            s = re.sub(r'\(([^),]+)\)\s*([+\-])', r'\1\2', s)
            s = re.sub(r'(?<![A-Z0-9_%$])\.(?=[+-])', "*", s)
            s = clamp_immediate_expr(s)
            s = fix_zero_page_minus_one(s)
            m_xwd = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*XWD\s+([^,]+)\s*,\s*(.+)$', s)
            if m_xwd:
                lab_raw = (m_xwd.group(1) + " " if m_xwd.group(1) else "    ")
                lab = normalize_symbol_tokens(lab_raw)
                left = m_xwd.group(2).strip()
                right = m_xwd.group(3).strip()
                emit(f"{lab}.word {left}, {right}" + ((" " + cmt) if cmt else ""))
                continue
            stripped = s.strip()
            if re.fullmatch(r'[0-9\-\+\^$][0-9A-Fa-f\-\+\*/\(\) ]*', stripped):
                indent = re.match(r"^\s*", s).group(0)
                expr = clamp_byte_expr(stripped)
                emit(f"{indent}.byte {expr}" + ((" " + cmt) if cmt else ""))
                continue
            if re.fullmatch(r"'[^'\\]'", stripped):
                indent = re.match(r"^\s*", s).group(0)
                emit(f"{indent}.byte {stripped}" + ((" " + cmt) if cmt else ""))
                continue
            emit(s + ((" " + cmt) if cmt else ""))
            if re.match(r'^\s*\.macro\b', s, re.IGNORECASE):
                macro_depth += 1
            if re.match(r'^\s*\.if(n?def)?\b', s, re.IGNORECASE):
                cond_stack.append(set())
            elif re.match(r'^\s*\.else(if)?\b', s, re.IGNORECASE):
                if cond_stack:
                    cond_stack[-1].clear()
            elif re.match(r'^\s*\.endif\b', s, re.IGNORECASE):
                if cond_stack:
                    cond_stack.pop()
            continue

        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*(DCI|DCE)\b(.*)$', s)
        if m and macro_depth == 0:
            s = preprocess_numbers(s, cur_radix)
            s = normalize_symbol_tokens(s)
            s = re.sub(r"\b(DCI|DCE)\s*'([^']*)'", r'\1 "\2"', s)
            s = re.sub(r'\b(DCI|DCE)\s*"', r'\1 "', s)
            emit(s + ((" " + cmt) if cmt else ""))
            step = "Q+1" if m.group(2) == "DCI" else "Q+2"
            emit(f"    ; Q={step}")
            emit_redefine("Q", step, "")
            continue

        # ORG -> .org
        s = re.sub(r'(?m)^\s*ORG\b', "    .org", s)

        # replace angle bracket grouping with parentheses (preserve <>)
        s = s.replace("<>", "__NEQ__")
        s = s.replace("<", "(").replace(">", ")")
        s = s.replace("__NEQ__", "<>")
        s = re.sub(r'\(([^),]+)\)\s*([+\-])', r'\1\2', s)

        # convert .+ / .- to ca65 PC-relative syntax
        s = re.sub(r'(?<![A-Z0-9_%$])\.(?=[+-])', "*", s)

        # collapse double colon labels
        s = s.replace("::", ":")

        # BLOCK (optional label)
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*BLOCK\s+(.+)$', s)
        if m:
            expr_raw = m.group(2).strip()
            if len(expr_raw.split()) >= 2 and not re.search(r"[#$()\[\],+\-*/]", expr_raw) and re.fullmatch(r"[A-Z0-9_ \t]+", expr_raw):
                out.append("; " + s.strip())
                continue
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            expr = normalize_symbol_tokens(preprocess_numbers(expr_raw, cur_radix))
            emit(f"{lab}.res {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # ADR(expr) or ADR expr
        s = re.sub(r'\bADR\(\s*([^)]+)\s*\)', r'.word \1', s)
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*ADR\s+(.+)$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            expr = normalize_symbol_tokens(preprocess_numbers(m.group(2).strip(), cur_radix))
            emit(f"{lab}.word {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # LABEL: SYMBOL -> LABEL = SYMBOL (alias)
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*([A-Za-z0-9_%$]+)\s*$', s)
        if m and m.group(2) not in BASE_OPS and not re.fullmatch(r'[0-9]+', m.group(2)) and m.group(2).upper() not in defined_macros:
            out.append(f"{m.group(1)} = {m.group(2)}" + ((" " + cmt) if cmt else ""))
            continue

        # XWD left,right -> placeholder: two 16-bit words
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*XWD\s+([^,]+)\s*,\s*(.+)$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            left = normalize_symbol_tokens(preprocess_numbers(m.group(2).strip(), cur_radix))
            right = normalize_symbol_tokens(preprocess_numbers(m.group(3).strip(), cur_radix))
            emit(f"{lab}.word {left}, {right}" + ((" " + cmt) if cmt else ""))
            continue

        # DC/DT string literals
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*DC\s*\(?\"([^\"]*)\"\)?\s*$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            emit(f"{lab}.byte {bytes_for_string(m.group(2), True)}" + ((" " + cmt) if cmt else ""))
            continue
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*DC\s*\(?\'([^\']*)\'\)?\s*$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            emit(f"{lab}.byte {bytes_for_string(m.group(2), True)}" + ((" " + cmt) if cmt else ""))
            continue
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*DT\s*\(?\"([^\"]*)\"\)?\s*$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            emit(f"{lab}.byte {bytes_for_string(m.group(2), False)}" + ((" " + cmt) if cmt else ""))
            continue
        m = re.match(r'^\s*([A-Za-z0-9_%$]+:)?\s*DT\s*\(?\'([^\']*)\'\)?\s*$', s)
        if m:
            lab_raw = (m.group(1) + " " if m.group(1) else "    ")
            lab = normalize_symbol_tokens(lab_raw)
            emit(f"{lab}.byte {bytes_for_string(m.group(2), False)}" + ((" " + cmt) if cmt else ""))
            continue

        # LABEL: numeric -> .byte
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*([0-9\-\+\^$].*)$', s)
        if m and not re.match(r'^[A-Z0-9_%$]+\s*(==|=)', m.group(2).strip()):
            lab = normalize_symbol_tokens(m.group(1)) + ":"
            expr = normalize_symbol_tokens(preprocess_numbers(m.group(2).strip(), cur_radix))
            expr = clamp_byte_expr(expr)
            emit(f"{lab} .byte {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # LABEL: char/expression -> .byte
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*(\'.+)$', s)
        if m:
            lab = normalize_symbol_tokens(m.group(1)) + ":"
            expr = normalize_symbol_tokens(preprocess_numbers(m.group(2).strip(), cur_radix))
            emit(f"{lab} .byte {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # bare numeric -> .byte
        if re.match(r'^\s*[0-9\-\+\^$][0-9A-Fa-f\-\+\*/\(\) ]*$', s.strip()):
            expr = normalize_symbol_tokens(preprocess_numbers(s.strip(), cur_radix))
            expr = clamp_byte_expr(expr)
            emit(f"    .byte {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # bare char/expression -> .byte
        if re.match(r'^\s*\'[^\'\\]\'', s.strip()):
            expr = normalize_symbol_tokens(preprocess_numbers(s.strip(), cur_radix))
            emit(f"    .byte {expr}" + ((" " + cmt) if cmt else ""))
            continue

        # LABEL: NAME=expr -> label applies to next emitted data line
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*([A-Za-z0-9_%$]+)\s*(==|=)\s*(.+)$', s)
        if m:
            pending_label = normalize_symbol_tokens(m.group(1))
            name = normalize_symbol_tokens(m.group(2))
            expr = normalize_symbol_tokens(preprocess_numbers(m.group(4).strip(), cur_radix))
            tokens = set(re.findall(r"\b[A-Z][A-Z0-9_%]*\b", expr))
            unknown = {t for t in tokens if t not in sym_values and t != name}
            expr_out = expr
            if not unknown:
                try:
                    val = safe_eval_int(expr, sym_values)
                    sym_values[name] = val
                    expr_out = str(val)
                except Exception:
                    pass
            if name in redefined:
                emit_redefine(name, expr_out, cmt, consume_label=False)
            else:
                emit(f"{name} = {expr_out}" + ((" " + cmt) if cmt else ""), consume_label=False)
            seen_names.add(name)
            continue

        # equates -> .define (allow redefinition)
        m = re.match(r'^\s*([A-Za-z0-9_%$]+)\s*(==|=)\s*(.+)$', s)
        if m:
            name = normalize_symbol_tokens(m.group(1))
            expr = normalize_symbol_tokens(preprocess_numbers(m.group(3).strip(), cur_radix))
            tokens = set(re.findall(r"\b[A-Z][A-Z0-9_%]*\b", expr))
            unknown = {t for t in tokens if t not in sym_values and t != name}
            expr_out = expr
            if not unknown:
                try:
                    val = safe_eval_int(expr, sym_values)
                    sym_values[name] = val
                    expr_out = str(val)
                except Exception:
                    pass
            if name in redefined:
                emit_redefine(name, expr_out, cmt)
            else:
                emit(f"{name} = {expr_out}" + ((" " + cmt) if cmt else ""))
            seen_names.add(name)
            continue

        # LABEL: SYMBOL -> LABEL = SYMBOL (alias)
        m = re.match(r'^\s*([A-Za-z0-9_%$]+):\s*([A-Za-z0-9_%$]+)\s*$', s)
        if m and not re.fullmatch(r'[0-9]+', m.group(2)) and m.group(2) not in BASE_OPS and m.group(2).upper() not in defined_macros:
            left = normalize_symbol_tokens(m.group(1))
            right = normalize_symbol_tokens(m.group(2))
            emit(f"{left} = {right}" + ((" " + cmt) if cmt else ""))
            continue

        # long-branch fixup for BEQ EXP (out of range under ca65)
        m = re.match(r'^(\s*(?:[A-Za-z0-9_%$]+:\s*)?)BEQ\s+EXP\s*$', s)
        if m:
            emit(f"{m.group(1)}BNE *+3" + ((" " + cmt) if cmt else ""))
            emit("    JMP EXP")
            continue

        s = preprocess_numbers(s, cur_radix)
        s = normalize_symbol_tokens(s)
        s = replace_bang_or(s)
        s = clamp_immediate_expr(s)
        if macro_depth > 0:
            s = preprocess_numbers(s, cur_radix)
            s = normalize_symbol_tokens(s)
            s = replace_bang_or(s)
            s = clamp_immediate_expr(s)
            s = fix_zero_page_minus_one(s)
            emit(s + ((" " + cmt) if cmt else ""))
        else:
            s = fix_zero_page_minus_one(s)
            emit(s + ((" " + cmt) if cmt else ""))

        if re.match(r'^\s*\.macro\b', s, re.IGNORECASE):
            macro_depth += 1
        elif re.match(r'^\s*\.endmacro\b', s, re.IGNORECASE):
            macro_depth = max(0, macro_depth - 1)
        if re.match(r'^\s*\.if(n?def)?\b', s, re.IGNORECASE):
            cond_stack.append(set())
        elif re.match(r'^\s*\.else(if)?\b', s, re.IGNORECASE):
            if cond_stack:
                cond_stack[-1].clear()
        elif re.match(r'^\s*\.endif\b', s, re.IGNORECASE):
            if cond_stack:
                cond_stack.pop()
    return "\n".join(out) + "\n"

def build_sym_table(src: str, first_only: bool = False):
    sym = {}
    for ln_ in src.splitlines():
        code = ln_.split(";", 1)[0].strip()
        if not code:
            continue
        m = re.match(r'^([A-Z][A-Z0-9_$%]*)\s*(==|=)\s*(.+)$', code)
        if m and "<" not in code and ">" not in code:
            name, expr = m.group(1), m.group(3)
            if first_only and name in sym:
                continue
            try:
                sym[name] = safe_eval_int(expr, sym)
            except Exception:
                pass
    return sym

def convert(src_text: str):
    src_text = nl(src_text)
    config = extract_config_overrides(src_text)
    stage0 = convert_conditionals_to_ca65(src_text)
    stage0, defines = convert_defines_to_macros(stage0)
    stage0 = convert_repeat_to_ca65(stage0)
    stage0 = convert_irpc_to_ca65(stage0)
    sym = build_sym_table(stage0)
    sym.update(config)
    stage1 = expand_exp_lines(stage0, sym)
    stage2 = normalize_instruction_macros(stage1)
    stage3 = normalize_directives(stage2, defines)
    helper = (
        "; ca65 helper macros for converted source\n"
        ".macro DC S,\n"
        "    .if .match ({S}, \"\")\n"
        "    .elseif .match (.left (1, {S}), 0)\n"
        "        .byte {S}\n"
        "    .else\n"
        "        .repeat .strlen ({S}), I\n"
        "            .if I = .strlen ({S}) - 1\n"
        "                .byte (.strat ({S}, I) | $80)\n"
        "            .else\n"
        "                .byte .strat ({S}, I)\n"
        "            .endif\n"
        "        .endrepeat\n"
        "    .endif\n"
        ".endmacro\n"
        "\n"
    )
    return helper + stage3, defines

def find_unresolved(text: str):
    CA65_DIRS = {".org",".byte",".word",".res",".include",".segment",".proc",".endproc",".export",".import",
                ".scope",".endscope",".setcpu",".ifdef",".ifndef",".if",".else",".elseif",".endif",".repeat",".endrepeat",
                ".macro",".endmacro",".local",".endlocal",".assert",".align",".asciiz",".ascii",".bank",".define",".undef",
                ".exitmacro",".set"}
    unresolved = {}
    for ln_ in text.splitlines():
        code = ln_.split(";", 1)[0].strip()
        if not code:
            continue
        if re.fullmatch(r'[A-Z0-9_%$]+:', code):
            continue
        m = re.match(r'^([A-Z0-9_%$]+):\s*(.*)$', code)
        if m:
            code = m.group(2).strip()
        if not code:
            continue
        tok = code.split()[0]
        if tok in BASE_OPS:
            continue
        if tok.startswith("."):
            if tok.lower() in CA65_DIRS:
                continue
            unresolved[tok] = unresolved.get(tok, 0) + 1
            continue
        if re.match(r'^[A-Z0-9_%$]+\s*(==|=)', ln_.strip()):
            continue
        if re.match(r'^[A-Z][A-Z0-9_%$]*$', tok):
            unresolved[tok] = unresolved.get(tok, 0) + 1
    return dict(sorted(unresolved.items(), key=lambda kv: (-kv[1], kv[0])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--dump-macros", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="write unresolved token frequency as json")
    args = ap.parse_args()

    src = args.input.read_text(errors="ignore")
    out, defines = convert(src)
    args.output.write_text(out, encoding="utf-8")

    if args.dump_macros:
        args.dump_macros.write_text(json.dumps({k: v["params"] for k, v in defines.items()}, indent=2), encoding="utf-8")

    if args.report:
        args.report.write_text(json.dumps(find_unresolved(out), indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
