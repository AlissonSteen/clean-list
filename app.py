import io
import re
import json
from flask import Flask, request, jsonify, send_file, render_template, Response
import pandas as pd
import numpy as np

app = Flask(__name__)

# ─── Phone helpers ─────────────────────────────────────────────────────────────

def extract_phones(raw: str):
    raw = str(raw).strip()
    parts = re.split(r'[;|/\\\n]+|(?<!\d),(?!\d{3})', raw)
    return [re.sub(r'[^\d]', '', p.strip()) for p in parts if re.sub(r'[^\d]', '', p.strip())]


def normalize_phone(digits: str):
    if not digits:
        return None
    if digits.startswith('55') and len(digits) >= 12:
        digits = digits[2:]
    if len(digits) < 10:
        return None
    ddd = int(digits[:2])
    number = digits[2:]
    if ddd >= 28:
        if len(number) == 9 and number.startswith('9'):
            number = number[1:]
    else:
        if len(number) == 8:
            number = '9' + number
    if len(number) < 8:
        return None
    return f'55{ddd:02d}{number}'


def looks_like_phone(val: str) -> str | None:
    """Return normalized phone if val looks like a phone number, else None."""
    digits = re.sub(r'[^\d]', '', str(val))
    if 10 <= len(digits) <= 13:
        return normalize_phone(digits)
    return None


# ─── Name helpers ──────────────────────────────────────────────────────────────

def split_name(full: str):
    parts = str(full).strip().split()
    if not parts:
        return '', ''
    return parts[0].capitalize(), ' '.join(p.capitalize() for p in parts[1:])


# ─── File reading ──────────────────────────────────────────────────────────────

def read_file(file) -> pd.DataFrame:
    name = file.filename.lower()
    raw = file.read()
    buf = io.BytesIO(raw)
    if name.endswith('.csv'):
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc, dtype=str)
            except Exception:
                pass
    elif name.endswith(('.xlsx', '.xls')):
        buf.seek(0)
        return pd.read_excel(buf, dtype=str)
    elif name.endswith('.pdf'):
        import pdfplumber
        rows = []
        buf.seek(0)
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in table:
                        rows.append([str(c) if c else '' for c in row])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=rows[0]).astype(str)
    raise ValueError(f'Formato não suportado: {name}')


# ─── Detect misplaced phones ───────────────────────────────────────────────────

def detect_misplaced_phones(df: pd.DataFrame, phone_col: str, keep_cols: list) -> list:
    """
    For rows where phone_col is empty, scan other columns for phone-like values.
    Returns list of dicts: {row_index, name, found_col, found_value, normalized}
    """
    suggestions = []
    other_cols = [c for c in df.columns if c != phone_col]

    for idx, row in df.iterrows():
        phone_val = str(row.get(phone_col, '')).strip()
        if phone_val and phone_val.lower() not in ('nan', ''):
            continue  # already has phone
        for col in other_cols:
            val = str(row.get(col, '')).strip()
            if not val or val.lower() in ('nan', ''):
                continue
            normalized = looks_like_phone(val)
            if normalized:
                suggestions.append({
                    'row_index':  int(idx),
                    'name':       str(row.get(list(df.columns)[0], f'Linha {idx+2}')),
                    'found_col':  col,
                    'found_value': val,
                    'normalized': normalized,
                })
                break  # one suggestion per row
    return suggestions


# ─── Core processing (vectorized) ─────────────────────────────────────────────

def process_dataframe(df: pd.DataFrame, config: dict) -> dict:
    name_col      = config.get('name_col')
    phone_col     = config.get('phone_col')
    split_names   = config.get('split_names', True)
    dup_action    = config.get('dup_action', 'keep')
    keep_cols     = config.get('keep_cols', list(df.columns))
    # List of {row_index, normalized} accepted by user
    phone_fixes   = {int(f['row_index']): f['normalized']
                     for f in config.get('phone_fixes', [])}

    warnings = []

    # Convert everything to string
    df = df.astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
    df = df.apply(lambda col: col.str.strip())
    df = df[df.apply(lambda r: r.str.strip().any(), axis=1)]
    df = df.reset_index(drop=True)

    # Apply accepted phone fixes before processing
    if phone_col and phone_fixes:
        for row_idx, norm_phone in phone_fixes.items():
            if row_idx in df.index:
                df.at[row_idx, phone_col] = norm_phone

    # ── Handle multiple phones per row (explode) ──
    if phone_col and phone_col in df.columns:
        df['_phones'] = df[phone_col].apply(
            lambda v: extract_phones(v) if v.strip() else ['']
        )
        df = df.explode('_phones').reset_index(drop=True)
        df['_phones'] = df['_phones'].fillna('').astype(str).replace(
            {'nan': '', 'None': '', 'NaN': ''})

        def _norm(v):
            if not isinstance(v, str):
                return None
            return normalize_phone(re.sub(r'[^\d]', '', v))

        normalized = df['_phones'].apply(_norm)

        bad_mask = normalized.isna() & df['_phones'].str.strip().ne('')
        for idx in df[bad_mask].index:
            warnings.append(f'Linha {idx+2}: telefone inválido "{df["_phones"][idx]}"')

        df[phone_col] = normalized.where(~normalized.isna(), df['_phones'])
        df = df.drop(columns=['_phones'])

    # ── Select columns ──
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    result = df[cols_to_keep].copy()

    # ── Name handling ──
    if name_col and name_col in result.columns:
        name_series = result[name_col].astype(str).str.strip()
        result[name_col] = name_series
        if split_names:
            split = name_series.apply(split_name)
            result.insert(result.columns.get_loc(name_col) + 1, 'first_name', split.apply(lambda x: x[0]))
            result.insert(result.columns.get_loc(name_col) + 2, 'last_name',  split.apply(lambda x: x[1]))

    # ── Reorder columns ──
    ordered = []
    if name_col and name_col in result.columns:
        ordered.append(name_col)
    if split_names and name_col:
        if 'first_name' in result.columns: ordered.append('first_name')
        if 'last_name'  in result.columns: ordered.append('last_name')
    if phone_col and phone_col in result.columns:
        ordered.append(phone_col)
    for c in result.columns:
        if c not in ordered:
            ordered.append(c)
    result = result[[c for c in ordered if c in result.columns]]

    # ── Remove rows without phone ──
    if phone_col and phone_col in result.columns:
        result = result[result[phone_col].astype(str).str.strip().ne('')]

    # ── Duplicates ──
    dup_count = int(result.duplicated(keep='first').sum())
    if dup_action == 'remove':
        result = result.drop_duplicates()

    return {
        'df':        result,
        'dup_count': dup_count,
        'warnings':  warnings,
    }


# ─── FakeFile helper ───────────────────────────────────────────────────────────

class FakeFile:
    def __init__(self, buf, name):
        self.filename = name
        self._buf = buf
    def read(self):
        self._buf.seek(0)
        return self._buf.read()
    def seek(self, *a):
        return self._buf.seek(*a)
    def tell(self):
        return self._buf.tell()


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    try:
        df = read_file(file)
        df = df.fillna('')
        return jsonify({
            'columns':    list(df.columns),
            'preview':    df.head(10).to_dict(orient='records'),
            'total_rows': len(df),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_stream', methods=['POST'])
def process_stream():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    config   = json.loads(request.form.get('config', '{}'))
    filename = file.filename
    file_buf = io.BytesIO(file.read())
    fake     = FakeFile(file_buf, filename)

    def generate():
        def emit(pct, msg, data=None):
            payload = {'pct': pct, 'msg': msg}
            if data: payload['data'] = data
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            yield from emit(5, 'Lendo arquivo…')
            fake.seek(0)
            df    = read_file(fake)
            df    = df.fillna('')
            total = len(df)

            yield from emit(20, f'{total} linhas encontradas. Processando…')
            result    = process_dataframe(df, config)
            result_df = result['df']

            # Detect misplaced phones (only if no fixes already provided)
            phone_col = config.get('phone_col')
            misplaced = []
            if phone_col and not config.get('phone_fixes'):
                df_clean = df.astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
                df_clean = df_clean.apply(lambda col: col.str.strip())
                misplaced = detect_misplaced_phones(df_clean, phone_col, config.get('keep_cols', []))

            yield from emit(85, 'Finalizando…')

            payload = {
                'columns_before': list(df.columns),
                'columns_after':  list(result_df.columns),
                'preview_before': df.head(10).to_dict(orient='records'),
                'preview_after':  result_df.head(10).to_dict(orient='records'),
                'total_before':   total,
                'total_after':    len(result_df),
                'dup_count':      result['dup_count'],
                'warnings':       result['warnings'],
                'misplaced':      misplaced,
            }
            yield from emit(100, 'Concluído!', payload)

        except Exception as e:
            import traceback; traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/download', methods=['POST'])
def download():
    file   = request.files.get('file')
    config = json.loads(request.form.get('config', '{}'))
    fmt    = request.form.get('format', 'xlsx')
    if not file:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    df        = read_file(file)
    df        = df.fillna('')
    result    = process_dataframe(df, config)
    result_df = result['df']
    buf = io.BytesIO()
    if fmt == 'csv':
        result_df.to_csv(buf, index=False, encoding='utf-8-sig')
        buf.seek(0)
        return send_file(buf, mimetype='text/csv',
                         as_attachment=True, download_name='lista_higienizada.csv')
    else:
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Higienizado')
        buf.seek(0)
        return send_file(buf,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name='lista_higienizada.xlsx')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)