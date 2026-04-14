import io
import os
import re
import json
import uuid
from collections import OrderedDict
from flask import Flask, request, jsonify, send_file, render_template, Response, stream_with_context
import pandas as pd
import numpy as np

app = Flask(__name__)
UPLOAD_CACHE_MAX = 8
RESULT_CACHE_MAX = 16
UPLOAD_CACHE = OrderedDict()
RESULT_CACHE = OrderedDict()

# ─── Phone helpers ─────────────────────────────────────────────────────────────

def extract_phones(raw: str):
    raw = str(raw).strip()
    parts = re.split(r'[;|/\\\n]+|(?<!\d),(?!\d{3})', raw)
    return [re.sub(r'[^\d]', '', p.strip()) for p in parts if re.sub(r'[^\d]', '', p.strip())]


def only_digits(val: str) -> str:
    return re.sub(r'[^\d]', '', str(val))


def is_repeated_digits(digits: str) -> bool:
    return bool(digits) and len(set(digits)) == 1


def is_valid_cpf(digits: str) -> bool:
    if len(digits) != 11 or is_repeated_digits(digits):
        return False

    total = sum(int(digits[i]) * (10 - i) for i in range(9))
    check_1 = (total * 10 % 11) % 10
    total = sum(int(digits[i]) * (11 - i) for i in range(10))
    check_2 = (total * 10 % 11) % 10
    return digits[-2:] == f'{check_1}{check_2}'


def is_valid_cnpj(digits: str) -> bool:
    if len(digits) != 14 or is_repeated_digits(digits):
        return False

    def _calc(base: str, weights: list[int]) -> str:
        total = sum(int(d) * w for d, w in zip(base, weights))
        remainder = total % 11
        return '0' if remainder < 2 else str(11 - remainder)

    check_1 = _calc(digits[:12], [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
    check_2 = _calc(digits[:12] + check_1, [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
    return digits[-2:] == f'{check_1}{check_2}'


def looks_like_document_id(val: str) -> bool:
    digits = only_digits(val)
    return is_valid_cpf(digits) or is_valid_cnpj(digits)


def is_document_column_name(col_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]', '', str(col_name).lower())
    return any(token in normalized for token in ('cpf', 'cnpj', 'documento', 'doc', 'rg'))


def normalize_phone(digits: str):
    digits = only_digits(digits)
    if not digits:
        return None
    if digits.startswith('55') and len(digits) in (12, 13):
        digits = digits[2:]
    if len(digits) not in (10, 11):
        return None
    ddd = int(digits[:2])
    if not 11 <= ddd <= 99:
        return None
    number = digits[2:]
    if len(number) == 8:
        if number[0] in '2345':
            return f'55{ddd:02d}{number}'
        if number[0] in '6789':
            number = '9' + number
        else:
            return None
    elif len(number) == 9:
        if not number.startswith('9'):
            return None
    else:
        return None
    return f'55{ddd:02d}{number}'


def looks_like_phone(val: str, reject_documents: bool = True) -> str | None:
    """Return normalized phone if val looks like a phone number, else None."""
    digits = only_digits(val)
    normalized = normalize_phone(digits)
    if not normalized:
        return None
    if reject_documents and looks_like_document_id(digits):
        return None
    return normalized


# ─── Name helpers ──────────────────────────────────────────────────────────────

def split_name(full: str):
    parts = str(full).strip().split()
    if not parts:
        return '', ''
    return parts[0].capitalize(), ' '.join(p.capitalize() for p in parts[1:])


def infer_phone_column(df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = 0.0

    for col in df.columns:
        if is_document_column_name(col):
            continue
        series = df[col].fillna('').astype(str).str.strip()
        sample = series[series.ne('')].head(200)
        if sample.empty:
            continue

        phone_hits = sample.apply(lambda val: looks_like_phone(val) is not None).sum()
        score = phone_hits / len(sample)

        if phone_hits >= 2 and score > best_score:
            best_col = col
            best_score = score

    return best_col if best_score >= 0.35 else None


def infer_name_column(df: pd.DataFrame, phone_col: str | None = None) -> str | None:
    best_col = None
    best_score = 0.0

    for col in df.columns:
        if phone_col and col == phone_col:
            continue

        series = df[col].fillna('').astype(str).str.strip()
        sample = series[series.ne('')].head(200)
        if sample.empty:
            continue

        def _looks_like_name(val: str) -> bool:
            if looks_like_phone(val):
                return False
            cleaned = re.sub(r'[^A-Za-zÀ-ÿ\s]', ' ', str(val)).strip()
            parts = [p for p in cleaned.split() if p]
            if len(parts) < 2:
                return False
            letters = re.sub(r'[^A-Za-zÀ-ÿ]', '', cleaned)
            return len(letters) >= 5

        name_hits = sample.apply(_looks_like_name).sum()
        score = name_hits / len(sample)

        if name_hits >= 2 and score > best_score:
            best_col = col
            best_score = score

    return best_col if best_score >= 0.3 else None


def infer_columns(df: pd.DataFrame) -> dict:
    phone_col = infer_phone_column(df)
    name_col = infer_name_column(df, phone_col)
    return {
        'name_col': name_col,
        'phone_col': phone_col,
    }


def build_download_name(original_name: str, fmt: str) -> str:
    base_name = os.path.basename(original_name or 'arquivo')
    root, _ = os.path.splitext(base_name)
    root = root or 'arquivo'
    ext = 'csv' if fmt == 'csv' else 'xlsx'
    return f'{root}_higienizado.{ext}'


def make_error_payload(title: str, message: str, hint: str = '', kind: str = 'error',
                       details: str = '') -> dict:
    payload = {
        'title': title,
        'message': message,
        'kind': kind,
    }
    if hint:
        payload['hint'] = hint
    if details:
        payload['details'] = details
    return payload


def classify_exception(exc: Exception, context: str = 'general') -> dict:
    message = str(exc).strip()

    if 'Nenhum arquivo enviado' in message:
        return make_error_payload(
            'Arquivo não enviado',
            'Nenhum arquivo foi enviado para o sistema.',
            'Selecione um arquivo antes de continuar.'
        )

    if 'Formato não suportado' in message:
        return make_error_payload(
            'Formato não suportado',
            'Não conseguimos ler esse tipo de arquivo.',
            'Use um arquivo .xlsx, .xls, .csv ou .pdf.'
        )

    if 'Arquivo sem colunas legíveis' in message:
        return make_error_payload(
            'Arquivo sem colunas legíveis',
            'O arquivo foi recebido, mas não encontramos colunas válidas para montar a tabela.',
            'Revise o arquivo de origem e confira se ele tem cabeçalhos e dados estruturados.'
        )

    if isinstance(exc, pd.errors.EmptyDataError):
        return make_error_payload(
            'Arquivo vazio',
            'O arquivo não contém dados suficientes para leitura.',
            'Verifique se ele possui linhas e colunas preenchidas.'
        )

    lowered = message.lower()
    if 'excel file format cannot be determined' in lowered:
        return make_error_payload(
            'Planilha inválida',
            'Não foi possível identificar a estrutura da planilha enviada.',
            'Tente salvar novamente o arquivo como .xlsx ou envie em .csv.'
        )

    if 'unicode' in lowered or 'codec' in lowered or 'encoding' in lowered:
        return make_error_payload(
            'Erro de leitura do arquivo',
            'Não conseguimos interpretar a codificação do arquivo enviado.',
            'Tente abrir e salvar novamente o arquivo como CSV UTF-8 ou planilha .xlsx.'
        )

    if context == 'upload':
        return make_error_payload(
            'Falha ao ler o arquivo',
            'Não foi possível abrir esse arquivo para montar a prévia.',
            'Confira se o arquivo não está corrompido e tente novamente.',
            details=message
        )

    if context == 'process':
        return make_error_payload(
            'Falha na higienização',
            'O arquivo foi recebido, mas houve um problema durante o processamento dos dados.',
            'Revise o mapeamento das colunas e tente novamente.',
            details=message
        )

    if context == 'download':
        return make_error_payload(
            'Falha ao gerar o download',
            'Não foi possível montar o arquivo final higienizado.',
            'Tente processar novamente a lista antes de baixar.',
            details=message
        )

    if context == 'preview':
        return make_error_payload(
            'Falha na visualização',
            'Não foi possível atualizar a prévia com os filtros atuais.',
            'Tente alterar os filtros ou recarregar o arquivo.',
            details=message
        )

    return make_error_payload(
        'Erro inesperado',
        'Ocorreu um erro que não esperávamos neste fluxo.',
        'Tente novamente. Se persistir, revise o arquivo e os dados de entrada.',
        details=message
    )


def jsonify_error(exc: Exception, status_code: int, context: str):
    return jsonify({'error': classify_exception(exc, context)}), status_code


def _cache_put(cache: OrderedDict, key: str, value: dict, max_size: int) -> None:
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


def store_uploaded_dataframe(filename: str, df: pd.DataFrame) -> str:
    token = uuid.uuid4().hex
    _cache_put(UPLOAD_CACHE, token, {'filename': filename, 'df': df}, UPLOAD_CACHE_MAX)
    return token


def get_uploaded_dataframe(token: str) -> dict | None:
    cached = UPLOAD_CACHE.get(token)
    if not cached:
        return None
    UPLOAD_CACHE.move_to_end(token)
    return cached


def build_result_cache_key(upload_token: str | None, filename: str, config: dict) -> str | None:
    if not upload_token:
        return None
    config_key = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return f'{upload_token}:{filename}:{config_key}'


def store_processed_result(cache_key: str | None, result: dict) -> None:
    if not cache_key:
        return
    _cache_put(RESULT_CACHE, cache_key, result, RESULT_CACHE_MAX)


def get_processed_result(cache_key: str | None) -> dict | None:
    if not cache_key:
        return None
    cached = RESULT_CACHE.get(cache_key)
    if not cached:
        return None
    RESULT_CACHE.move_to_end(cache_key)
    return cached


def load_dataframe_from_request(req) -> tuple[pd.DataFrame, str, str | None]:
    upload_token = req.form.get('upload_token')
    if upload_token:
        cached = get_uploaded_dataframe(upload_token)
        if cached:
            return cached['df'], cached['filename'], upload_token

    file = req.files.get('file')
    if not file:
        raise ValueError('Nenhum arquivo enviado')

    df = read_file(file).fillna('')
    if len(df.columns) == 0:
        raise ValueError('Arquivo sem colunas legíveis')
    return df, file.filename, None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.fillna('').astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
    cleaned = cleaned.apply(lambda col: col.str.strip())
    return cleaned.loc[cleaned.ne('').any(axis=1)].reset_index(drop=True)


def get_or_process_result(df: pd.DataFrame, filename: str, upload_token: str | None, config: dict) -> dict:
    result_cache_key = build_result_cache_key(upload_token, filename, config)
    result = get_processed_result(result_cache_key)
    if result is None:
        result = process_dataframe(df, config)
        store_processed_result(result_cache_key, result)
    return result


def parse_datetime_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna('').astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=cleaned.index, dtype='datetime64[ns]')
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d'):
        pending = parsed.isna() & cleaned.ne('')
        if not pending.any():
            break
        parsed.loc[pending] = pd.to_datetime(cleaned.loc[pending], format=fmt, errors='coerce')
    return parsed


def infer_date_columns(df: pd.DataFrame, sample_size: int = 200) -> list[str]:
    date_cols = []
    for col in df.columns:
        series = df[col].fillna('').astype(str).str.strip()
        sample = series[series.ne('')].head(sample_size)
        if sample.empty:
            continue
        parsed = parse_datetime_series(sample)
        hits = int(parsed.notna().sum())
        if hits >= 2 and hits / len(sample) >= 0.35:
            date_cols.append(col)
    return date_cols


def build_preview_meta(df: pd.DataFrame, config: dict) -> dict:
    name_col = config.get('name_col') if config.get('name_col') in df.columns else None
    phone_col = config.get('phone_col') if config.get('phone_col') in df.columns else None
    date_cols = infer_date_columns(df)

    phone_filter_options = [{'value': 'all', 'label': 'Todos os registros'}]
    if phone_col:
        phone_filter_options.extend([
            {'value': 'phone_present', 'label': 'Com telefone'},
            {'value': 'phone_missing', 'label': 'Sem telefone'},
            {'value': 'phone_valid', 'label': 'Telefones válidos'},
            {'value': 'phone_invalid', 'label': 'Telefones inválidos'},
        ])

    sort_options = [{'value': '', 'label': 'Sem ordenação'}]
    if name_col:
        sort_options.extend([
            {'value': f'name::{name_col}::asc', 'label': f'{name_col} A → Z'},
            {'value': f'name::{name_col}::desc', 'label': f'{name_col} Z → A'},
        ])
    if phone_col:
        sort_options.extend([
            {'value': f'ddd::{phone_col}::asc', 'label': 'DDD crescente'},
            {'value': f'ddd::{phone_col}::desc', 'label': 'DDD decrescente'},
        ])
    for col in date_cols:
        sort_options.extend([
            {'value': f'date::{col}::desc', 'label': f'{col} mais recente'},
            {'value': f'date::{col}::asc', 'label': f'{col} mais antiga'},
        ])

    return {
        'phone_col': phone_col,
        'name_col': name_col,
        'date_columns': date_cols,
        'phone_filter_options': phone_filter_options,
        'sort_options': sort_options,
    }


def apply_phone_filter(frame: pd.DataFrame, phone_col: str | None, phone_filter: str) -> pd.DataFrame:
    if not phone_col or phone_col not in frame.columns or phone_filter == 'all':
        return frame

    series = frame[phone_col].fillna('').astype(str).str.strip()
    normalized = series.map(lambda val: looks_like_phone(val, reject_documents=False))
    has_value = series.ne('')
    is_valid = normalized.notna()

    if phone_filter == 'phone_present':
        return frame.loc[has_value]
    if phone_filter == 'phone_missing':
        return frame.loc[~has_value]
    if phone_filter == 'phone_valid':
        return frame.loc[is_valid]
    if phone_filter == 'phone_invalid':
        return frame.loc[has_value & ~is_valid]
    return frame


def apply_date_filter(frame: pd.DataFrame, date_col: str | None, date_from: str, date_to: str) -> pd.DataFrame:
    if not date_col or date_col not in frame.columns or (not date_from and not date_to):
        return frame

    parsed = parse_datetime_series(frame[date_col])
    mask = parsed.notna()
    if date_from:
        mask &= parsed >= pd.to_datetime(date_from)
    if date_to:
        mask &= parsed <= pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return frame.loc[mask]


def apply_preview_sort(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if not sort_mode:
        return frame

    parts = sort_mode.split('::', 2)
    if len(parts) != 3:
        return frame

    mode, col, direction = parts
    ascending = direction != 'desc'
    if col not in frame.columns:
        return frame

    if mode == 'name':
        sort_key = frame[col].fillna('').astype(str).str.strip().str.casefold()
    elif mode == 'ddd':
        normalized = frame[col].fillna('').astype(str).map(
            lambda val: looks_like_phone(val, reject_documents=False)
        )
        sort_key = pd.to_numeric(normalized.str[2:4], errors='coerce')
    elif mode == 'date':
        sort_key = parse_datetime_series(frame[col])
    else:
        return frame

    return frame.assign(_sort_key=sort_key).sort_values(
        by=['_sort_key', col], ascending=ascending, kind='mergesort', na_position='last'
    ).drop(columns=['_sort_key'])


def build_preview_payload(df: pd.DataFrame, cols: list[str], config: dict, search: str = '', limit: int = 10,
                          phone_filter: str = 'all', sort_mode: str = '',
                          date_col: str | None = None, date_from: str = '', date_to: str = '') -> dict:
    frame = df.copy()
    frame = frame.astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
    total_rows = len(frame)
    meta = build_preview_meta(frame, config)

    search = str(search or '').strip()
    if search:
        escaped = re.escape(search)
        mask = frame.apply(lambda col: col.str.contains(escaped, case=False, na=False))
        frame = frame.loc[mask.any(axis=1)]

    frame = apply_phone_filter(frame, meta.get('phone_col'), phone_filter)
    frame = apply_date_filter(frame, date_col, date_from, date_to)
    filtered_total = len(frame)
    frame = apply_preview_sort(frame, sort_mode)

    limit = max(1, min(int(limit or 10), 200))
    frame = frame.head(limit)

    return {
        'columns': cols,
        'rows': frame[cols].to_dict(orient='records'),
        'shown_count': len(frame),
        'filtered_total': filtered_total,
        'total_rows': total_rows,
        'meta': meta,
    }


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
    For rows where phone_col is empty or invalid, scan other columns for phone-like values.
    Returns list of dicts: {row_index, name, found_col, found_value, normalized}
    """
    if phone_col not in df.columns:
        return []

    suggestions = []
    other_cols = [c for c in df.columns if c != phone_col and not is_document_column_name(c)]
    phone_series = df[phone_col].fillna('').astype(str).str.strip()
    pending_rows = phone_series.map(lambda val: looks_like_phone(val, reject_documents=False) is None)
    if not pending_rows.any():
        return []

    name_series = df.iloc[:, 0].fillna('').astype(str)
    found_frames = []
    for col in other_cols:
        candidate_values = df.loc[pending_rows, col].fillna('').astype(str).str.strip()
        if candidate_values.empty:
            continue

        normalized = candidate_values.map(looks_like_phone)
        matched = normalized.notna()
        if not matched.any():
            continue

        matched_index = candidate_values.index[matched]
        found_frames.append(pd.DataFrame({
            'row_index': matched_index.astype(int),
            'name': name_series.loc[matched_index].where(
                name_series.loc[matched_index].str.strip().ne(''),
                [f'Linha {idx+2}' for idx in matched_index]
            ),
            'found_col': col,
            'found_value': candidate_values.loc[matched_index].values,
            'normalized': normalized.loc[matched_index].values,
        }))
        pending_rows.loc[matched_index] = False
        if not pending_rows.any():
            break

    if found_frames:
        suggestions = (
            pd.concat(found_frames, ignore_index=True)
            .sort_values('row_index')
            .to_dict(orient='records')
        )
    return suggestions


# ─── Core processing (vectorized) ─────────────────────────────────────────────

def process_dataframe(df: pd.DataFrame, config: dict) -> dict:
    name_col      = config.get('name_col')
    phone_col     = config.get('phone_col')
    split_names   = config.get('split_names', True)
    dup_action    = config.get('dup_action', 'keep')
    keep_cols     = config.get('keep_cols', list(df.columns))
    # List of accepted misplaced phones from the UI
    phone_fixes   = {
        int(f['row_index']): {
            'normalized': f.get('normalized', ''),
            'found_col':  f.get('found_col'),
        }
        for f in config.get('phone_fixes', [])
        if 'row_index' in f
    }

    warnings = []

    df = clean_dataframe(df)

    # Apply accepted phone fixes before processing
    if phone_col and phone_fixes:
        for row_idx, fix in phone_fixes.items():
            if row_idx in df.index:
                norm_phone = normalize_phone(re.sub(r'[^\d]', '', str(fix.get('normalized', ''))))
                if not norm_phone:
                    continue
                df.at[row_idx, phone_col] = norm_phone
                found_col = fix.get('found_col')
                if found_col and found_col in df.columns and found_col != phone_col:
                    df.at[row_idx, found_col] = ''

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

        # Keep only normalized phones in the final file.
        df[phone_col] = normalized.fillna('')
        df = df.drop(columns=['_phones'])

    # ── Select columns ──
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    result = df[cols_to_keep].copy()

    # ── Name handling ──
    if name_col and name_col in result.columns:
        name_series = result[name_col].astype(str).str.strip()
        result[name_col] = name_series
        if split_names:
            split = name_series.str.split(n=1, expand=True)
            first_name = split[0].fillna('').str.capitalize() if 0 in split.columns else pd.Series('', index=result.index)
            last_name = split[1].fillna('').str.title() if 1 in split.columns else pd.Series('', index=result.index)
            name_pos = result.columns.get_loc(name_col)
            if 'first_name' in result.columns:
                result['first_name'] = first_name
            else:
                result.insert(name_pos + 1, 'first_name', first_name)
            if 'last_name' in result.columns:
                result['last_name'] = last_name
            else:
                name_pos = result.columns.get_loc(name_col)
                first_name_pos = result.columns.get_loc('first_name') if 'first_name' in result.columns else name_pos
                insert_pos = max(name_pos, first_name_pos) + 1
                result.insert(insert_pos, 'last_name', last_name)

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
        return jsonify_error(ValueError('Nenhum arquivo enviado'), 400, 'upload')
    try:
        df = read_file(file)
        df = df.fillna('')
        if len(df.columns) == 0:
            raise ValueError('Arquivo sem colunas legíveis')
        upload_token = store_uploaded_dataframe(file.filename, df)
        inferred = infer_columns(df)
        return jsonify({
            'columns':           list(df.columns),
            'preview':           df.head(10).to_dict(orient='records'),
            'total_rows':        len(df),
            'suggested_columns': inferred,
            'upload_token':      upload_token,
        })
    except Exception as e:
        status_code = 400 if isinstance(e, (ValueError, pd.errors.EmptyDataError)) else 500
        return jsonify_error(e, status_code, 'upload')


@app.route('/process_stream', methods=['POST'])
def process_stream():
    config   = json.loads(request.form.get('config', '{}'))
    try:
        df, filename, upload_token = load_dataframe_from_request(request)
    except Exception as e:
        def generate_error():
            payload = {'error': classify_exception(e, 'process')}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(generate_error()), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    def generate():
        def emit(pct, msg, data=None):
            payload = {'pct': pct, 'msg': msg}
            if data: payload['data'] = data
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            yield from emit(5, 'Lendo arquivo…')
            total = len(df)

            yield from emit(20, f'{total} linhas encontradas. Processando…')
            result = get_or_process_result(df, filename, upload_token, config)
            result_df = result['df']

            # Detect misplaced phones (only if no fixes already provided)
            phone_col = config.get('phone_col')
            misplaced = []
            if phone_col and not config.get('phone_fixes'):
                df_clean = clean_dataframe(df)
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
            yield f"data: {json.dumps({'error': classify_exception(e, 'process')}, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/download', methods=['POST'])
def download():
    config = json.loads(request.form.get('config', '{}'))
    fmt    = request.form.get('format', 'xlsx')
    try:
        df, filename, upload_token = load_dataframe_from_request(request)
    except ValueError as e:
        return jsonify_error(e, 400, 'download')
    try:
        download_name = build_download_name(filename, fmt)
        result = get_or_process_result(df, filename, upload_token, config)
        result_df = result['df']
        buf = io.BytesIO()
        if fmt == 'csv':
            result_df.to_csv(buf, index=False, encoding='utf-8-sig')
            buf.seek(0)
            return send_file(buf, mimetype='text/csv',
                             as_attachment=True, download_name=download_name)
        else:
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Higienizado')
            buf.seek(0)
            return send_file(buf,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True, download_name=download_name)
    except Exception as e:
        status_code = 400 if isinstance(e, (ValueError, pd.errors.EmptyDataError)) else 500
        return jsonify_error(e, status_code, 'download')


@app.route('/preview', methods=['POST'])
def preview():
    config = json.loads(request.form.get('config', '{}'))
    stage = request.form.get('stage', 'before')
    search = request.form.get('search', '')
    phone_filter = request.form.get('phone_filter', 'all')
    sort_mode = request.form.get('sort_mode', '')
    date_col = request.form.get('date_col') or None
    date_from = request.form.get('date_from', '')
    date_to = request.form.get('date_to', '')
    limit = request.form.get('limit', '10')

    try:
        df, filename, upload_token = load_dataframe_from_request(request)
    except ValueError as e:
        return jsonify_error(e, 400, 'preview')
    try:
        if stage == 'after':
            result = get_or_process_result(df, filename, upload_token, config)
            preview_df = result['df']
        else:
            preview_df = df.fillna('').astype(str)

        cols = list(preview_df.columns)
        payload = build_preview_payload(
            preview_df,
            cols=cols,
            config=config,
            search=search,
            limit=limit,
            phone_filter=phone_filter,
            sort_mode=sort_mode,
            date_col=date_col,
            date_from=date_from,
            date_to=date_to,
        )
        return jsonify(payload)
    except Exception as e:
        status_code = 400 if isinstance(e, (ValueError, pd.errors.EmptyDataError)) else 500
        return jsonify_error(e, status_code, 'preview')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
