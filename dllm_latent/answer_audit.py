"""Gold-independent, conservative answer extraction for OFFLINE review.

This is a diagnostic parser, not an official GSM8K evaluator. Ambiguous cases
are retained for manual review, never resolved by searching for the gold number.
"""
from fractions import Fraction
import re

ATOM = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?'
VALUE = re.compile(r'(?:' + ATOM + r')(?:\s*/\s*(?:' + ATOM + r'))?')
FINAL = re.compile(r'(?:final\s+answer|(?:the\s+)?answer\s+is)\s*[:=]?\s*', re.I)


def exact_number(text):
    try:
        clean = text.replace(',', '').replace(' ', '')
        # Avoid pathological exponents while admitting scientific notation in saved gold.
        if len(clean)>100 or any(abs(int(e))>1000 for e in re.findall(r'[eE]([-+]?\d+)', clean)):
            return None
        return str(Fraction(clean))
    except (ValueError, ZeroDivisionError):
        return None


def extract_review_answer(text):
    text = text.replace('\u2212', '-').replace('\\,', '').strip()
    # Reduce simple numeric LaTeX fractions; no symbolic math evaluation.
    text = re.sub(r'\\(?:d?frac)\{\s*('+ATOM+r')\s*\}\{\s*('+ATOM+r')\s*\}',
                  r'\1/\2', text)
    signals=[]
    for m in re.finditer(r'####', text):signals.append((m.start(),m.end(),'marker'))
    for m in re.finditer(r'\\boxed\{', text):signals.append((m.start(),m.end(),'boxed'))
    for m in FINAL.finditer(text):signals.append((m.start(),m.end(),'answer_phrase'))
    if not signals:
        return dict(candidate=None,source='none',needs_review=True,reason='no_explicit_final_answer')
    _,end,source=max(signals)
    tail=text[end:].strip()
    # Formatting is harmless, but mathematical operators and words are not stripped.
    tail=re.sub(r'^(?:[\s$*`:=]|\\[\[\](])*(?:\\boxed\{)?', '', tail)
    m=VALUE.match(tail)
    if m is None:
        return dict(candidate=None,source=source,needs_review=True,reason='answer_not_numeric_at_start')
    candidate=exact_number(m.group())
    rest=tail[m.end():]
    reasons=[]
    if candidate is None:reasons.append('invalid_numeric_value')
    # Reject partial matches, equations, percentages, ranges and trailing corrections.
    if rest and (rest[0].isalnum() or rest[0] in '/,+-='):
        reasons.append('possible_expression_or_partial_number')
    if VALUE.search(rest):reasons.append('additional_numbers_after_answer')
    if re.search(r'[%=+]|\\(?:times|cdot)|\b(?:or|correction|actually|instead)\b',rest,re.I):
        reasons.append('ambiguous_answer_suffix')
    if '\n' in rest.strip():reasons.append('trailing_multiline_text')
    # Prose conclusions always enter manual review; explicit markers/boxes may resolve automatically.
    if source=='answer_phrase':reasons.append('prose_answer_requires_review')
    return dict(candidate=candidate,source=source,needs_review=bool(reasons),reason=';'.join(reasons))
