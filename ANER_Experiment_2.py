"""ANER experiment 2: a fixed ML checkpoint plus bounded inference-time rules.

In the existing Colab notebook, upload this file after experiment 1 and execute:
    exec(uploaded['ANER_Experiment_2.py'].decode('utf-8'), globals())

No retraining, gold edits or new split. Evaluation runs automatically; test is
disabled. Import this module before loading a saved hybrid pipeline in a fresh
process so spaCy can register its custom component.
"""
import copy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import spacy
from IPython.display import display
from spacy.language import Language
from spacy.tokens import Doc

RULE_VERSION = 'aner_hybrid_v1'
COMPONENT_NAME = 'aner_hybrid_rules_v1'
DEFAULT_RULE_CONFIG = {
    'version': RULE_VERSION,
    'org_names': ['PLACSP', 'Registro Mercantil', 'ROLECE'],
    'loc_names': ['Tenerife'],
    'seguridad_social_context_rule': True,
    'enable_dates': True,
    'enable_schedules': True,
    'enable_law_forms': True,
    'enable_email': True,
    'enable_technical_references': True,
}
# Test results are exploratory: the existing test set has already been inspected.
RUN_HYBRID_TEST = globals().get('RUN_HYBRID_TEST', False)

MONTHS = 'enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre'
MONTH_NUMBER = {name: number for number, name in enumerate(
    ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto',
     'septiembre', 'octubre', 'noviembre', 'diciembre'], start=1)}
MONTH_NUMBER['setiembre'] = 9
DATE_WORDS = rf'(?:0?[1-9]|[12]\d|3[01])\s+(?:de\s+)?(?:{MONTHS})(?:\s+(?:de\s+)?(?:19|20)\d{{2}})?'
TIME = r'(?:[01]?\d|2[0-3])(?::[0-5]\d(?:\s*(?:horas|h))?|h)'
DAY = r'(?:lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)'
ARTICLE_NUMBER = r'\d+(?:\.\d+)*(?:\.[a-z]\))?'
ARTICLE_LIST = rf'{ARTICLE_NUMBER}(?:\s*(?:,|y|a)\s*{ARTICLE_NUMBER})*'
ARTICLE_PREFIX = rf'(?:art[íi]culos?|art\.?)\s+{ARTICLE_LIST}\s+(?:(?:de\s+la|del|de)\s+)?'
LEGAL_ID = r'\d{1,4}/\d{4}'
LEGAL_ANCHOR = re.compile(r'\b(?:LCSP|Ley\s+(?:Org[aá]nica\s+)?\d+/\d{4}|Real\s+Decreto\s+\d+/\d{4}|Reglamento\s+\d+/\d{4}|Directiva\s+\d+(?:/\d+)+/(?:CE|UE|CEE))\b', re.I)
ORG_PREFIX = re.compile(r'^(?:Ayuntamiento|Consejer[íi]a|Instituto|Plataforma|Entidad|Registro|Junta|Universidad|Ministerio|Complejo\s+Hospitalario)\b', re.I)


def overlaps(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def contains(outer, inner):
    return outer['start'] <= inner['start'] and inner['end'] <= outer['end']


def literal_pattern(name):
    return r'(?<!\w)' + r'\s+'.join(re.escape(part) for part in name.split()) + r'(?!\w)'


def valid_date(text):
    words = re.fullmatch(rf'(\d{{1,2}})\s+(?:de\s+)?({MONTHS})(?:\s+(?:de\s+)?((?:19|20)\d{{2}}))?', text, re.I)
    numeric = re.fullmatch(r'(\d{1,2})([/.-])(\d{1,2})\2((?:19|20)\d{2})', text)
    try:
        if words:
            day, month, year = words.groups()
            date(int(year) if year else 2024, MONTH_NUMBER[month.lower()], int(day))
        elif numeric:
            day, _, month, year = numeric.groups()
            date(int(year), int(month), int(day))
        else:
            return False
    except ValueError:
        return False
    return True


class HybridRules:
    """Deterministic proposals and conflict decisions; never reads gold labels."""
    def __init__(self, rule_config):
        self.config = copy.deepcopy(rule_config)
        if self.config['version'] != RULE_VERSION:
            raise ValueError('Rule code and saved configuration versions differ.')
        self.patterns = []

        def add(rule_id, label, pattern, priority):
            self.patterns.append((rule_id, label, re.compile(pattern, re.I), priority))

        # Full names/acronyms and certificate references precede shorter names.
        add('platform_full_name', 'ORG', r'\bPlataforma\s+de\s+Contrataci[oó]n\s+del\s+Sector\s+P[uú]blico(?:\s*\(PLACSP\))?(?!\w)', 95)
        add('rolece_certificate', 'MISC', r'\bCertificado(?:\s+(?:de|del))?\s+ROLECE\b', 100)
        for name in self.config['org_names']:
            add('organization:' + name, 'ORG', literal_pattern(name), 80)
        for name in self.config['loc_names']:
            add('place:' + name, 'LOC', literal_pattern(name), 50)
        if self.config['seguridad_social_context_rule']:
            add('seguridad_social_institution', 'ORG', r'\bSeguridad\s+Social\b', 80)
        if self.config['enable_law_forms']:
            add('article_lcsp', 'LAW', rf'\b{ARTICLE_PREFIX}LCSP(?:\s+{LEGAL_ID})?(?![\w/])', 110)
            add('article_regulation', 'LAW', rf'\b{ARTICLE_PREFIX}Reglamento\s+{LEGAL_ID}(?![\w/])', 110)
            add('directive_identifier', 'LAW', r'\bDirectiva\s+\d{2,4}/\d+(?:/(?:CE|UE|CEE))(?![\w/])', 110)
            add('lcsp_identifier', 'LAW', rf'\bLCSP(?:\s+{LEGAL_ID})?(?![\w/])', 90)
        if self.config['enable_email']:
            add('email_address', 'EMAIL', r'(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.[A-Z]{2,}(?![\w-])', 105)
        if self.config['enable_technical_references']:
            add('technical_standard', 'MISC', r'\b(?:UNE(?:-EN)?(?:\s+ISO(?:/IEC)?)?|ISO(?:/IEC)?)\s+\d+(?:-\d+)?(?::\d{4})?(?![\w/])', 100)
            add('cpv_description', 'MISC', r'\bC[oó]digo\s+CPV\s*:\s*\d{8}(?:-\d)?(?:\s+(?:“[^”\n]+”|«[^»\n]+»|"[^"\n]+"))?(?!\w)', 100)
        if self.config['enable_schedules']:
            add('weekday_hours', 'SCHEDULE', rf'\b{DAY}(?:\s+a\s+{DAY})?\s+de\s+{TIME}\s+a\s+{TIME}(?!\w)', 95)
            add('abbreviated_hours', 'SCHEDULE', rf'\bL\s*[-–]\s*J\s+{TIME}\s+a\s+{TIME}(?:\s*;\s*V\s+{TIME}\s+a\s+{TIME})?(?!\w)', 95)
            add('clock_time', 'SCHEDULE', rf'(?<![\w:]){TIME}(?![\w:])', 70)
        if self.config['enable_dates']:
            add('spanish_calendar_date', 'DATE', rf'(?<!\w){DATE_WORDS}(?!\w)', 60)
            add('numeric_calendar_date', 'DATE', r'(?<![\w/.-])(?:0?[1-9]|[12]\d|3[01])([/.-])(?:0?[1-9]|1[0-2])\1(?:19|20)\d{2}(?![\w/.-])', 60)

    def __call__(self, doc):
        original = [{'start': e.start_char, 'end': e.end_char, 'label': e.label_, 'text': e.text, 'span': e}
                    for e in doc.ents]
        doc.spans['aner_ml_original'] = list(doc.ents)
        audit, candidates = [], []
        for rule_id, label, pattern, priority in self.patterns:
            for match in pattern.finditer(doc.text):
                item = {'rule_id': rule_id, 'start': match.start(), 'end': match.end(),
                        'label': label, 'text': match.group(), 'priority': priority}
                reason = None
                if label == 'DATE' and not valid_date(item['text']):
                    reason = 'invalid_calendar_date'
                if label == 'DATE':
                    before = doc.text[max(0, match.start() - 120):match.start()]
                    # A publication date attached to an explicit legal citation
                    # is not a standalone DATE, even when ML missed the LAW.
                    anchors = list(LEGAL_ANCHOR.finditer(before))
                    if anchors and re.fullmatch(r'(?:\s+\d+/\d{4})?\s*,\s*(?:de\s+)?', before[anchors[-1].end():], re.I):
                        reason = 'date_attached_to_explicit_legal_citation'
                if rule_id == 'seguridad_social_institution':
                    before = doc.text[max(0, match.start() - 40):match.start()]
                    if not re.search(r'\b(?:con|ante)\s+(?:la\s+)?$', before, re.I):
                        reason = 'institution_context_not_established'
                if label == 'LAW':
                    after = doc.text[match.end():match.end() + 60]
                    # Abstain from partial new citations followed by an unhandled
                    # attached date/title. Existing longer LAW spans are retained.
                    if re.match(r'(?:,\s*de\s+|\s*,?\s+reguladora\b)', after, re.I):
                        reason = 'citation_continues_beyond_supported_rule'
                if label == 'LOC':
                    before = doc.text[max(0, match.start() - 65):match.start()]
                    if re.search(r'\b(?:Ayuntamiento|Universidad|Diputaci[oó]n|Cabildo)\s+(?:de|del)\s+$', before, re.I):
                        reason = 'place_is_inside_an_institution_name'
                span = doc.char_span(item['start'], item['end'], label=label, alignment_mode='strict')
                if span is None:
                    reason = 'not_strictly_token_aligned'
                if reason:
                    audit.append({**item, 'decision': 'skipped', 'reason': reason})
                else:
                    candidates.append({**item, 'span': span})

        # Full, high-priority references win over their own shorter candidates.
        selected = []
        for candidate in sorted(candidates, key=lambda x: (-x['priority'], -(x['end'] - x['start']), x['start'])):
            blockers = [e for e in selected if overlaps(e, candidate)]
            reason = 'overlaps_preferred_rule' if blockers else None
            for entity in original:
                if not contains(entity, candidate) or entity['end'] - entity['start'] <= candidate['end'] - candidate['start']:
                    continue
                if entity['label'] == 'LAW' and LEGAL_ANCHOR.search(entity['text']):
                    extra = doc.text[entity['start']:candidate['start']] + doc.text[candidate['end']:entity['end']]
                    if candidate['label'] != 'LAW' or extra.strip(' \t\r\n.,;:!?()[]\"\'«»“”'):
                        reason = 'inside_complete_ml_legal_reference'
                if entity['label'] == 'ORG' and ORG_PREFIX.match(entity['text']) and candidate['label'] in {'ORG', 'LOC'}:
                    # A rule can fix trailing sentence punctuation, but a nested
                    # place/acronym must not replace a complete institution name.
                    extra = doc.text[entity['start']:candidate['start']] + doc.text[candidate['end']:entity['end']]
                    if extra.strip(' \t\r\n.,;:!?'):
                        reason = 'inside_complete_ml_organization'
                if entity['label'] == 'MISC' and candidate['label'] == 'ORG' and re.match(r'^Certificado\b', entity['text'], re.I):
                    reason = 'inside_ml_certificate_reference'
            if reason:
                audit.append({**{k: v for k, v in candidate.items() if k != 'span'}, 'decision': 'skipped', 'reason': reason})
            else:
                selected.append(candidate)

        kept = []
        for entity in original:
            replacing = [r for r in selected if overlaps(entity, r)]
            generic_social_subject = (entity['label'] == 'ORG' and re.fullmatch(r'Seguridad\s+Social', entity['text'], re.I)
                                      and re.search(r'\ben\s+materia\s+de\s+(?:la\s+)?$', doc.text[max(0, entity['start'] - 45):entity['start']], re.I))
            if replacing or generic_social_subject:
                audit.append({'start': entity['start'], 'end': entity['end'], 'label': entity['label'],
                              'text': entity['text'], 'decision': 'removed_ml_span',
                              'reason': 'overlap_with_selected_rule' if replacing else 'generic_subject_area',
                              'replacing_rules': [r['rule_id'] for r in replacing]})
            else:
                kept.append(entity['span'])
        for selected_rule in selected:
            audit.append({**{k: v for k, v in selected_rule.items() if k != 'span'}, 'decision': 'accepted'})
        doc.ents = sorted(kept + [r['span'] for r in selected], key=lambda e: e.start_char)
        doc._.aner_hybrid_audit = audit
        return doc


if not Doc.has_extension('aner_hybrid_audit'):
    Doc.set_extension('aner_hybrid_audit', default=None)


def create_component(nlp, name, rule_config):
    return HybridRules(rule_config)


if not Language.has_factory(COMPONENT_NAME):
    Language.factory(COMPONENT_NAME, default_config={'rule_config': DEFAULT_RULE_CONFIG})(create_component)


def build_hybrid(baseline, rule_config=None):
    config = copy.deepcopy(rule_config or DEFAULT_RULE_CONFIG)
    if COMPONENT_NAME in baseline.pipe_names:
        raise ValueError('Supply the ML-only baseline, not an already modified hybrid pipeline.')
    hybrid = copy.deepcopy(baseline)
    hybrid.add_pipe(COMPONENT_NAME, last=True, config={'rule_config': config})
    return hybrid


def build_rules_only(baseline, rule_config=None):
    rules_only = spacy.blank('es')
    rules_only.tokenizer.from_bytes(baseline.tokenizer.to_bytes())
    rules_only.add_pipe(COMPONENT_NAME, config={'rule_config': copy.deepcopy(rule_config or DEFAULT_RULE_CONFIG)})
    return rules_only


def run_experiment_2(ns):
    """Use experiment-1 variables from the current Colab runtime."""
    needed = ['trained_model', 'score_rows', 'split_rows', 'split_indices', 'sources',
              'RUN_DIR', 'run_metadata', 'SPLIT_SHA256', 'APPROVED_SHA256', 'LABELS']
    missing = [name for name in needed if ns.get(name) is None]
    if missing:
        raise RuntimeError('Run experiment 1 in this runtime first; missing: ' + ', '.join(missing))
    baseline = ns['trained_model']
    score_rows = ns['score_rows']
    assert ns['run_metadata']['split_manifest_sha256'] == ns['SPLIT_SHA256']
    assert ns['run_metadata']['annotation_sha256'] == ns['APPROVED_SHA256']
    before = baseline.to_bytes()
    hybrid = build_hybrid(baseline)
    rules_only = build_rules_only(baseline)
    assert baseline.to_bytes() == before, 'The ML baseline was changed while constructing the hybrid.'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = Path(ns['RUN_DIR']) / 'experiment_2' / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    config_bytes = json.dumps(DEFAULT_RULE_CONFIG, sort_keys=True, ensure_ascii=False).encode('utf-8')
    rule_source = Path(ns.get('_ANER_EXPERIMENT_2_SOURCE', 'ANER_Experiment_2.py'))
    metadata = {'experiment': 2, 'method': 'same ML checkpoint plus inference-time regex/dictionary rules',
                'baseline_run_metadata': ns['run_metadata'], 'annotation_sha256': ns['APPROVED_SHA256'],
                'split_manifest_sha256': ns['SPLIT_SHA256'], 'rule_version': RULE_VERSION,
                'rule_config': DEFAULT_RULE_CONFIG, 'rule_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                'rule_source_sha256': hashlib.sha256(rule_source.read_bytes()).hexdigest() if rule_source.is_file() else None,
                'rule_development_source': 'training examples and the user-reviewed evaluation errors',
                'same_split_and_scoring_as_baseline': True, 'retraining_performed': False,
                'test_previously_inspected': True, 'test_comparison_is_exploratory': True,
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    save('experiment_metadata.json', metadata)
    models = {'ML baseline': baseline, 'Rules only': rules_only, 'ML + rules': hybrid}
    evaluation_reports = {name: score_rows(model, ns['split_rows']['evaluation'], include_predictions=True)
                          for name, model in models.items()}
    comparison = []
    per_label = []
    for name, report in evaluation_reports.items():
        comparison.append({'system': name, **report['micro']})
        for label in ns['LABELS']:
            per_label.append({'system': name, 'label': label, **report['per_label'][label]})
        for row, index in zip(report['predictions'], ns['split_indices']['evaluation']):
            row['original_annotation_index'] = index
            row['source'] = ns['sources'][index]
        save(name.lower().replace(' ', '_').replace('+', 'plus') + '_evaluation.json', report)
    print('EVALUATION — same checkpoint, same records, exact-span scoring')
    display(pd.DataFrame(comparison).set_index('system'))
    display(pd.DataFrame(per_label).set_index(['label', 'system'])[['precision', 'recall', 'f1', 'gold_support']])
    save('evaluation_comparison.json', {'overall': comparison, 'per_label': per_label})

    # Match experiment 1's secondary diagnostic: keep the first original
    # passage per whitespace/case-normalized text, without modifying its text
    # or gold offsets. Repeated boilerplate then contributes only once.
    seen, unique_rows, unique_indices = set(), [], []
    for row, index in zip(ns['split_rows']['evaluation'], ns['split_indices']['evaluation']):
        key = re.sub(r'\s+', ' ', row[0].casefold()).strip()
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
            unique_indices.append(index)
    unique_reports = {name: score_rows(model, unique_rows) for name, model in models.items()}
    save('evaluation_normalized_unique_comparison.json',
         {'normalization': 'casefold; collapse whitespace; strip; retain first original passage',
          'original_annotation_indices': unique_indices, 'reports': unique_reports})
    print('NORMALIZED UNIQUE EVALUATION TEXTS:', len(unique_rows))
    display(pd.DataFrame([{'system': name, **report['micro']} for name, report in unique_reports.items()]).set_index('system'))

    changes = []
    base_predictions = evaluation_reports['ML baseline']['predictions']
    for doc, base_row, index in zip(hybrid.pipe((r[0] for r in ns['split_rows']['evaluation']), batch_size=32),
                                    base_predictions, ns['split_indices']['evaluation']):
        gold = {tuple(e) for e in base_row['gold']}
        old = {tuple(e) for e in base_row['predicted']}
        new = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        if old != new or any(item['decision'] == 'skipped' and item['reason'] == 'not_strictly_token_aligned'
                             for item in doc._.aner_hybrid_audit):
            changes.append({'original_annotation_index': index, 'source': ns['sources'][index], 'text': doc.text,
                            'recovered_gold': sorted((new & gold) - old), 'lost_gold': sorted((old & gold) - new),
                            'removed_false_positives': sorted((old - gold) - new),
                            'introduced_false_positives': sorted((new - gold) - old),
                            'rule_audit': doc._.aner_hybrid_audit})
    save('evaluation_rule_changes.json', changes)
    print('Changed evaluation passages:', len(changes))
    print('Recovered gold:', sum(len(r['recovered_gold']) for r in changes),
          '| lost previously correct gold:', sum(len(r['lost_gold']) for r in changes))
    print('Introduced false positives:', sum(len(r['introduced_false_positives']) for r in changes),
          '| removed false positives:', sum(len(r['removed_false_positives']) for r in changes))
    # Save and verify that a reloaded pipeline reproduces this hybrid's predictions.
    hybrid.to_disk(output_dir / 'hybrid-model')
    reloaded = spacy.load(output_dir / 'hybrid-model')
    original_predictions = evaluation_reports['ML + rules']['predictions']
    for doc, row in zip(reloaded.pipe((r[0] for r in ns['split_rows']['evaluation']), batch_size=32), original_predictions):
        assert [(e.start_char, e.end_char, e.label_) for e in doc.ents] == [tuple(e) for e in row['predicted']]

    if RUN_HYBRID_TEST:
        test_results = {name: score_rows(model, ns['split_rows']['test']) for name, model in models.items()}
        save('exploratory_test_comparison.json', test_results)
        print('EXPLORATORY TEST COMPARISON — this test set was previously inspected')
        display(pd.DataFrame([{'system': name, **r['micro']} for name, r in test_results.items()]).set_index('system'))
    else:
        print('Test scoring is disabled. Review evaluation changes before enabling RUN_HYBRID_TEST.')
    print('Saved comparison, rule configuration, changes and hybrid model:', output_dir)
    return {'hybrid_model': hybrid, 'rules_only_model': rules_only,
            'hybrid_evaluation_reports': evaluation_reports, 'hybrid_rule_changes': changes,
            'hybrid_unique_evaluation_reports': unique_reports,
            'HYBRID_RUN_DIR': output_dir}


if __name__ == '__main__':
    globals().update(run_experiment_2(globals()))
