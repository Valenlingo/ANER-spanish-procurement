# ANER Colab notebook Python export
# Generated from the final notebook; cell order and code are preserved.

# %% [markdown]
# # ANER: Spanish public procurement named entity recognition
# 
# **Author:** Valentín Barbosa · **Guidelines:** 2.1 · **Annotated dataset:** revision 3
# 
# This notebook rebuilds a coursework NER project as a documented, reproducible
# portfolio experiment. The objective is to recognize useful named entities and
# procurement references in Spanish public procurement texts. We use a small
# statistical spaCy baseline trained from scratch on the revised annotations.
# 
# Work through the numbered stages separately: **project → acquisition and
# exploration → annotation review → 50/25/25 split → training → test → analysis**.
# The setup cell controls training and testing; both were enabled for the saved run.
# No earlier coursework scores are reused as results of this experiment.
# 
# Each cell also has machine-readable `metadata.aner` fields documenting its step,
# purpose, inputs and outputs. Generated JSON reports record data hashes, original
# source locations, split assignments, software versions and training settings.

# %% [markdown]
# ## 1. Entities and annotation guidelines
# 
# | Label | Include | Example |
# | --- | --- | --- |
# | ORG | Named organizations, public bodies, companies and institutional platforms/registries | Ayuntamiento de Alcántara; PLACSP |
# | PER | Named people; exclude generic roles and job titles | María García |
# | LOC | Geographic names and named physical facilities, infrastructure or areas | Palma del Río; EBAR del Hombre del Mar; ERRP TORREVIEJA |
# | DATE | Calendar dates and standalone calendar years | 29 de octubre de 2024; 2027 |
# | LAW | Identifiable statutes, regulations/directives and legal acronyms, with attached articles, dates or titles | artículo 159.6.b) de la LCSP; Directiva 2007/46/CE |
# | EMAIL | Complete email address, without surrounding punctuation | licitacionE@hacienda.gob.es |
# | SCHEDULE | Clock times, time windows and recurring hours | 12:00h; lunes a jueves de 9:00 a 19:00 |
# | EVENT | Particular disasters, pandemics or other identifiable events in context | depresión aislada en niveles altos (DANA), referring to a specific disaster |
# | MISC | Procurement documents, clauses/annexes, plans, mechanisms, standards/certificates, codes and approved eligibility categories | Anexo II del PCAP; SDA; SAJV-1; Plan Sanitario frente a Legionella |
# 
# **Boundaries:** one label per coherent span; no overlaps or nested entities.
# Exclude outer spaces and sentence punctuation; preserve internal identifier
# punctuation. A continuous statute citation includes its attached article/date/title
# as LAW, with no nested DATE. Separate statutes receive separate LAW spans.
# Contract documents and their clauses/annexes remain MISC. An article reference
# without an identifiable law is MISC. Use specific labels before MISC.
# 
# **Context matters:** generic roles, amounts, durations, obligations and subject
# areas stay untagged. `Seguridad Social` is ORG when it denotes an institution,
# but is untagged in `en materia de Seguridad Social`. Eligibility categories such
# as `CENTROS ESPECIALES DE EMPLEO DE INICIATIVA SOCIAL` are MISC, not named
# organizations. A named physical area/facility is LOC; a nonphysical initiative
# designation can be MISC. A generic weather phenomenon or disease name is not an
# EVENT; a particular DANA disaster or the COVID pandemic can be EVENT.
# `Legionella` alone stays untagged; the complete sanitary-plan name is MISC.
# 
# **Corpus policy:** Spanish only. Exclude blank records, other-language records,
# whole records referring to `hora/horario peninsular`, and unusable truncated
# fragments. Keep valid short headings and texts without entities. Review truncation
# before annotation; never silently change annotated text or expand/shrink spans
# to satisfy tokenization. Offsets count Unicode characters: start inclusive,
# end exclusive. The full approved guideline is in `docs/ANER_Annotation_Guidelines.docx`.
# 
# The current corpus has no PER examples. EVENT has three copies of one passage.
# These are schema categories, not evidence of adequate training/evaluation coverage.

# %% [markdown]
# ### Setup in Google Colab
# 
# Raw procurement texts are downloaded directly from the official public websites
# in stage 2. You do not need to upload the four raw text or Atom files from your PC.
# 
# The manually revised annotations and grouping review are project inputs. Run from
# a repository checkout containing `data/annotations/`, or upload
# `ANER_AI_Assisted_Annotations.json` and `grouping_review.json` when prompted.
# The existing `ANER_Colab_Project.zip` is also accepted for convenience.
# 
# `DOWNLOAD_SOURCE = True` is the default. Set it to `False` only to explicitly use
# the frozen `data/raw/` snapshots included with the project. The saved run used
# `RUN_TRAINING = True` and `RUN_FINAL_TEST = True`; disable them for staged review.
# The notebook uses **evaluation** for development and **test** for the separate
# test partition. No personal Google Drive paths are used.

# %%
import subprocess
import sys

# Colab setup: run once before importing spaCy.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'spacy==3.8.16', 'requests', 'beautifulsoup4', 'lxml', 'pandas'], check=True)


# %%
from collections import Counter, defaultdict
from datetime import datetime, timezone
import sys
import hashlib
import importlib.metadata
import json
import random
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from IPython.display import display, Markdown
from spacy.symbols import ORTH
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

SEED = 42
SAMPLE_CHARACTERS = 25_000
SPLIT_RATIOS = {'train': 0.50, 'evaluation': 0.25, 'test': 0.25}
DOWNLOAD_SOURCE = True    # Download the four original Atom documents from the official website.
RUN_TRAINING = True      # Enable after inspecting the prepared data and split.
RUN_FINAL_TEST = True    # Enable only after choosing the model using evaluation data.
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
DROPOUT = 0.2
APPROVED_SHA256 = '5d0196d2eaebcc9b4d8178b61a288eb86f6114c9e2bec2c04f9f28c829ada989'

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

# Locate project annotation inputs; raw data are gathered online in stage 2.
required_inputs = ['ANER_AI_Assisted_Annotations.json', 'grouping_review.json']
candidate_roots = [Path.cwd(), Path.cwd() / 'ANER_Colab_Project', Path.cwd().parent]
PROJECT_ROOT = next((p for p in candidate_roots
                     if all((p / 'data/annotations' / name).is_file() for name in required_inputs)), None)
if PROJECT_ROOT is None:
    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError('Run from a project containing the two data/annotations JSON files.') from error
    print('Upload the two annotation JSON files, or ANER_Colab_Project.zip. Raw text files are not required.')
    uploaded = files.upload()
    bundles = [name for name in uploaded if name.endswith('.zip')]
    if bundles:
        if len(bundles) != 1 or len(uploaded) != 1:
            raise ValueError('Upload one project ZIP, or the two JSON files separately.')
        with zipfile.ZipFile(bundles[0]) as archive:
            base = Path.cwd().resolve()
            for member in archive.infolist():
                target = (base / member.filename).resolve()
                if not target.is_relative_to(base) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP contains an unsupported path.')
            archive.extractall(base)
        PROJECT_ROOT = Path.cwd() / 'ANER_Colab_Project'
    else:
        if not all(name in uploaded for name in required_inputs):
            raise ValueError('Upload ANER_AI_Assisted_Annotations.json and grouping_review.json.')
        PROJECT_ROOT = Path.cwd() / 'ANER_Colab_Project'
        annotation_folder = PROJECT_ROOT / 'data/annotations'
        annotation_folder.mkdir(parents=True, exist_ok=True)
        for name in required_inputs:
            (annotation_folder / name).write_bytes(uploaded[name])
    if not all((PROJECT_ROOT / 'data/annotations' / name).is_file() for name in required_inputs):
        raise FileNotFoundError('The project is missing its annotation JSON or grouping review.')

RAW_DIR = PROJECT_ROOT / 'data/raw'
ANNOTATION_DIR = PROJECT_ROOT / 'data/annotations'
WEB_RAW_DIR = PROJECT_ROOT / 'data/downloaded'
PROCESSED_DIR = PROJECT_ROOT / 'data/processed'
REPORT_DIR = PROJECT_ROOT / 'reports'
for folder in [PROCESSED_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
environment = {'python': sys.version.split()[0], 'spacy': spacy.__version__,
               'pandas': pd.__version__, 'requests': requests.__version__,
               'beautifulsoup4': importlib.metadata.version('beautifulsoup4'),
               'seed': SEED, 'recorded_at_utc': utc_now()}
save_json(REPORT_DIR / 'environment.json', environment)
print('Project:', PROJECT_ROOT)
print(environment)

# The approved annotation metadata holds the original raw-text checksums.
annotation_input = ANNOTATION_DIR / 'ANER_AI_Assisted_Annotations.json'
assert sha256(annotation_input) == APPROVED_SHA256, 'Annotation file differs from the approved revision.'
source_metadata = json.loads(annotation_input.read_text(encoding='utf-8'))['metadata']['source_files']
EXPECTED_SAMPLE_SHA256 = {item['filename']: item['sha256_original_utf8'] for item in source_metadata}


# %% [markdown]
# ## 2. Web scraping, extraction and exploration
# 
# The source is the Ministry of Finance's
# [official procurement catalog](https://www.hacienda.gob.es/es-ES/GobiernoAbierto/Datos%20Abiertos/Paginas/LicitacionesContratante.aspx).
# The code finds the **January 2025 archive link** on that page, then downloads the
# same four original Atom documents directly from the publisher. Their individual
# URLs are listed in the acquisition code. Downloading the four documents avoids
# transferring the complete monthly archive.
# 
# Extract `cbc:Description` in document order, append a newline after each field,
# and preserve the original coursework's universal-newline handling (`CRLF`/`CR`
# become `LF` when its text file was read in Python). Apply this step **before**
# taking the first **25,000 Unicode characters** of each document. There is no
# additional whitespace cleanup, language filtering or text normalization here.
# 
# Downloaded Atom files, complete extracted texts and the four samples are saved
# in `data/downloaded/`. Exploration and annotation validation read those newly
# downloaded samples. Their checksums must match the original checksums recorded
# in the approved annotations. A mismatch stops the workflow: offsets cannot be
# reused on changed text.
# 
# `DOWNLOAD_SOURCE = True` is the default. An explicit `False` uses frozen
# `data/raw/` snapshots for offline reproduction, with the same checksum checks.
# There is no automatic switch to local snapshots if the website fails. Source
# URLs, download times, Atom checksums, extraction counts and sample checksums are
# recorded in `reports/acquisition.json`.
# 
# The saved model results below belong to the recorded experiment. All four online
# samples were checked against its annotated snapshots and matched exactly; the
# revised acquisition cells have their old outputs cleared for rerunning.

# %%
import xml.etree.ElementTree as ET

ATOM_FILENAMES = [
    'licitacionesPerfilesContratanteCompleto3_20250108_182854_5.atom',
    'licitacionesPerfilesContratanteCompleto3_20250121_145342.atom',
    'licitacionesPerfilesContratanteCompleto3_20250122_141634_2.atom',
    'licitacionesPerfilesContratanteCompleto3_20250128_175733_1.atom',
]
SAMPLE_FILENAMES = [f'raw_data_sample{n}.txt' for n in range(2, 6)]
CATALOG_URLS = [
    'https://www.hacienda.gob.es/es-ES/GobiernoAbierto/Datos%20Abiertos/Paginas/LicitacionesContratante.aspx',
    'https://www.hacienda.gob.es/gl-ES/GobiernoAbierto/Datos%20Abiertos/Paginas/LicitacionesContratante.aspx',
]
ARCHIVE_FILENAME = 'licitacionesPerfilesContratanteCompleto3_202501.zip'
OFFICIAL_SOURCE_BASE = 'https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/'
ATOM_URLS = [urljoin(OFFICIAL_SOURCE_BASE, name) for name in ATOM_FILENAMES]
CBC_DESCRIPTION = '{urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2}Description'
ATOM_ENTRY = '{http://www.w3.org/2005/Atom}entry'

def find_archive_link(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = [urljoin(base_url, tag['href']) for tag in soup.find_all('a', href=True)]
    return next((url for url in links if url.split('?')[0].endswith(ARCHIVE_FILENAME)), None)

def extract_descriptions(atom_path):
    # Streaming XML parsing keeps memory bounded while preserving field order.
    descriptions = []
    for _, element in ET.iterparse(atom_path, events=('end',)):
        if element.tag == CBC_DESCRIPTION:
            descriptions.append(''.join(element.itertext()))
        elif element.tag == ATOM_ENTRY:
            element.clear()
    if not descriptions:
        raise ValueError('No cbc:Description fields found; inspect the XML namespace.')
    text = ''.join(value + '\n' for value in descriptions)
    # Reproduce the original read of the extracted UTF-8 text, before sampling.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text, len(descriptions)

def download_original_samples():
    archive_url, catalog_url, catalog_errors = None, None, []
    for url in CATALOG_URLS:
        try:
            response = requests.get(url, timeout=(20, 60))
            response.raise_for_status()
            found = find_archive_link(response.text, response.url)
            if found:
                archive_url, catalog_url = found, response.url
                break
            catalog_errors.append('January 2025 archive link absent from ' + url)
        except requests.RequestException as error:
            catalog_errors.append(str(error))
    if archive_url is None:
        raise RuntimeError('Could not verify the original archive in the official catalog. ' + '; '.join(catalog_errors))
    source_base = archive_url.rsplit('/', 1)[0] + '/'
    WEB_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for atom_name, sample_name in zip(ATOM_FILENAMES, SAMPLE_FILENAMES):
        atom_url = urljoin(source_base, atom_name)
        atom_path = WEB_RAW_DIR / atom_name
        temporary_path = atom_path.with_suffix('.atom.part')
        print('Downloading:', atom_url, flush=True)
        try:
            with requests.get(atom_url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                final_url = response.url
                with temporary_path.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
            temporary_path.replace(atom_path)
        except requests.RequestException:
            temporary_path.unlink(missing_ok=True)
            raise
        text, description_count = extract_descriptions(atom_path)
        if len(text) < SAMPLE_CHARACTERS:
            raise ValueError(atom_name + ' contains fewer than 25,000 extracted characters.')
        full_text_path = WEB_RAW_DIR / (Path(atom_name).stem + '.txt')
        full_text_path.write_bytes(text.encode('utf-8'))
        sample_path = WEB_RAW_DIR / sample_name
        sample_path.write_bytes(text[:SAMPLE_CHARACTERS].encode('utf-8'))
        rows.append({'atom_file': atom_name, 'source_url': atom_url,
                     'download_url': final_url, 'downloaded_at_utc': utc_now(),
                     'atom_sha256': sha256(atom_path), 'sample_file': sample_name,
                     'description_count': description_count, 'full_characters': len(text),
                     'sample_characters': SAMPLE_CHARACTERS, 'sample_sha256': sha256(sample_path),
                     'expected_sample_sha256': EXPECTED_SAMPLE_SHA256[sample_name],
                     'matches_annotated_snapshot': sha256(sample_path) == EXPECTED_SAMPLE_SHA256[sample_name]})
    report = {'mode': 'web_download_original_atom_documents', 'retrieved_at_utc': utc_now(),
              'new_web_download_performed': True, 'catalog_url': catalog_url,
              'archive_reference_url': archive_url, 'input_directory': str(WEB_RAW_DIR),
              'extraction_field': 'cbc:Description', 'separator': 'LF after every description',
              'newline_handling': 'CRLF/CR to LF, reproducing original Python text-file read before sampling',
              'documents': rows}
    save_json(REPORT_DIR / 'acquisition.json', report)
    if not all(row['matches_annotated_snapshot'] for row in rows):
        raise ValueError('Downloaded text differs from the annotated snapshots. Review and annotate it separately.')
    return report

if DOWNLOAD_SOURCE:
    acquisition_report = download_original_samples()
    SOURCE_DIR = WEB_RAW_DIR
else:
    SOURCE_DIR = RAW_DIR
    rows = []
    for atom, url, sample in zip(ATOM_FILENAMES, ATOM_URLS, SAMPLE_FILENAMES):
        path = SOURCE_DIR / sample
        if not path.is_file():
            raise FileNotFoundError('Offline mode needs frozen data/raw/ snapshots; set DOWNLOAD_SOURCE = True to gather them online.')
        assert sha256(path) == EXPECTED_SAMPLE_SHA256[sample], 'Frozen sample differs from the approved source.'
        rows.append({'atom_file': atom, 'source_url': url, 'sample_file': sample,
                     'sample_sha256': sha256(path), 'matches_annotated_snapshot': True})
    acquisition_report = {'mode': 'explicit_offline_snapshots', 'recorded_at_utc': utc_now(),
                          'new_web_download_performed': False, 'input_directory': str(SOURCE_DIR),
                          'documents': rows}
    save_json(REPORT_DIR / 'acquisition.json', acquisition_report)
display(pd.DataFrame(acquisition_report['documents']))


# %%
# Explore all four 25,000-character samples; counts are characters and tokens separately.
exploration_nlp = spacy.blank('es')
sample_texts = {}
exploration_rows = []
for sample_name in SAMPLE_FILENAMES:
    text = (SOURCE_DIR / sample_name).read_bytes().decode('utf-8')
    assert len(text) == SAMPLE_CHARACTERS, sample_name
    sample_texts[sample_name] = text
    doc = exploration_nlp.make_doc(text)
    token_count = sum(not token.is_space for token in doc)
    words = [token.lower_ for token in doc if token.is_alpha and not token.is_stop]
    exploration_rows.append({'sample': sample_name, 'characters': len(text),
                             'physical_lines': len(text.splitlines()), 'tokens': token_count,
                             'content_word_tokens': len(words), 'content_word_types': len(set(words)),
                             'content_word_TTR': len(set(words)) / len(words) if words else 0,
                             'top_10_content_words': Counter(words).most_common(10)})
exploration = pd.DataFrame(exploration_rows)
display(exploration.drop(columns='top_10_content_words'))
for row in exploration_rows:
    print(row['sample'], row['top_10_content_words'])
save_json(REPORT_DIR / 'corpus_exploration.json', {'scope': 'four frozen samples, not the full archive',
                                                'rows': exploration_rows})
print('First passage:', sample_texts[SAMPLE_FILENAMES[0]][:300])


# %% [markdown]
# ## 3. AI-assisted annotation and human review
# 
# The first annotations were produced with AI assistance and revised through
# discussion of difficult cases. The user reviewed examples and chose the scope
# of LAW, LOC, MISC, EVENT and the other labels. Guidelines version 2.1 records
# those decisions. This notebook **loads the resulting annotation file**; it does
# not ask an AI to regenerate labels during training.
# 
# Before the revision, unusable clipped endings were removed or shortened. The
# final corpus has **1,165 records and 605 entity spans**. Across the preparation
# history, 58 records were excluded (27 non-Spanish, 19 blank, 8 peninsular-time
# references and 4 unusable fragments). Two retained records were shortened at
# reviewed boundaries. Valid headings, wrapped continuations and negative examples
# remain. The JSON metadata retains exclusion reasons, trimming/annotation changes
# and original file/line locations.
# 
# These are **AI-assisted, user-reviewed decisions**, not independent agreement
# between two human annotators. Model evaluation measures agreement with this
# annotation set; its quality and limited coverage constrain the conclusions.
# Before proceeding, inspect the counts and several records below.

# %%
annotation_path = ANNOTATION_DIR / 'ANER_AI_Assisted_Annotations.json'
assert sha256(annotation_path) == APPROVED_SHA256, 'Annotation file differs from the approved revision.'
dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
records = dataset['annotations']
sources = dataset['metadata']['record_sources']
LABELS = dataset['classes']

def make_span_tokenizer(nlp):
    # Preserve gold boundaries for the two reviewed Spanish tokenizer cases.
    nlp.tokenizer.add_special_case('A.', [{ORTH: 'A'}, {ORTH: '.'}])
    nlp.tokenizer.add_special_case('2007/46/CE),para',
                                 [{ORTH: part} for part in ['2007/46/CE', ')', ',', 'para']])
    return nlp

tokenizer_nlp = make_span_tokenizer(spacy.blank('es'))
entity_counts = Counter()
assert len(records) == len(sources)
for i, ((text, annotation), source) in enumerate(zip(records, sources)):
    raw = sample_texts[source['source_file']]
    assert text == raw[source['retained_source_start']:source['retained_source_end_exclusive']], i
    doc = tokenizer_nlp.make_doc(text)
    previous_end = 0
    for start, end, label in annotation['entities']:
        assert label in LABELS and 0 <= start < end <= len(text) and start >= previous_end, i
        assert doc.char_span(start, end, label=label, alignment_mode='strict') is not None, (i, text[start:end])
        entity_counts[label] += 1
        previous_end = end
for item in dataset['metadata']['source_files']:
    assert sha256(SOURCE_DIR / item['filename']) == item['sha256_original_utf8']
display(pd.DataFrame([{'label': label, 'spans': entity_counts[label]} for label in LABELS]))
validation = {'records': len(records), 'entity_spans': sum(entity_counts.values()),
              'records_without_entities': sum(not a['entities'] for _, a in records),
              'annotation_sha256': APPROVED_SHA256, 'source_substrings_verified': True,
              'strict_token_alignment_verified': True, 'exclusion_counts': dataset['metadata']['exclusion_counts'],
              'guideline': dataset['metadata']['guideline']}
save_json(REPORT_DIR / 'annotation_validation.json', validation)
print(validation)

def review_record(index=None, source_file=None, line_number=None):
    if index is None:
        matches = [i for i, source in enumerate(sources)
                   if source['source_file'] == source_file and source['line_number'] == line_number]
        if len(matches) != 1:
            raise ValueError('No unique retained record found at that original source location.')
        index = matches[0]
    text, annotation = records[index]
    print('Combined annotation index:', index, '| original source:', sources[index])
    print(text)
    display(pd.DataFrame([{'start': start, 'end': end, 'label': label, 'text': text[start:end]}
                          for start, end, label in annotation['entities']],
                         columns=['start', 'end', 'label', 'text']))

review_record(source_file='raw_data_sample2.txt', line_number=188)
# Change the original source file and line above to inspect a different case.


# %% [markdown]
# ## 4. Split the data: 50% training, 25% evaluation, 25% test
# 
# We split **records**, not entities. Identical passages and identified close
# template variants/continuations must stay together. The bundled grouping review
# contains 572 text families, link reasons and original line ranges; it contains
# no split assignments. Its annotation checksum prevents using it with a different
# corpus. Grouping is heuristic: without original procurement document IDs, it
# cannot guarantee complete document-level independence.
# 
# The code chooses a seeded group assignment close to the requested record ratios
# while preserving label coverage where possible. A refinement moves whole groups
# to improve the counts. The targets use integer rounding because 1,165 cannot be
# divided into exact percentages. EVENT's only independent group stays in training.
# PER has no examples. All original repeated records are retained. The old 80/10/10
# split is not used here.
# 
# **Training** updates model parameters. **Evaluation** chooses the checkpoint and
# settings. **Test** is held aside until those choices are fixed.

# %%
group_review = json.loads((ANNOTATION_DIR / 'grouping_review.json').read_text(encoding='utf-8'))
assert group_review['annotation_sha256'] == APPROVED_SHA256
groups = group_review['groups']
assert sorted(i for group in groups for i in group) == list(range(len(records)))
group_counts = [Counter(e[2] for i in group for e in records[i][1]['entities']) for group in groups]
label_groups = {label: [g for g, counts in enumerate(group_counts) if counts[label]] for label in LABELS}
forced_train = {g for ids in label_groups.values() if len(ids) == 1 for g in ids}
coverage_labels = [label for label, ids in label_groups.items() if len(ids) >= 3]
names = list(SPLIT_RATIOS)
ratios = list(SPLIT_RATIOS.values())
targets = [int(len(records) * ratios[0]), int(len(records) * ratios[1])]
targets.append(len(records) - sum(targets))
total_counts = sum(group_counts, Counter())
rng = random.Random(SEED)

def assignment_stats(assignment):
    sizes = [sum(len(groups[g]) for g, split in enumerate(assignment) if split == s) for s in range(3)]
    counts = [sum((group_counts[g] for g, split in enumerate(assignment) if split == s), Counter())
              for s in range(3)]
    return sizes, counts

def split_loss(sizes, counts):
    # Record-count balance comes first; supported entity counts break ties.
    size_error = sum(abs(size - target) for size, target in zip(sizes, targets))
    label_error = sum((counts[s][label] / total_counts[label] - ratios[s]) ** 2
                      for label in coverage_labels for s in range(3))
    return size_error, label_error

best = None
for _ in range(5000):
    candidate = [0 if g in forced_train else rng.choices(range(3), weights=ratios)[0]
                 for g in range(len(groups))]
    if any(not any(candidate[g] == 0 for g in ids) for ids in label_groups.values() if ids):
        continue
    if any({candidate[g] for g in label_groups[label]} != {0, 1, 2} for label in coverage_labels):
        continue
    sizes, counts = assignment_stats(candidate)
    score = split_loss(sizes, counts)
    if best is None or score < best[0]:
        best = score, candidate
if best is None:
    raise ValueError('Could not satisfy group/label constraints. Review the corpus coverage.')
assignment = best[1]
sizes, counts = assignment_stats(assignment)

def move_group(g, old, new):
    sizes[old] -= len(groups[g]); sizes[new] += len(groups[g])
    counts[old].subtract(group_counts[g]); counts[new].update(group_counts[g])

for _ in range(20):
    improved = False
    for g in range(len(groups)):
        if g in forced_train:
            continue
        old = assignment[g]
        if any(counts[old][label] == group_counts[g][label] and group_counts[g][label]
               for label in LABELS if label in coverage_labels or old == 0):
            continue
        selected, score = old, split_loss(sizes, counts)
        for new in range(3):
            if new == old:
                continue
            move_group(g, old, new)
            proposed = split_loss(sizes, counts)
            move_group(g, new, old)
            if proposed < score:
                selected, score = new, proposed
        if selected != old:
            move_group(g, old, selected)
            assignment[g] = selected
            improved = True
    if not improved:
        break

# Same-size group swaps improve label balance without changing record counts.
for _ in range(5):
    improved = False
    for g in range(len(groups)):
        if g in forced_train:
            continue
        for h in range(g):
            if h in forced_train or len(groups[g]) != len(groups[h]) or assignment[g] == assignment[h]:
                continue
            old_g, old_h = assignment[g], assignment[h]
            before = split_loss(sizes, counts)
            move_group(g, old_g, old_h)
            move_group(h, old_h, old_g)
            valid = (all(counts[s][label] > 0 for label in coverage_labels for s in range(3))
                     and all(counts[0][label] > 0 for label in LABELS if total_counts[label]))
            if valid and split_loss(sizes, counts) < before:
                assignment[g], assignment[h] = old_h, old_g
                improved = True
            else:
                move_group(h, old_g, old_h)
                move_group(g, old_h, old_g)
    if not improved:
        break

split_indices = {name: sorted(i for g, group in enumerate(groups) if assignment[g] == s for i in group)
                 for s, name in enumerate(names)}
assert sorted(i for ids in split_indices.values() for i in ids) == list(range(len(records)))
index_split = {i: name for name, ids in split_indices.items() for i in ids}
assert all(len({index_split[i] for i in group}) == 1 for group in groups)
assert all(index_split[a] == index_split[b] for link in group_review['links']
           for a, b in [link['annotation_indices']])
normalize = lambda text: re.sub(r'\s+', ' ', text.casefold()).strip()
normalized_splits = defaultdict(set)
for i, (text, _) in enumerate(records):
    normalized_splits[normalize(text)].add(index_split[i])
assert all(len(values) == 1 for values in normalized_splits.values())
split_rows = {name: [records[i] for i in ids] for name, ids in split_indices.items()}
split_manifest = {'annotation_sha256': APPROVED_SHA256, 'grouping_sha256': sha256(ANNOTATION_DIR / 'grouping_review.json'),
                  'seed': SEED, 'requested_ratios': SPLIT_RATIOS, 'integer_record_targets': dict(zip(names, targets)),
                  'original_annotation_indices': split_indices,
                  'groups': [{'group_id': g, 'split': names[assignment[g]], 'annotation_indices': ids}
                             for g, ids in enumerate(groups)],
                  'independent_groups_per_label': {label: len(ids) for label, ids in label_groups.items()},
                  'no_detected_group_leakage': True}
save_json(PROCESSED_DIR / 'split_manifest.json', split_manifest)
SPLIT_SHA256 = sha256(PROCESSED_DIR / 'split_manifest.json')
coverage_rows = []
for label in LABELS:
    coverage_rows.append({'label': label, **{name: sum(e[2] == label for _, a in rows for e in a['entities'])
                                            for name, rows in split_rows.items()},
                          'independent_groups': len(label_groups[label])})
display(pd.DataFrame([{'split': name, 'records': len(ids), 'percent': round(100 * len(ids) / len(records), 2)}
                      for name, ids in split_indices.items()]))
display(pd.DataFrame(coverage_rows))
save_json(REPORT_DIR / 'split_coverage.json', {'rows': coverage_rows})


# %%
# Export readable JSON and strict spaCy DocBin files, retaining original source locations.
for name, ids in split_indices.items():
    selected = split_rows[name]
    save_json(PROCESSED_DIR / f'{name}.json',
              {'classes': LABELS, 'annotations': selected,
               'metadata': {'split': name, 'seed': SEED, 'guideline': '2.1',
                            'annotation_sha256': APPROVED_SHA256, 'split_manifest_sha256': SPLIT_SHA256,
                            'original_annotation_indices': ids, 'record_sources': [sources[i] for i in ids]}})
    docbin = DocBin()
    for text, annotation in selected:
        doc = tokenizer_nlp.make_doc(text)
        doc.ents = [doc.char_span(start, end, label=label, alignment_mode='strict')
                    for start, end, label in annotation['entities']]
        docbin.add(doc)
    path = PROCESSED_DIR / f'{name}.spacy'
    docbin.to_disk(path)
    restored = list(DocBin().from_disk(path).get_docs(tokenizer_nlp.vocab))
    assert len(restored) == len(selected)
    for doc, (text, annotation) in zip(restored, selected):
        assert doc.text == text
        assert [[e.start_char, e.end_char, e.label_] for e in doc.ents] == annotation['entities']
print('Split JSON and spaCy files saved. All texts and spans survived conversion unchanged.')


# %% [markdown]
# ## 5. Train a statistical NER baseline
# 
# This is a **spaCy transition-based NER model initialized from scratch** with its
# default neural feature encoder. We start with `spacy.blank('es')`, add `ner`, and
# learn entity boundaries/labels from the training records. There is no pretrained
# Spanish NER pipeline, transformer or rule-based correction in this baseline.
# Negative examples teach the model to leave ordinary text untagged.
# 
# Each epoch shuffles training records and updates the model in mini-batches.
# Dropout is 0.2; the maximum is 30 epochs. We score the **evaluation** partition
# after each epoch and save the checkpoint with its highest exact-span micro F1.
# Training stops after 5 epochs without improvement. Test data are not used for
# these decisions. PER is omitted from the trained label inventory because it has
# no training examples; EVENT is trained but lacks held-out support.
# 
# The custom tokenizer is used throughout. When reloading this checkpoint, apply
# `make_span_tokenizer` as below. Inspect the split first, then enable
# `RUN_TRAINING` and run the training cell. Training duration depends on the runtime.

# %%
def score_rows(model, rows, include_predictions=False):
    """Strict entity scoring: start, end and label must all match exactly."""
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    predictions = []
    for doc, (text, annotation) in zip(model.pipe((row[0] for row in rows), batch_size=32), rows):
        gold = {tuple(entity) for entity in annotation['entities']}
        predicted = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        tp.update(entity[2] for entity in gold & predicted)
        fp.update(entity[2] for entity in predicted - gold)
        fn.update(entity[2] for entity in gold - predicted)
        support.update(entity[2] for entity in gold)
        if include_predictions:
            predictions.append({'text': text, 'gold': sorted(gold), 'predicted': sorted(predicted),
                                'false_positives': sorted(predicted - gold), 'false_negatives': sorted(gold - predicted)})
    def metrics(t, f, n, gold):
        return {'precision': t / (t + f) if t + f else None,
                'recall': t / (t + n) if t + n else None,
                'f1': 2 * t / (2 * t + f + n) if 2 * t + f + n else None,
                'tp': t, 'fp': f, 'fn': n, 'gold_support': gold}
    micro = metrics(sum(tp.values()), sum(fp.values()), sum(fn.values()), sum(support.values()))
    per_label = {label: metrics(tp[label], fp[label], fn[label], support[label]) for label in LABELS}
    result = {'records': len(rows), 'micro': micro, 'per_label': per_label}
    if include_predictions:
        result['predictions'] = predictions
    return result

def train_baseline(train_rows, evaluation_rows, run_dir, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    fix_random_seed(SEED)
    rng = random.Random(SEED)
    model = make_span_tokenizer(spacy.blank('es'))
    ner = model.add_pipe('ner')
    trained_labels = sorted({label for _, a in train_rows for _, _, label in a['entities']})
    for label in trained_labels:
        ner.add_label(label)
    examples = [Example.from_dict(model.make_doc(text), annotation) for text, annotation in train_rows]
    optimizer = model.initialize(lambda: examples)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale_epochs = -1, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        ordered = list(train_rows)
        rng.shuffle(ordered)
        losses = {}
        for batch in minibatch(ordered, size=BATCH_SIZE):
            batch_examples = [Example.from_dict(model.make_doc(text), annotation) for text, annotation in batch]
            model.update(batch_examples, sgd=optimizer, drop=DROPOUT, losses=losses)
        evaluation = score_rows(model, evaluation_rows)
        f1 = evaluation['micro']['f1'] or 0.0
        history.append({'epoch': epoch, 'ner_loss': float(losses.get('ner', 0)),
                        'evaluation_precision': evaluation['micro']['precision'],
                        'evaluation_recall': evaluation['micro']['recall'], 'evaluation_f1': f1})
        print(f'Epoch {epoch:02d} | NER loss {losses.get("ner", 0):.2f} | evaluation F1 {f1:.3f}')
        if f1 > best_f1:
            best_f1, best_epoch, stale_epochs = f1, epoch, 0
            model.to_disk(run_dir / 'model-best')
            save_json(run_dir / 'best_evaluation.json', evaluation)
        else:
            stale_epochs += 1
        save_json(run_dir / 'training_history.json', history)
        if stale_epochs >= patience:
            print('Early stopping: evaluation F1 has not improved.')
            break
    metadata = {'created_at_utc': utc_now(), 'seed': SEED, 'annotation_sha256': APPROVED_SHA256,
                'split_manifest_sha256': SPLIT_SHA256, 'trained_labels': trained_labels,
                'max_epochs': max_epochs, 'patience': patience, 'dropout': DROPOUT,
                'batch_size': BATCH_SIZE, 'best_epoch': best_epoch, 'best_evaluation_f1': best_f1,
                'checkpoint_selection': 'exact-span micro F1 on evaluation only',
                'test_used_for_checkpoint_selection': False, 'environment': environment}
    save_json(run_dir / 'run_metadata.json', metadata)
    reloaded = make_span_tokenizer(spacy.load(run_dir / 'model-best'))
    assert reloaded.tokenizer.to_bytes() == model.tokenizer.to_bytes()
    return reloaded, metadata


# %%
trained_model = None
run_metadata = None
RUN_DIR = None
if RUN_TRAINING:
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    RUN_DIR = PROJECT_ROOT / 'models' / run_id
    trained_model, run_metadata = train_baseline(split_rows['train'], split_rows['evaluation'], RUN_DIR)
    display(pd.DataFrame(json.loads((RUN_DIR / 'training_history.json').read_text())))
    print('Best model:', RUN_DIR / 'model-best')
else:
    print('Data preparation is complete. Enable RUN_TRAINING in the settings cell when ready, then rerun this cell.')


# %% [markdown]
# ## 6. Test the selected system
# 
# Test scoring is a separate deliberate stage. Enable `RUN_FINAL_TEST` when the
# configuration is fixed and the model has been selected with evaluation data.
# Precision = correct predictions / all predictions; recall = correct predictions /
# gold entities. F1 combines them. A correct entity needs the exact boundaries and
# label. `None` means an undefined metric or absent support, not a perfect score.
# 
# The primary report scores every retained test record. A second report removes
# exact normalized repeated texts within test so repeated boilerplate has less
# weight; it is a sensitivity check, not a new split. Both reports use the same
# selected checkpoint. Per-label support and false-positive/false-negative examples
# are shown. Do not select another model using these test results; that would make
# this partition part of development.

# %%
test_report = None
deduplicated_test_report = None
if RUN_FINAL_TEST:
    if trained_model is None or RUN_DIR is None:
        raise RuntimeError('Train the model in step 5 before running the final test.')
    assert run_metadata['split_manifest_sha256'] == SPLIT_SHA256
    assert run_metadata['annotation_sha256'] == APPROVED_SHA256
    test_report = score_rows(trained_model, split_rows['test'], include_predictions=True)
    for prediction, index in zip(test_report['predictions'], split_indices['test']):
        prediction['original_annotation_index'] = index
        prediction['source'] = sources[index]
    seen, unique_rows = set(), []
    for row in split_rows['test']:
        key = normalize(row[0])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    deduplicated_test_report = score_rows(trained_model, unique_rows)
    save_json(RUN_DIR / 'test_report.json', test_report)
    save_json(RUN_DIR / 'test_normalized_unique_report.json', deduplicated_test_report)
    print('All test records:', test_report['micro'])
    print('Unique normalized test texts:', deduplicated_test_report['micro'])
    display(pd.DataFrame.from_dict(test_report['per_label'], orient='index').rename_axis('label'))
    errors = [row for row in test_report['predictions'] if row['false_positives'] or row['false_negatives']]
    for row in errors[:5]:
        print('Original index:', row['original_annotation_index'], '| source:', row['source'])
        print(row['text'][:500])
        print('False positives:', [(a, b, label, row['text'][a:b]) for a, b, label in row['false_positives']])
        print('False negatives:', [(a, b, label, row['text'][a:b]) for a, b, label in row['false_negatives']])
else:
    print('Final test is held aside. Enable RUN_FINAL_TEST only after model choices are fixed.')


# %%
# Apply the statistical model to new Spanish text: no gold annotations are supplied here.
NEW_TEXT = 'La empresa presentará el Anexo II del PCAP antes del 20 de enero de 2027.'
if trained_model is not None:
    prediction = trained_model(NEW_TEXT)
    display(pd.DataFrame([{'text': e.text, 'label': e.label_, 'start': e.start_char, 'end': e.end_char}
                          for e in prediction.ents], columns=['text', 'label', 'start', 'end']))
else:
    print('Train the model first. Then change NEW_TEXT to annotate another Spanish passage.')


# %% [markdown]
# ## 7. Findings and analysis
# 
# **Known data findings:** the revised corpus contains 1,165 records and 605 spans,
# with MISC dominating the entity inventory. Many records repeat procurement
# boilerplate. PER has no examples, and EVENT has one independent example group.
# EMAIL, SCHEDULE, LOC, DATE and LAW have small evaluation/test samples. Record
# counts therefore overstate the amount of independent evidence.
# 
# **Model findings are generated below only after training/testing.** Discuss the
# primary micro F1 alongside each label's support. Compare precision and recall:
# high precision with low recall means the model misses entities; high recall with
# low precision means it overtags. Inspect whether errors concern span boundaries,
# LAW/MISC distinctions, facility names or generic terms. Compare the full test
# score with the unique-text sensitivity score and explain any difference.
# 
# Conclusions should stay within this small corpus: a single split and AI-assisted
# reference labels do not establish broad performance on Spanish legal texts.
# Collect independent examples for rare labels and improve annotation consistency
# before adding model complexity. Improvements informed by test errors need a new
# unseen test set. Add rules only as a separately evaluated later experiment.

# %%
summary_lines = [
    '# ANER findings',
    f"Data: {len(records)} records, {sum(entity_counts.values())} annotated spans, guidelines 2.1.",
    'Split: ' + ', '.join(f'{name}={len(ids)}' for name, ids in split_indices.items()) + '.',
    'Known limitations: no PER examples; EVENT occurs in one training group; other rare labels have very small support.',
]
if run_metadata is not None:
    summary_lines.append(f"Selected epoch: {run_metadata['best_epoch']}; evaluation micro F1: {run_metadata['best_evaluation_f1']:.3f}.")
if test_report is not None:
    m = test_report['micro']
    metric_text = lambda value: f'{value:.3f}' if value is not None else 'undefined'
    summary_lines.append(f"Final test micro F1: {metric_text(m['f1'])}; precision: {metric_text(m['precision'])}; recall: {metric_text(m['recall'])}.")
    unique_f1 = deduplicated_test_report['micro']['f1']
    summary_lines.append(f"Normalized unique-text test F1: {metric_text(unique_f1)} on {deduplicated_test_report['records']} records.")
    summary_lines.append('Interpret label scores together with gold_support in test_report.json; absent gold support does not validate recognition of that label.')
else:
    summary_lines.append('No final test has been run. No test performance claim is available yet.')
summary_text = '\n\n'.join(summary_lines) + '\n'
summary_path = (RUN_DIR / 'findings.md') if RUN_DIR else REPORT_DIR / 'findings.md'
summary_path.write_text(summary_text, encoding='utf-8')
display(Markdown(summary_text))


# %% [markdown]
# ### Save the work for GitHub
# 
# Keep the notebook, `data/raw/`, `data/annotations/`, `docs/` and `requirements.txt`
# in the project repository. After running preparation, also commit the small JSON
# split/coverage reports so the experiment is traceable. Keep downloaded archives,
# large weights and temporary runtime files out of Git; `.gitignore` covers them.
# The export cell below bundles the generated data/reports and, when available,
# the trained model for download. It does not publish anything to GitHub.
# 
# In Colab, use **File → Save a copy in GitHub** to save your notebook execution.
# The project data files must also exist in that repository; saving the notebook
# alone does not upload them. Review the changed files before committing.
# 
# API references: [spaCy training](https://spacy.io/usage/training),
# [Example](https://spacy.io/api/example), [DocBin](https://spacy.io/api/docbin),
# [EntityRecognizer](https://spacy.io/api/entityrecognizer).

# %%
def export_run():
    output = PROJECT_ROOT.parent / 'ANER_Run_Outputs.zip'
    folders = [PROCESSED_DIR, REPORT_DIR] + ([RUN_DIR] if RUN_DIR else [])
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(PROJECT_ROOT))
    print('Saved:', output)
    try:
        from google.colab import files
    except ImportError:
        return output
    files.download(str(output))
    return output

# Run this when ready to download the generated split files, reports and model:
# export_run()


# %%
export_run()

# %%
"""Paste this cell after model training in ANER_Colab.ipynb.

Inspect exact-span false negatives and false positives without changing labels
or the trained model. Use evaluation errors to guide improvements.
"""
import json
from pathlib import Path
import pandas as pd
from IPython.display import display

REVIEW_SPLIT = 'evaluation'  # 'test' reads the already-generated test_report.
FOCUS_LABELS = ['ORG', 'DATE', 'LOC', 'LAW', 'SCHEDULE']
MAX_EXAMPLES_PER_ERROR_TYPE = 3
CONTEXT_CHARACTERS = 100

if REVIEW_SPLIT == 'evaluation':
    if trained_model is None:
        raise RuntimeError('Run the training cell first.')
    review_report = score_rows(trained_model, split_rows['evaluation'], include_predictions=True)
elif REVIEW_SPLIT == 'test':
    if globals().get('test_report') is None:
        raise RuntimeError('No existing test_report. Run final testing only when model choices are fixed.')
    review_report = test_report
else:
    raise ValueError("REVIEW_SPLIT must be 'evaluation' or 'test'.")

predictions = review_report['predictions']
original_indices = split_indices[REVIEW_SPLIT]
assert len(predictions) == len(original_indices)

def make_error_row(entity, other_entities, error_type, text, position, original_index):
    start, end, label = entity
    overlaps = sorted(e for e in other_entities if start < e[1] and e[0] < end)
    if any(e[0] == start and e[1] == end for e in overlaps):
        diagnosis = 'same boundaries, wrong label'
    elif any(e[2] == label for e in overlaps):
        diagnosis = 'overlapping span, same label: inspect boundaries'
    elif overlaps:
        diagnosis = 'overlapping span with another label'
    else:
        diagnosis = 'no prediction at this span' if error_type == 'false_negative' else 'no gold entity at this span'
    source = sources[original_index]
    return {
        'label': label, 'error_type': error_type, 'diagnosis': diagnosis,
        'entity_text': text[start:end], 'start': start, 'end': end,
        'counterparts': json.dumps([{'text': text[a:b], 'label': tag, 'start': a, 'end': b}
                                   for a, b, tag in overlaps], ensure_ascii=False),
        'context': text[max(0, start - CONTEXT_CHARACTERS):start] + '⟦' + text[start:end] + '⟧'
                   + text[end:end + CONTEXT_CHARACTERS],
        'original_annotation_index': original_index, 'source_file': source['source_file'],
        'line_number': source['line_number'], 'record_position': position,
    }

error_rows = []
for position, (prediction, original_index) in enumerate(zip(predictions, original_indices)):
    text = prediction['text']
    assert text == records[original_index][0], 'Report and split refer to different records.'
    gold = {tuple(entity) for entity in prediction['gold']}
    predicted = {tuple(entity) for entity in prediction['predicted']}
    # Counterparts use ALL labels, including MISC, to reveal label confusions.
    for error_type, entities, counterparts in [
        ('false_negative', gold - predicted, predicted),
        ('false_positive', predicted - gold, gold),
    ]:
        for entity in sorted(entities):
            if entity[2] in FOCUS_LABELS:
                error_rows.append(make_error_row(entity, counterparts, error_type, text, position, original_index))

columns = ['label', 'error_type', 'diagnosis', 'entity_text', 'start', 'end', 'counterparts',
           'context', 'original_annotation_index', 'source_file', 'line_number', 'record_position']
error_df = pd.DataFrame(error_rows, columns=columns).sort_values(
    ['label', 'error_type', 'original_annotation_index', 'start']).reset_index(drop=True)
error_df.index.name = 'error_row'
counts = pd.crosstab(error_df['label'], error_df['error_type']).reindex(
    index=FOCUS_LABELS, columns=['false_negative', 'false_positive'], fill_value=0).fillna(0).astype(int)
print(f'{REVIEW_SPLIT.upper()} ERRORS — false negative: missed gold span; false positive: incorrect predicted span')
display(counts)

for label in FOCUS_LABELS:
    for error_type in ['false_negative', 'false_positive']:
        examples = error_df[(error_df['label'] == label) & (error_df['error_type'] == error_type)]
        # Collapse repeated examples for DISPLAY only. Exports/counts retain all errors.
        examples = examples.drop_duplicates(subset=['entity_text', 'context', 'counterparts'])
        if not examples.empty:
            print(f'\n{label} | {error_type} | row numbers can be used with show_error()')
            with pd.option_context('display.max_colwidth', 180):
                display(examples[['entity_text', 'diagnosis', 'counterparts', 'context',
                                  'original_annotation_index', 'source_file', 'line_number']]
                        .head(MAX_EXAMPLES_PER_ERROR_TYPE))

def show_error(error_row=0):
    """Print the full passage and every gold/predicted entity for one displayed row."""
    row = error_df.loc[error_row]
    prediction = predictions[int(row['record_position'])]
    text = prediction['text']
    print(f"{row['source_file']} | original line {row['line_number']} | annotation {row['original_annotation_index']}")
    print(text)
    for title, key in [('GOLD', 'gold'), ('PREDICTED', 'predicted')]:
        print(title)
        display(pd.DataFrame([{'text': text[a:b], 'label': label, 'start': a, 'end': b}
                              for a, b, label in prediction[key]], columns=['text', 'label', 'start', 'end']))

output_dir = Path(RUN_DIR) if globals().get('RUN_DIR') is not None else Path(REPORT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / f'{REVIEW_SPLIT}_entity_errors.csv'
json_path = output_dir / f'{REVIEW_SPLIT}_error_review.json'
error_df.to_csv(csv_path, index=True, encoding='utf-8-sig')
json_path.write_text(json.dumps({'split': REVIEW_SPLIT, 'focus_labels': FOCUS_LABELS,
                                'original_annotation_indices': original_indices,
                                'record_sources': [sources[i] for i in original_indices],
                                'metrics': {label: review_report['per_label'][label] for label in FOCUS_LABELS},
                                'predictions': predictions}, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nSaved:', csv_path, '\nand:', json_path)
# To inspect a complete example, use a row number displayed above:
# show_error(0)


# %% [markdown]
# # **FINDINGS**
# 
# The first experiment trained a spaCy NER model from scratch on **1,165 Spanish public procurement records containing 605 entity annotations**, revised according to guidelines version 2.1. The corpus was divided into 50% training, 25% evaluation and 25% test data, keeping identified duplicate families together. In the reported baseline run, epoch 22 achieved the highest evaluation F1: **55.4%**.
# 
# On the test partition, the model achieved **67.9% precision, 48.7% recall and 56.7% F1**. F1 fell to **48.5%** when normalized duplicate test texts were removed. Performance was concentrated in MISC, which accounted for 73 of the 74 correctly recognized test entities. ORG training examples lacked diversity: 52 of its 59 spans referred to Seguridad Social or ROLECE. PER had no examples, while EVENT had no held-out support.
# 
# The subsequent evaluation-error review identified missed entities, incorrect labels and overextended boundaries, particularly in dates, legal citations and schedules. These findings establish a baseline with limited recognition across categories. The next experiment will combine the statistical model with carefully bounded patterns and dictionaries, measuring their effect separately against this baseline.

