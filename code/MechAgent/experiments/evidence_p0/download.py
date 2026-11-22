"""Fetch the article PDF and supplementary PDFs for papers.csv from the PMC Open Access S3 bucket.

The bucket (s3://pmc-oa-opendata, public) keeps one folder per article version,
PMC<id>.<version>/. The latest version is used: its PMC<id>.<v>.pdf becomes
papers/<id>/main.pdf and every other .pdf goes to papers/<id>/supp/. Which
supplementary file is the SI (vs. a peer-review file) is decided afterwards by
--assign, from the first page text.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import urllib.request

from extract import HERE, PAPERS

BUCKET = 'https://pmc-oa-opendata.s3.amazonaws.com'


def list_keys(pmcid):
    xml = urllib.request.urlopen(f'{BUCKET}/?list-type=2&prefix={pmcid}.', timeout=60).read().decode()
    return [(k, int(s)) for k, s in re.findall(r'<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>', xml)]


def fetch(pmcid):
    keys = list_keys(pmcid)
    if not keys:
        return pmcid, 'not in bucket', 0
    version = max(int(k.split('/')[0].split('.')[1]) for k, _ in keys)
    pdfs = [(k, s) for k, s in keys if k.startswith(f'{pmcid}.{version}/') and k.endswith('.pdf')]
    out = PAPERS / pmcid
    (out / 'supp').mkdir(parents=True, exist_ok=True)
    total = 0
    for key, size in pdfs:
        name = key.split('/')[1]
        dest = out / 'main.pdf' if name == f'{pmcid}.{version}.pdf' else out / 'supp' / name
        if not (dest.exists() and dest.stat().st_size == size):
            with urllib.request.urlopen(f'{BUCKET}/{key}', timeout=300) as r, dest.open('wb') as f:
                shutil.copyfileobj(r, f)
        total += size
    return pmcid, f'v{version}, {len(pdfs)} pdfs', total


def first_page(path):
    return subprocess.run(['pdftotext', '-l', '2', str(path), '-'], capture_output=True, text=True).stdout


def assign(pmcid):
    """Copy the supplementary PDF that looks like the SI to si.pdf; flag peer-review files."""
    supp = sorted((PAPERS / pmcid / 'supp').glob('*.pdf'))
    kinds = {}
    for p in supp:
        head = first_page(p)
        kinds[p.name] = ('peer_review' if re.search(r'peer review|reviewer', head, re.I) else
                         'reporting' if re.search(r'reporting summary', head, re.I) else
                         'si' if re.search(r'supporting information|supplementary information|electronic supplementary', head, re.I) else
                         'other')
    si = [p for p in supp if kinds[p.name] == 'si'] or [p for p in supp if kinds[p.name] == 'other']
    if si:
        best = max(si, key=lambda p: p.stat().st_size)
        shutil.copyfile(best, PAPERS / pmcid / 'si.pdf')
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assign', action='store_true', help='only pick si.pdf among downloaded supplementary files')
    cli = ap.parse_args()
    ids = [r['paper_id'] for r in csv.DictReader((HERE / 'papers.csv').open())]
    if not cli.assign:
        grand = 0
        for pid in ids:
            pid, msg, size = fetch(pid)
            grand += size
            print(f'{pid}: {msg}, {size / 1e6:.1f} MB', flush=True)
        print(f'total {grand / 1e6:.0f} MB')
    for pid in ids:
        if (PAPERS / pid / 'supp').exists():
            print(pid, assign(pid))


if __name__ == '__main__':
    main()
