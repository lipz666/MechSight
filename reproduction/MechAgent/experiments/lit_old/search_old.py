"""Europe PMC search for the OLD exposure strata (mechanism-origin study, 2026-11-05).

Same query as lit150/search.py (mechanistic studies + catalysis/radical chemistry, same journal list), but
first published 2008-01-01..2019-12-31 (widened from 2016 on 2026-11-05: OA PDFs before 2014 are scarce), in PMC (OA subset or author manuscript; the PDF fetch decides).
Strata by Europe PMC citedByCount (checked again against OpenAlex in select_old.py):
  high  >= 100 citations      low  <= 15 citations
Bio-related work is excluded twice: NOT clauses in the query and a regex on title + abstract + keywords
(any word starting with "bio", enzymes, proteins, peptides, nucleic acids, cells, drugs, medicine).
Writes candidates_old.json.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
JOURNALS = ['Nat Commun', 'Chem Sci', 'JACS Au', 'ACS Cent Sci', 'Commun Chem', 'Beilstein J Org Chem',
            'Angew Chem Int Ed Engl', 'J Am Chem Soc', 'Org Lett', 'Chem Commun (Camb)', 'Sci Adv', 'Precis Chem',
            'Chem', 'Nat Chem', 'Nat Catal', 'Nat Synth', 'J Org Chem', 'Org Chem Front', 'Green Chem', 'Chemistry']
BIO_Q = ('ABSTRACT:bio* OR TITLE:bio* OR ABSTRACT:enzym* OR ABSTRACT:protein* OR ABSTRACT:peptide* '
         'OR ABSTRACT:DNA OR ABSTRACT:RNA OR ABSTRACT:cell* OR ABSTRACT:"in vivo" OR ABSTRACT:drug*')
# The lit150 abstract phrases ("mechanistic studies", ...) are rare before 2017 (196 hits in PMC); the abstract
# condition is widened to mechanism words, and the full-text >=3 mechanistic-experiment categories filter in
# select_old.py (same as lit150) does the real selection.
MECH = ('(ABSTRACT:mechanis* OR ABSTRACT:"kinetic isotope" OR ABSTRACT:"radical clock" OR ABSTRACT:deuterium '
        'OR ABSTRACT:Hammett OR ABSTRACT:"kinetic studies" OR ABSTRACT:intermediate*)')
QUERY = (MECH + ' AND (ABSTRACT:catalys* OR ABSTRACT:photoredox OR ABSTRACT:radical OR ABSTRACT:photocataly*) '
         'AND IN_PMC:y AND FIRST_PDATE:[2008-01-01 TO 2019-12-31] AND ('
         + ' OR '.join(f'JOURNAL:"{j}"' for j in JOURNALS) + f') NOT ({BIO_Q})')
# lit150's title exclusions (materials, heterogeneous, enzymes, ...) plus the bio terms on all text fields
EXCLUDE = re.compile(r'heterogeneous|nanoparticle|single-atom|metal-organic framework|\bMOF\b|electrode|battery|'
                     r'zeolite|polymeri[sz]ation|perovskite|photovoltaic|CO2 reduction|water splitting|'
                     r'oxygen evolution|hydrogen evolution|membrane', re.I)
BIO = re.compile(r'\bbio\w*|enzym\w*|\bprotein\w*|\bpeptid\w*|\bDNA\b|\bRNA\b|\bcells?\b|cellular|in vivo|in vitro|'
                 r'\bdrugs?\b|pharma\w*|medicin\w*|antibacterial|anticancer|antitumou?r|tumou?r|metabol\w*|'
                 r'physiolog\w*|living|natural product|amino acid|nucleo\w*|glyco\w*|carbohydrate|sugar|lipid|'
                 r'antibod\w*|vaccin\w*|clinical|therap\w*|disease|bacteri\w*|fung\w*|virus|viral', re.I)


def get(url, tries=6):
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=180).read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * 2 ** i)


def main():
    out, cursor, seen, dropped = [], '*', 0, {'exclude': 0, 'bio': 0}
    while True:
        q = urllib.parse.urlencode({'query': QUERY, 'format': 'json', 'pageSize': 300, 'resultType': 'core',
                                    'cursorMark': cursor})
        d = get(f'{API}?{q}')
        res = d['resultList']['result']
        for r in res:
            seen += 1
            if not r.get('pmcid'):
                continue
            kw = ' '.join((r.get('keywordList') or {}).get('keyword') or [])
            text = ' '.join([r.get('title') or '', r.get('abstractText') or '', kw])
            if EXCLUDE.search(r.get('title') or ''):
                dropped['exclude'] += 1
                continue
            if BIO.search(text):
                dropped['bio'] += 1
                continue
            out.append({'pmcid': r['pmcid'], 'doi': r.get('doi'), 'date': r.get('firstPublicationDate'),
                        'year': int(r.get('pubYear') or 0), 'journal': (r.get('journalInfo') or {}).get('journal', {}).get('medlineAbbreviation'),
                        'title': r.get('title'), 'cited_by': int(r.get('citedByCount') or 0),
                        'oa': r.get('isOpenAccess'), 'author_manuscript': r.get('authMan'),
                        'first_author': (r.get('authorList') or {}).get('author', [{}])[0].get('lastName'),
                        'last_author': ((r.get('authorList') or {}).get('author') or [{}])[-1].get('lastName')})
        nxt = d.get('nextCursorMark')
        if not nxt or nxt == cursor or not res:
            break
        cursor = nxt
    from collections import Counter
    band = Counter('high' if r['cited_by'] >= 100 else 'low' if r['cited_by'] <= 15 else 'mid' for r in out)
    print(json.dumps({'hits': d.get('hitCount'), 'seen': seen, 'kept': len(out), 'dropped': dropped, 'bands': band}))
    (HERE / 'candidates_old.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
