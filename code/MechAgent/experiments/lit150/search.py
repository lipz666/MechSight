"""Europe PMC search for the expanded 2026 literature set (same criteria as evidence_p0, wider journal list).

OA, in PMC, first published 2026-01-01..2026-10-31, abstract mentions mechanistic studies/experiments
and catalysis/radical chemistry. Excludes heterogeneous/material/enzyme/battery work by title.
Run with HTTPS_PROXY set to the system proxy. Writes candidates.json.
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
JOURNALS = ['Nat Commun', 'Chem Sci', 'JACS Au', 'ACS Cent Sci', 'Commun Chem', 'Beilstein J Org Chem',
            'Angew Chem Int Ed Engl', 'J Am Chem Soc', 'Org Lett', 'Chem Commun (Camb)', 'Sci Adv', 'Precis Chem',
            'Chem', 'Nat Chem', 'Nat Catal', 'Nat Synth', 'J Org Chem', 'Org Chem Front', 'Green Chem', 'Chemistry']
QUERY = ('(ABSTRACT:"mechanistic studies" OR ABSTRACT:"mechanistic experiments" OR ABSTRACT:"mechanistic investigations" '
         'OR ABSTRACT:"mechanistic investigation" OR ABSTRACT:"mechanistic study" OR ABSTRACT:"control experiments" '
         'OR ABSTRACT:"mechanistic insights") AND (ABSTRACT:catalys* OR ABSTRACT:photoredox OR ABSTRACT:radical '
         'OR ABSTRACT:photocataly*) AND OPEN_ACCESS:y AND IN_PMC:y AND FIRST_PDATE:[2026-01-01 TO 2026-10-31] AND ('
         + ' OR '.join(f'JOURNAL:"{j}"' for j in JOURNALS) + ')')
EXCLUDE = re.compile(r'heterogeneous|nanoparticle|single-atom|metal-organic framework|\bMOF\b|electrode|battery|'
                     r'zeolite|enzyme|biocatal|protein|cell|tumou?r|polymeri[sz]ation|perovskite|photovoltaic|'
                     r'CO2 reduction|water splitting|oxygen evolution|hydrogen evolution|membrane', re.I)


def main():
    out, cursor = [], '*'
    while True:
        q = urllib.parse.urlencode({'query': QUERY, 'format': 'json', 'pageSize': 1000, 'resultType': 'lite',
                                    'cursorMark': cursor})
        d = json.loads(urllib.request.urlopen(f'{API}?{q}', timeout=120).read())
        for r in d['resultList']['result']:
            if r.get('pmcid') and not EXCLUDE.search(r.get('title', '')):
                out.append({'pmcid': r['pmcid'], 'doi': r.get('doi'), 'date': r.get('firstPublicationDate'),
                            'journal': r.get('journalTitle'), 'title': r.get('title')})
        nxt = d.get('nextCursorMark')
        if not nxt or nxt == cursor or not d['resultList']['result']:
            break
        cursor = nxt
    print('hits', d.get('hitCount'), 'kept', len(out))
    (HERE / 'candidates.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
