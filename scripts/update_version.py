'''
Set the spm1d version in every file that holds it.

    python scripts/update_version.py 1.2.3            # set the version
    python scripts/update_version.py --check 1.2.3    # check that every file holds it

The new version must be three whole numbers separated by dots and higher than
the current version;  otherwise nothing is changed.  The publish workflow runs
the --check form with the pushed tag, so a tag that disagrees with the files
publishes nothing.
'''

import datetime
import re
import sys
from pathlib import Path

root  = Path(__file__).resolve().parent.parent

#  (file, pattern of the line holding the version, that line for version v)
#  The first file holds the current version.
FILES = [
    ('spm1d/__init__.py', r"__version__\s*=\s*'([^'\s]+)", "__version__ = '{v}'  # {date}"),
    ('setup.py',          r"\s+version\s*=\s*'([^']+)'",   "    version          = '{v}',"),
    ('README.md',         r'!\[version\]\(https://img\.shields\.io/badge/version-([^-]+)-blue\)',
                          '![version](https://img.shields.io/badge/version-{v}-blue)'),
]


def current_version():
    path, pattern, _ = FILES[0]
    return find(path, pattern)[1]


def find(path, pattern):
    #  (line index, version) of the one line in path that holds the version
    lines = (root / path).read_text().split('\n')
    hits  = [(i, m.group(1)) for i, s in enumerate(lines) if (m := re.match(pattern, s))]
    if len(hits) != 1:
        sys.exit(f'error: expected one version line in {path}, found {len(hits)}')
    return hits[0]


def parse(v):
    if not re.fullmatch(r'\d+\.\d+\.\d+', v):
        sys.exit(f'error: "{v}" is not a version of the form 1.2.3 (three whole numbers separated by dots)')
    return tuple(int(x) for x in v.split('.'))


def check(v):
    #  every file must hold version v
    bad = [f'{path} has {found}' for path, pattern, _ in FILES if (found := find(path, pattern)[1]) != v]
    if bad:
        sys.exit(f'error: the files do not all hold version {v}:\n  ' + '\n  '.join(bad))
    print(f'all files hold version {v}')


def update(v):
    old = current_version()
    if parse(v) <= parse(old.split()[0]):
        sys.exit(f'error: the new version {v} is not higher than the current version {old}')
    date = datetime.date.today().isoformat()
    for path, pattern, template in FILES:      # find every line before changing anything
        find(path, pattern)
    for path, pattern, template in FILES:
        i, found = find(path, pattern)
        lines    = (root / path).read_text().split('\n')
        lines[i] = template.format(v=v, date=date)
        (root / path).write_text('\n'.join(lines))
        print(f'{path}:  {found}  ->  {v}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        parse(args[0])
        update(args[0])
    elif len(args) == 2 and args[0] == '--check':
        check(args[1])
    else:
        sys.exit(__doc__)
