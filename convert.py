import re
import sys

import ipdb

@ipdb.launch_ipdb_on_exception()
def main():
    clauses = []

    in_comment = False
    next_comment = False
    clause = ''
    for line in open(sys.argv[1]):
        if in_comment:
            if '*/' not in line:
                continue

            line = re.sub(r'\*/', '', line)
            in_comment = False

        line = line.partition('%')[0]
        line = re.sub(r'/\*.*\*/', '', line)
        if '/*' in line:
            line = re.sub(r'/\*.*', '', line)
            next_comment = True

        line = re.sub(r'\s', '', line)
        if not line:
            if next_comment:
                next_comment = False
                in_comment = True
            continue
        
        clause += line
        if line[-1] == '.':
            clauses.append(clause)
            clause = ''

        if next_comment:
            next_comment = False
            in_comment = True

    assert clause == '', f'Non-empty final clause {clause}'
    assert not in_comment, 'Non-finished comment'

    conj = []
    name = ''
    for p in clauses:
        regex = r'cnf\((\w+),(\w+),(?:\((.*)\)|(.*))\)\.'
        name, thing, formula, inner = re.fullmatch(regex, p).groups()
        if formula is None:
            formula = inner

        formula = formula.replace('~', '-')
        disj = formula.split('|')
        conj.append(disj)

    print(f'cnf({name}, theorem, {str(conj).replace("'", '')}).')
    import ipdb
    ipdb.set_trace()

if __name__ == '__main__':
    main()
