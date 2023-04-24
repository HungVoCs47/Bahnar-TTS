""" from https://github.com/keithito/tacotron """

import re


valid_symbols = [
    'ä', 'å', 'ë', 'ï', 'ñ', 'ö', 'ø', 'û', 'đ', 'ỉ'
]

_valid_symbol_set = set(valid_symbols)


class BNDict:
    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='utf8') as f:
                entries = _parse_bndict(f)
        else:
            entries = _parse_bndict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        return self._entries.get(word.upper())


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_bndict(file):
    bndict = {}
    for line in file:
        if len(line):
            parts = line.split('\t')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in bndict:
                    bndict[word].append(pronunciation)
                else:
                    bndict[word] = [pronunciation]
    return bndict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    # for part in parts:
    #     if part not in _valid_symbol_set:
    #         print(part, ' is not valid')
    #         return None
    return ' '.join(parts)
