""" from https://github.com/keithito/tacotron """

import re


valid_symbols = [
    'ɗ312', '́312', 'oː312', 'ɨə312', '–312', 'ɛ312', 'ɲ312', 'ñ312', 'uə312', 'th312', 'ŋ312', 
    '’312', 'ɔ312', 'ʔ312', '̀312', 'ŏ312', 'ɔː312', '̉312', 'kw312', '̃312', 'iə312', 'ɛː312', 
    'ɣ312', 'ə312', 'ɓ312', 'ɨ312', 'əː312', 'tɕ312', 'aː312', 'eː312', 'ɗ35g', '́35g', 'oː35g', 
    'ɨə35g', '–35g', 'ɛ35g', 'ɲ35g', 'ñ35g', 'uə35g', 'th35g', 'ŋ35g', '’35g', 'ɔ35g', 'ʔ35g', 
    '̀35g', 'ŏ35g', 'ɔː35g', '̉35g', 'kw35g', '̃35g', 'iə35g', 'ɛː35g', 'ɣ35g', 'ə35g', 'ɓ35g', 
    'ɨ35g', 'əː35g', 'tɕ35g', 'aː35g', 'eː35g', 'ɗ21g', '́21g', 'oː21g', 'ɨə21g', '–21g', 'ɛ21g', 
    'ɲ21g', 'ñ21g', 'uə21g', 'th21g', 'ŋ21g', '’21g', 'ɔ21g', 'ʔ21g', '̀21g', 'ŏ21g', 'ɔː21g', 
    '̉21g', 'kw21g', '̃21g', 'iə21g', 'ɛː21g', 'ɣ21g', 'ə21g', 'ɓ21g', 'ɨ21g', 'əː21g', 'tɕ21g', 
    'aː21g', 'eː21g', 'ɗ3g5', '́3g5', 'oː3g5', 'ɨə3g5', '–3g5', 'ɛ3g5', 'ɲ3g5', 'ñ3g5', 'uə3g5', 
    'th3g5', 'ŋ3g5', '’3g5', 'ɔ3g5', 'ʔ3g5', '̀3g5', 'ŏ3g5', 'ɔː3g5', '̉3g5', 'kw3g5', '̃3g5', 'iə3g5', 
    'ɛː3g5', 'ɣ3g5', 'ə3g5', 'ɓ3g5', 'ɨ3g5', 'əː3g5', 'tɕ3g5', 'aː3g5', 'eː3g5', 'ɗ33', '́33', 'oː33', 
    'ɨə33', '–33', 'ɛ33', 'ɲ33', 'ñ33', 'uə33', 'th33', 'ŋ33', '’33', 'ɔ33', 'ʔ33', '̀33', 'ŏ33', 'ɔː33', 
    '̉33', 'kw33', '̃33', 'iə33', 'ɛː33', 'ɣ33', 'ə33', 'ɓ33', 'ɨ33', 'əː33', 'tɕ33', 'aː33', 'eː33', 'ɗ21', 
    '́21', 'oː21', 'ɨə21', '–21', 'ɛ21', 'ɲ21', 'ñ21', 'uə21', 'th21', 'ŋ21', '’21', 'ɔ21', 'ʔ21', '̀21', 'ŏ21', 
    'ɔː21', '̉21', 'kw21', '̃21', 'iə21', 'ɛː21', 'ɣ21', 'ə21', 'ɓ21', 'ɨ21', 'əː21', 'tɕ21', 'aː21', 'eː21', 
    'ɗ45', '́45', 'oː45', 'ɨə45', '–45', 'ɛ45', 'ɲ45', 'ñ45', 'uə45', 'th45', 'ŋ45', '’45', 'ɔ45', 'ʔ45', '̀45', 
    'ŏ45', 'ɔː45', '̉45', 'kw45', '̃45', 'iə45', 'ɛː45', 'ɣ45', 'ə45', 'ɓ45', 'ɨ45', 'əː45', 'tɕ45', 'aː45', 
    'eː45', 'ɗ24', '́24', 'oː24', 'ɨə24', '–24', 'ɛ24', 'ɲ24', 'ñ24', 'uə24', 'th24', 'ŋ24', '’24', 'ɔ24', 'ʔ24', 
    '̀24', 'ŏ24', 'ɔː24', '̉24', 'kw24', '̃24', 'iə24', 'ɛː24', 'ɣ24', 'ə24', 'ɓ24', 'ɨ24', 'əː24', 'tɕ24', 'aː24', 
    'eː24', 'ɗ32', '́32', 'oː32', 'ɨə32', '–32', 'ɛ32', 'ɲ32', 'ñ32', 'uə32', 'th32', 'ŋ32', '’32', 'ɔ32', 'ʔ32', 
    '̀32', 'ŏ32', 'ɔː32', '̉32', 'kw32', '̃32', 'iə32', 'ɛː32', 'ɣ32', 'ə32', 'ɓ32', 'ɨ32', 'əː32', 'tɕ32', 'aː32', 'eː32'
]

_valid_symbol_set = set(valid_symbols)


class VNDict:
    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='utf8') as f:
                entries = _parse_vndict(f)
        else:
            entries = _parse_vndict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        return self._entries.get(word.upper())


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_vndict(file):
    vndict = {}
    for line in file:
        if len(line):
            parts = line.split('\t')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in vndict:
                    vndict[word].append(pronunciation)
                else:
                    vndict[word] = [pronunciation]
    return vndict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    # for part in parts:
    #     if part not in _valid_symbol_set:
    #         print(part, ' is not valid')
    #         return None
    return ' '.join(parts)
