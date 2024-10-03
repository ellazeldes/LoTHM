#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator




class HebrewBackend:
    """
        NEED MUCH MORE WORK!
    """

    def __init__(self):
        self.allowed_chars = "abcdefghijklmnopqrstuvwxyz"
        self.allowed_chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.allowed_chars += "אבגדהוזחטיכלמנסעפצקרשתןךףםץ"
        self.allowed_chars += "1234567890"
        self.allowed_chars += "!,\".':;?_()/=-׳"

        self.disallowed_char = '~'

        self.disallowed_chars_set = set()

    def phonemize(
            self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []

        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)

            phones = list()

            for char in _text:
                phones.append("|")
                if char in self.allowed_chars:
                    phones.append(char)
                else:
                    print(f"disallowed char: {char} added blank")
                    self.disallowed_chars_set.add(char)
                    phones.append(self.disallowed_char)

            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}").replace("|_|", "_")
            )

        # deal with english text inside

        print(f"got {text} - returned: {phonemized}")
        return phonemized  # reverse because hebrew


if __name__ == '__main__':
    text = "היי"
    backend = HebrewBackend()
    phonemized = backend.phonemize(
        text, separator=Separator(word="_", syllable="-", phone="|")
    )
