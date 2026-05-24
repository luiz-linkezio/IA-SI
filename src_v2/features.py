import re
from collections import Counter

import numpy as np


ATTACK_PATTERNS_V2 = {
    "ignore": [
        r"\bignore\s+(previous|past|prior|all|everything|above)\b",
        r"\bforget\s+(all|previous|past|prior|everything|about)\b",
        r"\bdisregard\s+(all|any|previous|prior|above|instructions?|rules?)\b",
    ],
    "act_as": [
        r"\bact\s+as\b",
        r"\bpretend\s+(to\s+)?be\b",
        r"\byou\s+are\s+now\s+(a|an|the|my)\b",
    ],
    "system": [
        r"\bsystem\s*:\s*[{(\"\[]",
        r"\badmin\s*:\s*[{(\"\[]",
        r"\broot\s*:\s*[{(\"\[]",
    ],
    "override": [
        r"\bbypass\b",
        r"\boverride\b",
        r"\bignore\s+(restrictions|rules)\b",
    ],
    "execute": [
        r"\b(execute|run|perform|do)\s+(this|the\s+following)\b",
        r"\binstead\s*,?\s*(do|execute|output|respond|answer|follow)\b",
        r"\bfrom\s+now\s+on\b",
    ],
    "instruction_verb": [
        r"\b(tell|make|force|order|command|demand|instruct)\s+(me|you|them|us|the\s+AI|the\s+model|the\s+assistant)\s+to\b",
    ],
    "role_switch": [
        r"\b(you\s+are\s+no\s+longer|stop\s+being|cease\s+to\s+be)\b",
    ],
}

COMPILED_PATTERNS_V2 = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in ATTACK_PATTERNS_V2.items()
}

FEATURE_NAMES_V2 = [
    "text_length",
    "word_count",
    "unique_words",
    "sentence_count",
    "avg_word_length",
    "char_count_no_space",
    "uppercase_ratio",
    "lowercase_ratio",
    "special_char_ratio",
    "punctuation_ratio",
    "space_ratio",
    "newline_count",
    "type_token_ratio",
    "lexical_diversity",
    "avg_word_frequency",
    "word_length_variance",
    "entropy",
    "colon_count",
    "keyword_diversity",
]

MEDIAN_LEXICAL_DIVERSITY = 1.041393


def extract_morfologicas(text):
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    return {
        "text_length": len(text),
        "word_count": len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?")),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0,
        "char_count_no_space": len(text.replace(" ", "")),
    }


def extract_caracteres(text):
    if not isinstance(text, str):
        text = str(text)
    length = len(text)
    if length == 0:
        return {
            f: 0
            for f in [
                "uppercase_ratio",
                "lowercase_ratio",
                "special_char_ratio",
                "punctuation_ratio",
                "space_ratio",
                "newline_count",
            ]
        }
    uppercase = sum(1 for c in text if c.isupper())
    lowercase = sum(1 for c in text if c.islower())
    spaces = text.count(" ")
    newlines = text.count("\n")
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    punctuation = sum(1 for c in text if c in ".,!?;:'\"")
    return {
        "uppercase_ratio": uppercase / length,
        "lowercase_ratio": lowercase / length,
        "special_char_ratio": special_chars / length,
        "punctuation_ratio": punctuation / length,
        "space_ratio": spaces / length,
        "newline_count": newlines,
    }


def _entropy_shannon(text):
    if not text:
        return 0
    counter = Counter(text)
    probs = [count / len(text) for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def extract_linguisticas(text):
    if not isinstance(text, str):
        text = str(text)
    words = text.lower().split()
    unique_words = len(set(words))
    word_count = len(words)
    if word_count == 0:
        return {
            "type_token_ratio": 0,
            "lexical_diversity": 0,
            "avg_word_frequency": 0,
            "word_length_variance": 0,
            "entropy": 0,
        }
    ttr = unique_words / word_count
    lex_div = np.log(word_count) / np.log(unique_words) if unique_words > 1 else float(word_count)
    word_freqs = Counter(words)
    avg_freq = sum(word_freqs.values()) / len(word_freqs) if word_freqs else 0
    word_lengths = [len(w) for w in words]
    word_len_var = float(np.var(word_lengths)) if word_lengths else 0
    ent = _entropy_shannon(text)
    return {
        "type_token_ratio": ttr,
        "lexical_diversity": lex_div,
        "avg_word_frequency": avg_freq,
        "word_length_variance": word_len_var,
        "entropy": ent,
    }


def extract_estruturais(text):
    if not isinstance(text, str):
        text = str(text)
    colon_count = text.count(":")
    return {
        "colon_count": colon_count,
    }


def extract_keyword_diversity(text):
    if not isinstance(text, str):
        text = str(text)
    categories_matched = 0
    for category, compiled_regexes in COMPILED_PATTERNS_V2.items():
        for regex in compiled_regexes:
            if regex.search(text):
                categories_matched += 1
                break
    return {"keyword_diversity": categories_matched}


class PromptInjectionFeatureEngineerV2:
    def __init__(self):
        self.attack_patterns = ATTACK_PATTERNS_V2
        self.compiled_patterns = COMPILED_PATTERNS_V2

    def extract_all(self, text):
        features = {}
        features.update(extract_morfologicas(text))
        features.update(extract_caracteres(text))
        features.update(extract_linguisticas(text))
        features.update(extract_estruturais(text))
        features.update(extract_keyword_diversity(text))
        ld = features.get("lexical_diversity")
        if ld is None or (isinstance(ld, float) and (np.isnan(ld) or np.isinf(ld))):
            features["lexical_diversity"] = MEDIAN_LEXICAL_DIVERSITY
        return features

    def extract_as_dataframe(self, text):
        import pandas as pd

        features = self.extract_all(text)
        df = pd.DataFrame([features], columns=FEATURE_NAMES_V2)
        return df[FEATURE_NAMES_V2]