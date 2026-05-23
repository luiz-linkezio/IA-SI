import re
from collections import Counter

import numpy as np


ATTACK_PATTERNS = {
    "ignore": [
        r"\bignore\s+(previous|past|prior|all)",
        r"\bforget\b",
        r"\bdisregard\b",
    ],
    "act_as": [
        r"\bact\s+as\b",
        r"\bpretend\s+(to\s+)?be\b",
        r"\byou\s+are\s+now\b",
        r"\bbecome\b",
    ],
    "system": [
        r"\bsystem\s*:\s*",
        r"\badmin\s*:\s*",
        r"\broot\s*:\s*",
    ],
    "override": [
        r"\bbypass\b",
        r"\boverride\b",
        r"\bignore\s+restrictions\b",
        r"\bignore\s+rules\b",
    ],
    "execute": [
        r"\b(execute|run|perform|do)\s+(this|the\s+following)\b",
        r"\binstead\s*,?\s*",
        r"\bfrom\s+now\s+on\b",
    ],
}

COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in ATTACK_PATTERNS.items()
}

FEATURE_NAMES = [
    "text_length",
    "word_count",
    "unique_words",
    "sentence_count",
    "avg_word_length",
    "char_count_no_space",
    "uppercase_ratio",
    "lowercase_ratio",
    "digit_ratio",
    "special_char_ratio",
    "punctuation_ratio",
    "space_ratio",
    "newline_count",
    "type_token_ratio",
    "lexical_diversity",
    "avg_word_frequency",
    "word_length_variance",
    "entropy",
    "has_ignore_keyword",
    "count_ignore_keyword",
    "has_act_as_keyword",
    "count_act_as_keyword",
    "has_system_keyword",
    "count_system_keyword",
    "has_override_keyword",
    "count_override_keyword",
    "has_execute_keyword",
    "count_execute_keyword",
    "total_injection_keywords",
    "keyword_density",
    "colon_count",
    "bracket_count",
    "parenthesis_count",
    "quote_count",
    "comma_to_period_ratio",
]

MEDIAN_LEXICAL_DIVERSITY = 1.041393
CLIP_COMMA_TO_PERIOD_RATIO = 3.5


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
                "digit_ratio",
                "special_char_ratio",
                "punctuation_ratio",
                "space_ratio",
                "newline_count",
            ]
        }
    uppercase = sum(1 for c in text if c.isupper())
    lowercase = sum(1 for c in text if c.islower())
    digits = sum(1 for c in text if c.isdigit())
    spaces = text.count(" ")
    newlines = text.count("\n")
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    punctuation = sum(1 for c in text if c in ".,!?;:'\"")
    return {
        "uppercase_ratio": uppercase / length,
        "lowercase_ratio": lowercase / length,
        "digit_ratio": digits / length,
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
    lex_div = np.log(word_count) / np.log(unique_words) if unique_words > 0 else 0
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


def extract_padroes_ataque(text):
    if not isinstance(text, str):
        text = str(text)
    features = {}
    total_keywords = 0
    word_count = max(1, len(text.split()))
    for category, compiled_regexes in COMPILED_PATTERNS.items():
        has_pattern = False
        count = 0
        for regex in compiled_regexes:
            matches = regex.findall(text)
            if matches:
                has_pattern = True
                count += len(matches)
        features[f"has_{category}_keyword"] = int(has_pattern)
        features[f"count_{category}_keyword"] = count
        total_keywords += count
    features["total_injection_keywords"] = total_keywords
    features["keyword_density"] = total_keywords / word_count
    return features


def extract_estruturais(text):
    if not isinstance(text, str):
        text = str(text)
    colon_count = text.count(":")
    bracket_count = text.count("[") + text.count("]")
    paren_count = text.count("(") + text.count(")")
    quote_count = text.count('"') + text.count("'")
    commas = text.count(",")
    periods = text.count(".")
    comma_to_period = commas / max(1, periods)
    return {
        "colon_count": colon_count,
        "bracket_count": bracket_count,
        "parenthesis_count": paren_count,
        "quote_count": quote_count,
        "comma_to_period_ratio": comma_to_period,
    }


class PromptInjectionFeatureEngineer:
    def __init__(self):
        self.attack_patterns = ATTACK_PATTERNS
        self.compiled_patterns = COMPILED_PATTERNS

    def extract_all(self, text):
        features = {}
        features.update(extract_morfologicas(text))
        features.update(extract_caracteres(text))
        features.update(extract_linguisticas(text))
        features.update(extract_padroes_ataque(text))
        features.update(extract_estruturais(text))
        if features.get("lexical_diversity") is None or (
            isinstance(features.get("lexical_diversity"), float)
            and np.isnan(features.get("lexical_diversity"))
        ):
            features["lexical_diversity"] = MEDIAN_LEXICAL_DIVERSITY
        features["comma_to_period_ratio"] = min(
            features["comma_to_period_ratio"], CLIP_COMMA_TO_PERIOD_RATIO
        )
        return features

    def extract_as_dataframe(self, text):
        import pandas as pd

        features = self.extract_all(text)
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        return df[FEATURE_NAMES]