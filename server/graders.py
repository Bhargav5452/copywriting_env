from __future__ import annotations

import re
from typing import Any

import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def _clamp(v: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, v))


def _vader_compound(text: str) -> float:
    return _vader.polarity_scores(text)["compound"]


def _flesch_ease(text: str) -> float:
    return textstat.flesch_reading_ease(text)


def grade_subject_line(text: str, gt: dict[str, Any]) -> dict[str, Any]:
    text = text.strip()

    char_count = len(text)
    length_score = 0.40 if char_count <= gt["max_length"] else 0.0

    text_lower = text.lower()
    hit_words = [w for w in gt["power_words"] if w in text_lower]
    power_score = 0.30 if hit_words else 0.0

    compound = _vader_compound(text)
    if compound >= gt.get("vader_min_compound", 0.05):
        sentiment_score = 0.30
    else:
        sentiment_score = 0.30 * ((compound + 1.0) / 2.0)

    reward = _clamp(length_score + power_score + sentiment_score)

    feedback = [
        f"{'Good' if length_score else 'Too long'} length ({char_count}/{gt['max_length']} chars)",
        f"Power words: {hit_words or 'none'}",
        f"VADER: {compound:.3f} -> {sentiment_score:.3f}",
    ]

    return {
        "reward": round(reward, 4),
        "feedback": " | ".join(feedback),
        "breakdown": {
            "length_score": round(length_score, 4),
            "power_score": round(power_score, 4),
            "sentiment_score": round(sentiment_score, 4),
            "char_count": char_count,
            "power_words_hit": hit_words,
            "vader_compound": round(compound, 4),
        },
    }


def grade_cold_email(text: str, gt: dict[str, Any]) -> dict[str, Any]:
    text = text.strip()

    word_count = len(text.split())
    wc_min, wc_max = gt["word_count_min"], gt["word_count_max"]

    if wc_min <= word_count <= wc_max:
        wc_score = 0.40
    else:
        nearest = wc_min if word_count < wc_min else wc_max
        wc_score = max(0.0, 0.40 - abs(word_count - nearest) * 0.004)

    lowered = text.lower()
    matched_phrase = next((p for p in gt["cta_phrases"] if p in lowered), None)
    cta_score = 0.35 if matched_phrase else 0.0

    fk = _flesch_ease(text)
    fk_min, fk_max = gt["fk_ease_min"], gt["fk_ease_max"]
    if fk_min <= fk <= fk_max:
        fk_score = 0.25
    elif 15.0 <= fk < fk_min:
        fk_score = 0.12
    else:
        fk_score = 0.0

    reward = _clamp(wc_score + cta_score + fk_score)

    feedback = [
        f"Words: {word_count} ({wc_min}-{wc_max}) -> {wc_score:.3f}",
        f"CTA: {matched_phrase or 'none found'}",
        f"Flesch: {fk:.1f} ({fk_min}-{fk_max}) -> {fk_score:.3f}",
    ]

    return {
        "reward": round(reward, 4),
        "feedback": " | ".join(feedback),
        "breakdown": {
            "wc_score": round(wc_score, 4),
            "cta_score": round(cta_score, 4),
            "fk_score": round(fk_score, 4),
            "word_count": word_count,
            "cta_match": matched_phrase,
            "flesch_ease": round(fk, 2),
        },
    }


def grade_ab_judge(text: str, gt: dict[str, Any]) -> dict[str, Any]:
    text = text.strip()

    match = re.search(r"(?:CHOICE|WINNER)\s*:\s*([AB])", text, re.IGNORECASE)
    chosen = match.group(1).upper() if match else None
    choice_score = 0.50 if chosen == gt["correct_choice"] else 0.0

    reason_hits = re.findall(gt["reason_pattern"], text, re.IGNORECASE)
    unique_reasons = len({r.upper() for r in reason_hits})
    reason_score = 0.30 * _clamp(unique_reasons / gt["required_reasons"])

    lowered = text.lower()
    kw_hits = [kw for kw in gt["evidence_keywords"] if kw in lowered]
    unique_kw = list(dict.fromkeys(kw_hits))[:3]
    kw_score = 0.20 * (len(unique_kw) / 3)

    reward = _clamp(choice_score + reason_score + kw_score)

    if choice_score == 0.0:
        choice_feedback = f"Wrong choice: {chosen}" if chosen else "No WINNER/CHOICE line"
    else:
        choice_feedback = f"Correct choice: {chosen}"

    feedback = [
        choice_feedback,
        f"Reasons: {unique_reasons}/{gt['required_reasons']} -> {reason_score:.3f}",
        f"Keywords: {unique_kw} -> {kw_score:.3f}",
    ]

    return {
        "reward": round(reward, 4),
        "feedback": " | ".join(feedback),
        "breakdown": {
            "choice_score": round(choice_score, 4),
            "reason_score": round(reason_score, 4),
            "kw_score": round(kw_score, 4),
            "chosen": chosen,
            "correct": gt["correct_choice"],
            "reasons_found": unique_reasons,
            "keywords_hit": unique_kw,
        },
    }
