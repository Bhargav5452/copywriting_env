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

    # 1. Evaluate Choice (60% Weight)
    match = re.search(r"(?:CHOICE|WINNER)\s*:\s*([AB])", text, re.IGNORECASE)
    chosen = match.group(1).upper() if match else None
    choice_correct = (chosen == gt["correct_choice"])
    # Per user: 0.95 if correct else 0.05
    choice_base = 0.95 if choice_correct else 0.05
    choice_weighted = 0.6 * choice_base

    # 2. Evaluate Reasoning Quality (40% Weight)
    # Count unique reason patterns (e.g., REASON 1:)
    reason_hits = re.findall(gt["reason_pattern"], text, re.IGNORECASE)
    unique_reasons = len({r.upper() for r in reason_hits})
    reason_norm = _clamp(unique_reasons / gt["required_reasons"], 0.0, 1.0)

    # Keywords evidence
    lowered = text.lower()
    kw_hits = [kw for kw in gt["evidence_keywords"] if kw in lowered]
    unique_kw = list(dict.fromkeys(kw_hits))[:3]
    kw_norm = len(unique_kw) / 3

    # reasoning_quality (0.0 to 1.0)
    reasoning_quality = (0.6 * reason_norm) + (0.4 * kw_norm)
    reasoning_weighted = 0.4 * reasoning_quality

    # Final Weighted score
    reward = _clamp(choice_weighted + reasoning_weighted)

    if not choice_correct:
        choice_feedback = f"Wrong choice: {chosen}" if chosen else "No WINNER/CHOICE line"
    else:
        choice_feedback = f"Correct choice: {chosen}"

    feedback = [
        choice_feedback,
        f"Reasoning Quality: {reasoning_quality:.2f} ({unique_reasons} reasons, {len(unique_kw)} keywords)",
    ]

    return {
        "reward": round(reward, 4),
        "feedback": " | ".join(feedback),
        "breakdown": {
            "choice_weighted": round(choice_weighted, 4),
            "reasoning_weighted": round(reasoning_weighted, 4),
            "choice_correct": choice_correct,
            "reasons_found": unique_reasons,
            "keywords_hit": unique_kw,
            "reasoning_quality": round(reasoning_quality, 4),
        },
    }
