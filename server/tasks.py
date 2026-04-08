"""
Task definitions and ground-truth data for copywriting_env.
"""

try:
    from graders import grade_ab_judge, grade_cold_email, grade_subject_line
except ImportError:
    from .graders import grade_ab_judge, grade_cold_email, grade_subject_line


TASKS = {
    "subject_line_rewrite": {
        "id": "subject_line_rewrite",
        "difficulty": "easy",
        "grader": grade_subject_line,
        "prompt": (
            "Rewrite the following weak email subject line to maximise open rates.\n\n"
            "ORIGINAL: 'Monthly newsletter - please read'\n\n"
            "Rules:\n"
            "  • Maximum 60 characters (spaces included)\n"
            "  • Must feel urgent, exclusive, or benefit-driven\n"
            "  • Return ONLY the rewritten subject line — no explanation"
        ),
        "context": {
            "original": "Monthly newsletter - please read",
            "max_chars": 60,
        },
        "ground_truth": {
            "max_length": 60,
            "power_words": [
                "boost", "save", "grow", "proven", "instant", "exclusive",
                "free", "secret", "unlock", "win", "limited", "guaranteed",
                "results", "transform", "discover", "reveal", "master",
                "skyrocket", "double", "triple", "earn", "fast",
            ],
            "vader_min_compound": 0.05,
        },
    },

    "cold_email_draft": {
        "id": "cold_email_draft",
        "difficulty": "medium",
        "grader": grade_cold_email,
        "prompt": (
            "You previously wrote this ad headline for a campaign:\n"
            "{headline}\n\n"
            "Now write a cold email that delivers on the promise of that headline.\n"
            "Target: CFO of a mid-size manufacturing company.\n"
            "Include: opener referencing the headline's promise, value prop, one CTA.\n"
            "Length: 80-160 words. Professional tone."
        ),
        "context": {
            "product_name": "FinFlow",
            "product_description": "AI-powered cash-flow forecasting platform",
            "target_role": "CFO",
            "target_company_size": "200-person manufacturing company",
        },
        "ground_truth": {
            "word_count_min": 80,
            "word_count_max": 160,
            "cta_phrases": [
                "schedule", "book a", "reply", "let me know", "sign up",
                "get started", "reach out", "open to", "15-minute", "15 minute",
                "quick call", "brief call", "demo", "connect", "hop on",
                "would you be", "are you available", "free to chat", "meet"
            ],
            "fk_ease_min": 20.0,
            "fk_ease_max": 75.0,
        },
    },

    "ab_copy_judge": {
        "id": "ab_copy_judge",
        "difficulty": "hard",
        "grader": grade_ab_judge,
        "prompt": (
            "You have two complete B2B marketing campaigns:\n\n"
            "CAMPAIGN A:\n"
            "  Headline: 'We Have Great Software For You'\n"
            "  Email opener: 'Dear Finance Team, our product has many features...'\n\n"
            "CAMPAIGN B:\n"
            "  Headline: 'Cut Invoice Processing Time by 80%'\n"
            "  Email opener: 'Hi [Name], manufacturers like Toyota saved 6hrs/week...'\n\n"
            "Which campaign performs better for cold B2B outreach to CFOs?\n"
            "Use this exact format:\n"
            "WINNER: [A or B]\n"
            "REASON 1: ...\n"
            "REASON 2: ...\n"
            "REASON 3: ..."
        ),
        "_ground_truth": "B",
        "context": {
            "campaign_a_headline": "We Have Great Software For You",
            "campaign_a_opener": "Dear Finance Team, our product has many features...",
            "campaign_b_headline": "Cut Invoice Processing Time by 80%",
            "campaign_b_opener": "Hi [Name], manufacturers like Toyota saved 6hrs/week...",
            "correct_choice": "B",
        },
        "ground_truth": {
            "correct_choice": "B",
            "reason_pattern": r"REASON\s*[123]\s*:",
            "evidence_keywords": [
                "social proof", "personali", "specific",
                "benefit", "pain point", "quantif", "value",
                "outcome", "credib", "feature", "result",
                "trust", "friction", "conver"
            ],
            "required_reasons": 3,
        },
    },
}
