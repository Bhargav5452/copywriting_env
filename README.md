---
title: Copywriting AI Grading Environment
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Copywriting AI Grading Environment ✍️

Welcome to the **Copywriting Assessment Environment**! This environment is designed to benchmark and grade AI marketing agents across three tasks of increasing difficulty:

1.  **Task 1 (Easy)**: Rewriting raw subject lines for maximum open rates.
2.  **Task 2 (Medium)**: Drafting a cold outreach email focused on ROI and CFO pain points.
3.  **Task 3 (Hard)**: Choosing the winner of an A/B test and providing structured reasoning.

---

## 🛠️ How to Use (Gradio UI)

When using the **Step** tab in this interface, you need to provide the **Type**, **Tool Name**, and its **Arguments**.

### 1. Subject Line Rewrite
- **Type**: `call_tool`
- **Tool Name**: `subject_line_rewrite`
- **Arguments (JSON dictionary)**:
  ```json
  {
    "candidate": "Cut Your Invoice Processing Time by 50% With AI"
  }
  ```

### 2. Cold Email Draft
- **Type**: `call_tool`
- **Tool Name**: `cold_email_draft`
- **Arguments (JSON dictionary)**:
  ```json
  {
    "candidate": "Subject: Cut Invoice Processing Costs\n\nHi [Name],\n\nI noticed your AP process relies on manual entry. We help companies like yours automate this to save 40% in operational costs..."
  }
  ```

### 3. A/B Copy Judge
- **Type**: `call_tool`
- **Tool Name**: `ab_copy_judge`
- **Arguments (JSON dictionary)**:
  ```json
  {
    "candidate": "CHOICE: B\nREASON 1: Campaign B uses a stronger benefit-driven headline.\nREASON 2: It includes specific ROI metrics.\nREASON 3: It addresses the CFO's primary pain point directly."
  }
  ```

---

## 🤖 For AI Agents

This environment follows the **Model Context Protocol (MCP)**. You can discover the full JSON schema for these tools by taking a step with:

- **Action Type**: `list_tools`

### Connecting via Python
```python
from copywriting_env import CopywritingEnv

# Connect to this Space
with CopywritingEnv.from_env("your-hf-username/copywriting_env") as env:
    # 1. Start a session
    obs = await env.reset()
    
    # 2. Call a grading tool
    result = await env.step(subject_line_rewrite(candidate="Top-tier subject line here"))
    print(f"Reward: {result.reward}")
```

---

## 📊 Graders
- **Length**: Keeps copy concise (e.g., < 60 chars for subject lines).
- **Sentiment**: Uses VADER for optimal emotional resonance.
- **Readability**: Monitors Flesch-Kincaid ease scores.
- **Keywords**: Checks for ROI, Benefits, and Personalization evidence.
