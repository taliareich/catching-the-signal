# Catching the Signal

A hands-on, small-scale replication of the core methodology in Azaria & Mitchell (2023), "The Internal State of an LLM Knows When It's Lying," extending the ideas in my independent research proposal, [*Catching the First Lie: An Outsider's Case for Studying Deception at Its Origin*](https://taliareich.medium.com/catching-the-first-lie-an-outsiders-case-for-studying-deception-at-its-origin-81db6b467563).

## What this is

A working pipeline that extracts a small language model's internal activations (via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)) while it processes true and false factual statements, then trains a probe classifier to see whether those activations carry information about a statement's truth value, separate from anything in the model's literal output.

This notebook documents a real, honest research process, not a clean success story: forming a specific hypothesis about a confusing result, testing it directly against data, and ruling it out. The final result is genuinely inconclusive, and the notebook says so plainly. The value here is in the methodology, the diagnostic process, and the honest reporting, not a tidy headline number.

## What's inside

- **Steps 1-5:** Setup, loading GPT-2 small via `HookedTransformer`, extracting residual stream activations across 50 hand-built true/false statement pairs, and training a regularized logistic regression probe with cross-validation.
- **Steps 6-7:** Diagnosing an unexpected below-chance result, checking whether it reflects a labeling error, pure noise, or a systematic pattern, then formally testing and ruling out a specific hypothesis (that the probe was tracking the model's own word-fluency confidence rather than truth) using a statistical correlation test.
- **Step 8:** An honest findings section: what was ruled out, what remains unexplained, and why this result motivates a different, more tractable next step.
- **Step 9:** Concrete extensions, including a pivot toward tracking a model's behavior across training checkpoints, a direction suggested by an AI safety researcher after reviewing my original proposal.

## Built with

Python, PyTorch, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), scikit-learn, on GPT-2 small. Designed to run in [Google Colab](https://colab.research.google.com/) on a free T4 GPU.

## Related work

- [Catching the First Lie](https://taliareich.medium.com/catching-the-first-lie-an-outsiders-case-for-studying-deception-at-its-origin-81db6b467563) — the original research proposal this notebook extends
- [stroke-predictions](https://github.com/taliareich/stroke-predictions) — an earlier applied ML project (XGBoost, class imbalance handling)
