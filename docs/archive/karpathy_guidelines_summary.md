# Karpathy Guidelines Summary

These are behavioral guidelines for AI coding assistants, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on common LLM coding mistakes. The core idea: LLMs tend to over-engineer, make silent assumptions, and touch more code than necessary. These rules counteract that.

---

## 1. Think Before Coding

**Problem it solves:** LLMs often jump straight into writing code based on their best guess of what you want, without checking if that guess is right. This leads to wasted work when the assumption was wrong.

**The principle:** Before writing any code, state your assumptions out loud. If something is ambiguous, stop and ask rather than picking an interpretation silently. If a simpler approach exists than what was asked for, say so.

**Why it matters for this project:** In the Alpha Challenge, a wrong assumption about how a feature should be engineered or how the walk-forward loop should work could introduce look-ahead bias — which is a disqualifying condition. Better to surface uncertainty than to silently introduce a bug.

---

## 2. Simplicity First

**Problem it solves:** LLMs love to over-engineer. They add error handling for impossible cases, build abstractions for things used once, add "configurability" nobody asked for, and write 200 lines when 50 would do. This makes code harder to read, debug, and maintain.

**The principle:** Write the minimum code that solves the problem. No speculative features, no premature abstractions, no unnecessary flexibility. If a senior engineer would look at it and say "this is overcomplicated," simplify it.

**Why it matters for this project:** The grading rubric explicitly penalizes "excessive parameterization." A simple model with clear justification scores better than a complex one that looks like it was tuned to fit the data. Every extra parameter or feature needs to earn its place.

---

## 3. Surgical Changes

**Problem it solves:** When asked to fix one thing, LLMs often "improve" nearby code — reformatting comments, renaming variables, refactoring functions that work fine. This creates noisy diffs, can introduce new bugs, and makes it hard to review what actually changed.

**The principle:** Only touch what the task requires. Match existing code style even if you'd do it differently. If you notice unrelated issues, mention them but don't fix them unless asked. The test: every changed line should trace directly back to what was requested.

**Exception:** If your own changes make something unused (an import, a variable), clean that up. But don't delete pre-existing dead code.

**Why it matters for this project:** The notebook has evolved through many iterations. "Improving" working code risks breaking the walk-forward engine or changing results in subtle ways that are hard to catch.

---

## 4. Goal-Driven Execution

**Problem it solves:** Vague tasks like "fix the bug" or "add validation" lead to vague implementations. Without clear success criteria, it's impossible to know when you're done, and easy to either under-deliver or over-deliver.

**The principle:** Before starting, convert the task into verifiable goals with concrete checks. For example:
- "Add validation" becomes "write tests for invalid inputs, then make them pass"
- "Fix the bug" becomes "write a test that reproduces it, then make it pass"
- "Improve SR" becomes "run backtest, record SR/MDD, compare to baseline"

For multi-step work, state a brief plan with verification at each step.

**Why it matters for this project:** Every strategy change (K tuning, vol tilt, feature selection) needs a clear before/after comparison. Without explicit success criteria (SR improved? MDD still under 40%?), it's easy to make a change that looks good in one metric but fails another.

---

## Summary

| Guideline | One-liner | Bias |
|---|---|---|
| Think Before Coding | Don't assume, ask | Caution over speed |
| Simplicity First | Less code, fewer features | Minimalism over flexibility |
| Surgical Changes | Only touch what's needed | Precision over cleanup |
| Goal-Driven Execution | Define "done" before starting | Verification over vibes |

These guidelines bias toward caution over speed. For trivial tasks, use judgment — not everything needs a formal plan. But for anything that touches model logic, feature engineering, or backtesting, these principles help prevent the kind of silent mistakes that are hardest to catch.
