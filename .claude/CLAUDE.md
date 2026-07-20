# Coding Discipline

These are rules, not suggestions. Follow them on every task.

## Protocol

Follow this sequence for every task. Do not skip steps.

1. **Read** — Read every file you will touch. Read, not skim. Check imports, existing patterns, and conventions already in use. If you cannot find a pattern, ask — do not guess.
2. **Think** — State what you are about to do, which interpretation you picked, and what you are trading off. If the task is multi-step, state the full plan. Wait for confirmation before writing code.
3. **Write** — Write the minimum code that solves the stated problem. Then stop and self-check against the rules below before presenting it.

## Rules

### Simplicity
- Solve the problem in front of you now. Not every future version of it.
- No premature abstraction. If the only reason is "in case we need to," delete it.
- No error handling for errors that cannot occur.
- Hardcode values until there is a real reason to configure them.

### Surgical diffs
- Do not touch what you were not asked to touch.
- Match the existing style. Do not reformat.
- Every changed line must be justified by the task. "While I was in there" is not a justification — revert it.

### Verification
- When fixing a bug: write the failing test first, watch it fail, then fix it.
- Test behavior that can actually break, not that a constructor sets a field.
- Hard to test is information about the design, not permission to skip it.

### Dependencies
- Before adding a dependency, check whether the project or standard library already handles it.
- When you do add one, say why in your response.

### Debugging
- Read the whole error and stack trace. Reproduce the problem before changing anything.
- Change one thing at a time.
- Do not paper over a null with a null check. Find out why it is null.

## Communication

- Say what you did and why, not just code.
- Flag concerns even when you did exactly what was asked.
- Be precise about uncertainty. "I am not sure this library supports streaming" — good. "I think this should work" — unacceptable.

## Self-check: catch these failure modes

Before completing any task, check yourself for:
- **Kitchen Sink** — Am I restructuring code I was not asked to touch? Stop.
- **Wrong Abstraction** — Am I abstracting before I have seen the pattern repeat? Stop.
- **Optimistic Path** — Did I only handle the happy path? Handle the errors.
- **Runaway Refactor** — Is my fix cascading across files? Stop and reconsider scope.
