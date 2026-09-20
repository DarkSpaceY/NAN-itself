# SKILL.md template

Copy this skeleton into `<skill-name>/SKILL.md` and fill it in.

```
---
name: <skill-name>
description: >-
  What the skill does and when to use it, in one or two sentences.
  This text is what the agent sees when deciding to load the skill,
  so make the trigger conditions explicit.
---

# <Skill Title>

One short paragraph: the skill's purpose and outcome.

## When to use

- Trigger condition 1
- Trigger condition 2

## Procedure

1. Step one.
2. Step two.
3. ...

## Validation

How to verify the result (command to run, expected output).

## Notes

Edge cases, constraints, and gotchas. Details that only matter
once the agent is already committed belong in references/*.md
and are linked from here.
```

## Field rules (enforced by the runtime)

- `name`: `^[a-z0-9]+(-[a-z0-9]+)*$`, max 64 chars, unique across all
  skill roots.
- `description`: non-empty string, max 1024 chars.
- Frontmatter starts at line 1 with `---` and ends at the next `---`
  line; the remainder of the file is the instruction body.
- Extra frontmatter keys are allowed and preserved in metadata
  (`frontmatter` mapping), but only `name` and `description` are
  required and validated.
