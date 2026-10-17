# Supported policy grammar

The parser intentionally supports a small deterministic grammar. It does not
infer policy meaning from arbitrary natural language.

```text
<role> can <action> <resource>.
only <role> can <action> <resource>.
<role> cannot <action> <resource>.
```

Whitespace and capitalization are normalized. Roles and resources may contain
letters, digits, spaces, and hyphens; actions are single words.

## Semantics

- `<role> can ...` produces
  `∀x (Role(x) → CanAction(x, Resource))`.
- `only <role> can ...` produces
  `∀x (CanAction(x, Resource) → Role(x))`.
- `<role> cannot ...` produces
  `∀x (Role(x) → ¬CanAction(x, Resource))`.

The first two forms are deliberately not equivalent. Text outside these forms
returns an unsupported-policy error and is never converted by guessing.
