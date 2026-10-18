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

## Consistency modelling assumptions

The consistency verifier additionally supports two explicit modelling
assumptions, enabled by default and independently toggleable:

- **Role witnesses:** every role named by a policy has at least one individual
  member.
- **Mutually exclusive roles:** no individual belongs to two distinct named
  roles.

These assumptions are not part of the three policy formulas. They are
verification-world assumptions used when evaluating a finite policy set. With
both enabled, a guest grant conflicts with `only administrators can ...`
because the guest witness is permitted to act but cannot also be an
administrator. With either assumption disabled, that conflict may have a
countermodel.
