# Contributing

Thanks for your interest in improving **challenge-data-checker**.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contribution quality and reviewer bandwidth

This is maintained by a small team. To keep review sustainable:

- **Submit PRs in a "ready to merge" state.** Tests pass, lint/type checks pass, the change is self-contained, and the description explains *why*.
- **Keep changes scoped.** One focused issue per PR. For anything non-trivial or user-facing, open an issue first so we can agree on the approach before you invest time.
- **New behaviour needs tests and docs.** A bug fix needs a regression test.
- We may close PRs that don't meet these standards without a full review. It's not personal — it's the only way a small team keeps up.
- Final decisions on what gets merged rest with the maintainers.

## Using AI coding assistants

**You're welcome to use AI coding assistants** (Copilot, Claude, Cursor, ChatGPT, and the like). They're good tools and we use them too.

What we ask:

- **Audit everything before you submit it.** You are the author of your PR. Read every line, understand why it works, check the edge cases, and confirm it fits the existing patterns and style of this codebase. If you can't explain it, don't submit it.
- **Run it.** Tests, lint, and type checks — "it looked right" is not verification.
- **Don't open PRs speculatively.** A batch of unreviewed AI-generated changes thrown at the repo to see what sticks wastes maintainer time and will be closed.
- **Slop will not be appreciated.** Plausible-looking code that doesn't quite work, invented APIs, tests that assert nothing, comments that restate the code, sweeping unrequested "improvements" — if a review turns up that pattern, expect the PR to be closed and further submissions to be held until you've demonstrated you're checking the output.

If AI tools contributed materially to a PR, that's fine and you don't need to flag it — but the PR checklist asks you to confirm you've personally verified the logic and edge cases. Check that box honestly.

## Getting started

1. Make sure you have a [GitHub account](https://github.com/signup/free).
2. [Fork](https://docs.github.com/get-started/quickstart/fork-a-repo) this repository and clone your fork.
3. Create the environment and run the tests:

   ```bash
   conda env create -f environment.yml      # or: mamba env create -f environment.yml
   conda activate challenge-data-checker
   pip install -e ".[parquet,test,remote]"
   pytest
   ```

## Making changes

- **Branch** off `main` with a descriptive name (`fix/tanimoto-threshold`, `feat/extra-column-alias`).
- **Match the surrounding code** — naming, docstring style, comment density. Look at nearby files before adding new patterns.
- **Run the checks locally** before pushing:

  ```bash
  pytest
  black --check src tests
  ruff check src tests
  mypy src
  pydoclint src
  ```

  > If `mypy` fails with `rdkit-stubs/.../*.pyi: error: Parameter without a
  > default follows parameter with a default  [syntax]`, that's a known bug
  > in RDKit's bundled, auto-generated stubs (rdkit/rdkit#8339, #7554,
  > #8673), not your code. Remove the broken stub package and re-run:
  >
  > ```bash
  > python -c "import os, shutil, rdkit; sp = os.path.dirname(os.path.dirname(rdkit.__file__)); shutil.rmtree(os.path.join(sp, 'rdkit-stubs'), ignore_errors=True)"
  > ```

- **Docs:** update `README.md` when behaviour, configuration options, or the Python API change.

## Opening a pull request

- Push your branch and open a PR against `main`. Fill in the template.
- CI must pass. A maintainer review with an "Approved" is required to merge.
- Address review feedback with additional commits on the same branch (don't force-push away the history mid-review unless asked).

## Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) — a lightweight statement that you wrote the contribution or otherwise have the right to submit it under the project's licence. There is no CLA to sign.

Unless you state otherwise, every contribution you submit is licensed under the [Apache License 2.0](LICENSE). The PR template includes a checkbox to certify the DCO; check it when you open the PR. (You may also sign off commits with `git commit -s` if you prefer.)

## Reporting security issues

Do **not** open a public issue for a security vulnerability. Follow [`SECURITY.md`](SECURITY.md).

## Additional resources

- [General GitHub documentation](https://docs.github.com/)
- [PR best practices](https://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
- [A guide to contributing to software packages](https://www.contribution-guide.org/)
