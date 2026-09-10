# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests, or discussions.**

Instead, use one of these private channels:

1. **Preferred:** GitHub's private vulnerability reporting — go to the **Security** tab of this repository and click **"Report a vulnerability"**.
2. **Email:** `openadmet@omsf.io` — the same address as the Code of Conduct contact. Use a subject line starting with `[SECURITY]`.

Please include:

- the affected component and file(s),
- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- any suggested remediation.

You'll get an acknowledgement within **5 working days**. We aim to agree on a fix and disclosure timeline within **30 days** of the report, and will keep you updated on progress. We'll credit you in the release notes unless you ask us not to.

## Scope

This repository is a command-line tool and Python library that reads local or remote CSV/parquet files, parses them with RDKit, and writes a report. In scope for a report:

- A malformed or adversarial input file (CSV/parquet) that causes something worse than a clean error — e.g. arbitrary code execution, a crash exploitable for denial of service, or reading/writing outside the intended paths.
- The remote data-fetching path (`challenge_data_checker.remote`) — e.g. a URL or redirect that leaks credentials/tokens, follows an unsafe redirect, or is otherwise abusable beyond the intended HuggingFace Hub / plain-HTTP download.
- Report generation writing sensitive data (tokens, file-system paths outside the project) into output that wasn't requested.
- Committed secrets or credentials of any kind, or a dependency pin with a known vulnerability that affects this tool's usage.

Out of scope:

- Vulnerabilities in third-party dependencies (RDKit, pandas, etc.) themselves — report them upstream. If a dependency issue requires a change here (a version pin, a workaround), a normal issue/PR is fine unless the details are themselves sensitive.
- Issues that require the user to intentionally point the tool at a malicious/untrusted URL or file with no realistic accidental trigger.
- Findings that require control of a maintainer's machine or GitHub org admin.
