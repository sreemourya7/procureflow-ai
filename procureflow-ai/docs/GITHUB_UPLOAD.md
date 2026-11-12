# Upload to GitHub

Extract the ZIP, then open PowerShell inside the `procureflow-ai` folder (the folder
containing `pyproject.toml` and `README.md`). Use a **new, empty repository**.

## GitHub CLI (recommended if already installed)

```powershell
gh auth login
git init
git add .
git commit -m "Build evidence-backed procurement review agents with PyTorch and recovery benchmarks"
git branch -M main
gh repo create sreemourya7/procureflow-ai --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if you want to review it privately first. Run
`gh auth login` only if you are not already authenticated. The commands require Git
and GitHub CLI to be installed. No credential or token belongs in this repository.

## Standard Git (without GitHub CLI)

On GitHub create a repository named `procureflow-ai` under `sreemourya7`. Leave
**README, .gitignore, and license unchecked** so the remote is empty. Then:

```powershell
git init
git add .
git commit -m "Build evidence-backed procurement review agents with PyTorch and recovery benchmarks"
git branch -M main
git remote add origin https://github.com/sreemourya7/procureflow-ai.git
git push -u origin main
```

Authenticate through your normal GitHub credential flow. If the remote name already
exists, inspect `git remote -v` before changing it. Do not force-push over existing work.

## Suggested repository description

Evidence-backed procurement agents with PyTorch routing, policy retrieval,
human review, durable checkpoints, and reproducible synthetic failure benchmarks.

Suggested topics: `agentic-ai`, `pytorch`, `llm`, `fastapi`, `rag`, `mlops`,
`workflow-automation`, `evaluation`, `procurement`.
