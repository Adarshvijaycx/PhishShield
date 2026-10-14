# Publish the prepared project to GitHub

This project is prepared locally on branch `main`. Publishing remains a separate step; no remote repository or public upload is created by this preparation.

## Create the destination

On GitHub, create an empty repository named **PhishShield** in the intended personal or team account. Choose public or private visibility for the academic review. Leave GitHub's README, .gitignore, and license initialization options unchecked because this project already includes them.

Replace `OWNER` below with the destination account name. From the project folder:

```sh
git remote add origin https://github.com/OWNER/PhishShield.git
git push -u origin main
```

If a remote already exists, inspect `git remote -v` before changing it. Authenticate through the normal GitHub credential flow; do not embed a token in the remote URL or a committed file.

## What will be published

- Application/package source, tests, dependency pins, and Streamlit theme.
- MIT code license, README, contribution guide, model card, and academic plan.
- Small charts and a summary of the recorded version-2 evaluation.
- GitHub Actions workflow for tests on Python 3.12 and 3.13.

Local datasets, `.venv`, `artifacts`, `mlartifacts`, the MLflow database, credentials, and cache files are ignored. A fresh clone does not contain a trained model or the original MLflow runs; follow the README's download and training steps to create them. Keep the original local MLflow state for the academic demo. A model release asset can be added later if the team chooses to distribute trained artifacts.

## Verify after pushing

Open the repository and check the README, license, and chart images. Open **Actions → Tests** and confirm both Python jobs finish successfully. A local passing suite does not establish that the hosted workflow has already passed.

For team work, add the other members through repository settings and use branches/pull requests. Jira setup and Hugging Face deployment are later project phases.
