# PhishShield Jenkins setup and academic demonstration

Jenkins LTS **2.580.1** is configured locally at <http://127.0.0.1:8080>. Open [PhishShield-CI](http://127.0.0.1:8080/job/PhishShield-CI/) for the pipeline and build history.

## Login

- Username: `phishshield-admin`.
- The generated password is in the private local `.jenkins-local/access.json` file. This file is mode 0600 inside a mode 0700 directory and excluded from Git.
- Account signup and anonymous dashboard/API access are disabled. The controller listens on loopback only.

Local credentials and agent connection secrets are not included in repository documentation, build artifacts, or verification summaries.

## Verified configuration

| Item | Setting |
|---|---|
| Jenkins installation | Homebrew `jenkins-lts`, version 2.580.1 |
| Java runtime | Homebrew OpenJDK 21 |
| Controller executors | 0 |
| Dedicated inbound agent | `phishshield-local`, one executor, exclusive label |
| Agent transport | WebSocket to local controller; private secret file |
| Source checkout | GitHub `Adarshvijaycx/PhishShield`, branch `main` |
| Python | Installed Python 3.13.2; isolated workspace `.venv-ci` |
| Dependencies | `requirements.lock`, editable package, `pip check` |
| Tests | Pytest with JUnit XML; results published in Jenkins |
| Release checks | Model checksum and 20 frozen prediction snapshots; actual version 3 |
| Scheduling | Manual builds and `H/5 * * * *` SCM polling |
| Retention | 10 builds; artifacts from 5 builds |
| Startup | Controller and agent restart at user login |

Jenkins stages are Checkout → Python environment → Tests → optional Failure gate demonstration → Model release verification. Reports and release metadata are fingerprinted and archived. Routine builds verify the existing release and do not retrain, promote models, or contact submitted websites.

The configured job stores a sandboxed Pipeline script synchronized from the local [Jenkinsfile](../Jenkinsfile). Each build checks out GitHub main. After editing Jenkinsfile, synchronize its definition with the control command below; application-code commits on GitHub are detected by SCM polling. This setup did not push Jenkins changes to GitHub.

Python 3.13 is used because it is installed and matches the tested local environment. The application supports Python 3.12 and 3.13; the separate GitHub Actions matrix covers both.

## Completed build evidence

| Build | Result | Evidence |
|---|---|---|
| [#2](http://127.0.0.1:8080/job/PhishShield-CI/2/) | SUCCESS | First complete checkout, 217 passing tests, release verification |
| [#3](http://127.0.0.1:8080/job/PhishShield-CI/3/) | Expected FAILURE | `DEMONSTRATE_FAILURE=true`; real failing assertion, 217 ordinary tests passed, one failed demo test |
| [#4](http://127.0.0.1:8080/job/PhishShield-CI/4/) | SUCCESS | Normal pipeline rerun after the failure demonstration |

Build #1 records an initial Git tool configuration error; it did not reach the tests. That configuration was corrected before the successful runs. Build history is retained rather than presenting that setup error as a quality-gate demonstration.

[Repository verification summary](results/jenkins-verification.json) records real build results, source commit IDs, test counts, pipeline checksum, and the model release identity. Copies of JUnit/release reports are saved under `reports/jenkins/`, which is ignored by Git.

## Control commands

Run these from the project folder:

```sh
.venv/bin/python scripts/jenkins_control.py status
.venv/bin/python scripts/jenkins_control.py validate
.venv/bin/python scripts/jenkins_control.py build
```

After changing Jenkinsfile:

```sh
.venv/bin/python scripts/jenkins_control.py sync-pipeline
```

For the deliberate failure demonstration:

```sh
.venv/bin/python scripts/jenkins_control.py build --failure-demo
```

The failure parameter defaults to false. It creates an assertion failure only in the Jenkins workspace and publishes the resulting XML; it does not modify application tests in GitHub. Run a normal build afterward to restore a passing latest result.

## Two-minute review

1. Open the job and latest successful build. Show the stages, source checkout, and source commit artifact.
2. Open **Test Result** and show 217 passed, zero failed, zero skipped.
3. Open **Artifacts → reports/release-verification.json**: version 3, source MLflow run, model SHA-256, and 20 checks passed.
4. Open the deliberate-failure build. Show the failed assertion and published JUnit result, then return to the successful latest build.
5. Under **Manage Jenkins → Nodes**, show controller executors 0 and the online dedicated agent with one executor.

## Local services and recovery

The controller uses Homebrew's `homebrew.mxcl.jenkins-lts` user service:

```sh
brew services start jenkins-lts
brew services restart jenkins-lts
brew services stop jenkins-lts
```

The agent uses `com.phishshield.jenkins-agent` in `~/Library/LaunchAgents/`. Restart it while no build is active:

```sh
launchctl kickstart -k "gui/$(id -u)/com.phishshield.jenkins-agent"
```

Jenkins state is in `~/.jenkins`; the dedicated agent's workspace/logs are in `~/.jenkins-agent/phishshield-local`. Keep the project release directory at its configured location; if it moves, update the agent's `PHISHSHIELD_RELEASE_DIR` node environment setting. The job requires a locally exported release because model files are intentionally excluded from GitHub.

After stopping the services, back up Jenkins home, agent state, and private local access files together. The exported JSON evidence is a review record, not a complete controller backup. If you change the Jenkins password, update the private access file used by the control script.

References: [official macOS installation](https://www.jenkins.io/download/lts/macos/), [Pipeline syntax](https://www.jenkins.io/doc/book/pipeline/syntax/), [remote API](https://www.jenkins.io/doc/book/using/remote-access-api/), and [agent concepts](https://www.jenkins.io/doc/book/using/using-agents/).
