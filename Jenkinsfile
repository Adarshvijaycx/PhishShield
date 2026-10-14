pipeline {
    agent { label 'phishshield-local' }

    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10', artifactNumToKeepStr: '5'))
    }

    triggers { pollSCM('H/5 * * * *') }

    parameters {
        booleanParam(name: 'DEMONSTRATE_FAILURE', defaultValue: false,
            description: 'Run a deliberately failing test to demonstrate the CI quality gate.')
    }

    environment {
        MLFLOW_DISABLE_TELEMETRY = 'true'
        MLFLOW_DISABLE_AGENT_HINT = '1'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        MPLCONFIGDIR = "${WORKSPACE}/.cache/matplotlib"
        LOKY_MAX_CPU_COUNT = '2'
        OMP_NUM_THREADS = '2'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Adarshvijaycx/PhishShield.git'
                sh '''
                    rm -rf reports
                    mkdir -p reports
                    git rev-parse HEAD > reports/source-revision.txt
                '''
                script {
                    currentBuild.description = params.DEMONSTRATE_FAILURE ?
                        'Intentional CI failure demonstration' : 'Tests and exported model verification'
                }
            }
        }

        stage('Python environment') {
            steps {
                sh '''
                    phishshield_python="${PHISHSHIELD_PYTHON:-python3}"
                    "$phishshield_python" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
                    if [ ! -x .venv-ci/bin/python ]; then
                        "$phishshield_python" -m venv .venv-ci
                    fi
                    .venv-ci/bin/python -m pip install --no-input -r requirements.lock
                    .venv-ci/bin/python -m pip install --no-input --no-deps -e .
                    .venv-ci/bin/python -m pip check
                    .venv-ci/bin/python -m compileall -q src app.py
                '''
            }
        }

        stage('Tests') {
            steps {
                sh '.venv-ci/bin/python -m pytest -q --junitxml=reports/pytest.xml'
            }
        }

        stage('Failure gate demonstration') {
            when { expression { params.DEMONSTRATE_FAILURE } }
            steps {
                writeFile file: 'reports/test_ci_gate_demo.py', text: '''def test_ci_quality_gate():
    assert False, "Intentional Jenkins failure demonstration"
'''
                sh '.venv-ci/bin/python -m pytest reports/test_ci_gate_demo.py -q --junitxml=reports/failure-demo.xml'
            }
        }

        stage('Model release verification') {
            steps {
                sh '''
                    test -n "$PHISHSHIELD_RELEASE_DIR"
                    .venv-ci/bin/python -m phishshield.cli verify --release "$PHISHSHIELD_RELEASE_DIR" > reports/release-verification.json
                    cp "$PHISHSHIELD_RELEASE_DIR/manifest.json" reports/release-manifest.json
                    cat reports/release-verification.json
                '''
            }
        }
    }

    post {
        always {
            script {
                if (fileExists('reports/pytest.xml')) {
                    junit testResults: 'reports/*.xml', allowEmptyResults: false, skipPublishingChecks: true
                } else {
                    echo 'No tests ran; inspect the failed setup stage.'
                }
            }
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true, fingerprint: true
        }
    }
}
