"""Manage the configured local PhishShield Jenkins job using private credentials."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'validate', 'sync-pipeline', 'build'])
    parser.add_argument('--failure-demo', action='store_true', help='For build: run the intentional failing-test demonstration')
    args = parser.parse_args()
    try:
        access = json.loads((ROOT / '.jenkins-local/access.json').read_text())
        client = requests.Session()
        client.auth = (access['username'], access['password'])
        url = access['url']

        def get(path, **kwargs):
            response = client.get(url + path, timeout=20, **kwargs)
            response.raise_for_status()
            return response

        def post(path, **kwargs):
            crumb = get('/crumbIssuer/api/json').json()
            response = client.post(url + path, headers={crumb['crumbRequestField']:crumb['crumb']}, timeout=30, **kwargs)
            response.raise_for_status()
            return response

        if args.command == 'status':
            result = get('/job/PhishShield-CI/api/json', params={'tree':'name,buildable,lastBuild[number,result,building,url],lastSuccessfulBuild[number,url],lastFailedBuild[number,url],queueItem[id,why]'}).json()
            result['agents'] = get('/computer/api/json', params={'tree':'computer[displayName,offline,numExecutors]'}).json()['computer']
            print(json.dumps(result, indent=2))
        elif args.command in {'validate', 'sync-pipeline'}:
            pipeline = (ROOT / 'Jenkinsfile').read_text()
            validation = post('/pipeline-model-converter/validate', data={'jenkinsfile':pipeline}).text
            if 'Jenkinsfile successfully validated.' not in validation:
                print(validation, file=sys.stderr)
                return 1
            print(validation.strip())
            if args.command == 'sync-pipeline':
                encoded = base64.b64encode(pipeline.encode()).decode()
                script = ('import jenkins.model.Jenkins\n'
                          'import org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition\n'
                          'def job = Jenkins.get().getItem("PhishShield-CI")\n'
                          f'def source = new String(java.util.Base64.decoder.decode("{encoded}"), "UTF-8")\n'
                          'job.setDefinition(new CpsFlowDefinition(source, true))\njob.save()\n'
                          'println("PhishShield pipeline synchronized.")\n')
                print(post('/scriptText', data={'script':script}).text.strip())
        elif args.command == 'build':
            response = post('/job/PhishShield-CI/buildWithParameters', data={'DEMONSTRATE_FAILURE':str(args.failure_demo).lower()})
            print(json.dumps({'queued':True, 'queue_url':response.headers.get('Location'),
                              'intentional_failure':args.failure_demo}, indent=2))
        return 0
    except (OSError, ValueError, requests.RequestException) as exc:
        print(f'Jenkins command failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
