// The same pipeline as .github/workflows/ci.yml, written for Jenkins (declarative syntax).
pipeline {
  agent { docker { image 'python:3.11' } }          // every stage runs inside this container
  options { timeout(time: 15, unit: 'MINUTES') }
  stages {
    stage('Install')       { steps { sh 'pip install -r requirements.txt -r requirements-dev.txt' } }
    stage('Lint')          { steps { sh 'ruff check .' } }
    stage('Validate data') { steps { sh 'python ci/validate_data.py data/transactions.csv' } }
    stage('Train')         { steps { sh 'python train.py --data data/transactions.csv --out model' } }
    stage('Quality gate')  { steps { sh 'python ci/quality_gate.py' } }
    stage('Test')          { steps { sh 'python -m pytest -q --junitxml=test-results.xml' } }
    stage('Build image') {
      agent any                                      // needs a machine with Docker, not the python container
      steps {
        sh 'docker build -t fraud-api:${BUILD_NUMBER} .'
        sh 'docker run -d --name fraud-api-${BUILD_NUMBER} -p 8000:8000 fraud-api:${BUILD_NUMBER}'
        sh 'sleep 10 && python3 ci/smoke_test.py http://localhost:8000'
      }
      post { always { sh 'docker rm -f fraud-api-${BUILD_NUMBER} || true' } }
    }
  }
  post {
    always  { junit 'test-results.xml'; archiveArtifacts artifacts: 'model/**,metrics.json', allowEmptyArchive: true }
    failure { echo 'Pipeline failed: read the first red stage.' }
  }
}
