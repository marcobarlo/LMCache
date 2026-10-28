// SPDX-License-Identifier: Apache-2.0
// LMCache Ascend NPU CI — Jenkins Pipeline (trusted Jenkinsfile from branch dev only).
//
// Configure the pipeline job to load this file from `dev`, never from a PR head.
// Tests run from GIT_SHA via git archive inside ci/ascend/run_in_npu_container.sh.

pipeline {
    agent none

    options {
        disableConcurrentBuilds()
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '30'))
    }

    parameters {
        choice(name: 'MODE', choices: ['nightly', 'pr'], description: 'nightly or path-scoped pr run')
        string(name: 'GIT_SHA', description: 'Commit to test', trim: true)
        text(name: 'PYTEST_PATHS', description: 'Space-separated pytest paths (pr mode)', defaultValue: '')
        string(name: 'AGENT_LABEL', defaultValue: 'ascend && arm64 && 910b', description: 'Jenkins agent label')
        string(name: 'PR_NUMBER', defaultValue: '0', description: 'PR number (0 = nightly)')
    }

    stages {
        stage('Ascend tests') {
            agent { label "${params.AGENT_LABEL}" }
            options {
                lock(resource: 'ascend-910b-npu', inversePrecedence: true)
            }
            stages {
                stage('Checkout CI scripts') {
                    steps {
                        checkout scm
                        script {
                            if (!params.GIT_SHA?.trim()) {
                                error('GIT_SHA is required')
                            }
                            if (params.MODE == 'pr' && !params.PYTEST_PATHS?.trim()) {
                                error('PYTEST_PATHS is required for pr mode')
                            }
                            currentBuild.description = "mode=${params.MODE} sha=${params.GIT_SHA.take(7)} pr=${params.PR_NUMBER}"
                        }
                    }
                }

                stage('Pending status') {
                    steps {
                        withCredentials([string(credentialsId: 'github-lmcache-status', variable: 'GITHUB_TOKEN')]) {
                            sh """
                                python3 "\${WORKSPACE}/ci/ascend/github_status.py" \\
                                  --sha '${params.GIT_SHA}' \\
                                  --state pending \\
                                  --description 'Ascend NPU tests running' \\
                                  --target-url '${env.BUILD_URL}'
                            """
                        }
                    }
                }

                stage('Run container pytest') {
                    steps {
                        sh """
                            chmod +x ci/ascend/run_in_npu_container.sh
                            export GIT_SHA='${params.GIT_SHA}'
                            export MODE='${params.MODE}'
                            export PYTEST_PATHS='''${params.PYTEST_PATHS.replace("'", "'\"'\"'")}'''
                            export JUNIT_OUT="\${WORKSPACE}/ascend-junit.xml"
                            ./ci/ascend/run_in_npu_container.sh
                        """
                    }
                }
            }
            post {
                success {
                    withCredentials([string(credentialsId: 'github-lmcache-status', variable: 'GITHUB_TOKEN')]) {
                        sh """
                            python3 "\${WORKSPACE}/ci/ascend/github_status.py" \\
                              --sha '${params.GIT_SHA}' \\
                              --state success \\
                              --description 'Ascend NPU tests passed' \\
                              --target-url '${env.BUILD_URL}'
                        """
                    }
                }
                failure {
                    withCredentials([string(credentialsId: 'github-lmcache-status', variable: 'GITHUB_TOKEN')]) {
                        sh """
                            python3 "\${WORKSPACE}/ci/ascend/github_status.py" \\
                              --sha '${params.GIT_SHA}' \\
                              --state failure \\
                              --description 'Ascend NPU tests failed' \\
                              --target-url '${env.BUILD_URL}'
                        """
                    }
                }
                always {
                    junit allowEmptyResults: true, testResults: 'ascend-junit.xml'
                }
            }
        }
    }
}
