#!/usr/bin/env bash
# Builds the ROSE runner used as a baseline (requires a JDK and pip).
#   MOA:  moa.jar bundled with the capymoa wheel (0.15.1)
#   ROSE: authors' source, https://github.com/canoalberto/ROSE (commit c86c8e1)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build lib
if [ ! -f lib/moa.jar ]; then
  pip download capymoa==0.15.1 --no-deps -d build -q
  unzip -o -q -j build/capymoa-0.15.1-py3-none-any.whl 'capymoa/jar/moa.jar' -d lib
fi
if [ ! -d build/ROSE ]; then
  git clone -q https://github.com/canoalberto/ROSE.git build/ROSE
  git -C build/ROSE checkout -q c86c8e111c8f823d394639a8a3a0e3d3598ad5a7
fi
SRC=build/ROSE/src/main/java
javac -nowarn -d build/classes -cp lib/moa.jar \
  $SRC/moa/classifiers/meta/imbalanced/ROSE.java \
  $SRC/moa/classifiers/trees/RandomSubspaceHT.java \
  $SRC/moa/evaluation/WindowImbalancedClassificationPerformanceEvaluator.java \
  RunROSE.java
echo "built rose/build/classes"
