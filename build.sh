#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Force using binary packages when available
export PIP_PREFER_BINARY=1

# Skip Cython compilation
export SKLEARN_NO_OPENMP=1

# Install dependencies with no build isolation to avoid Cython issues
pip install --no-build-isolation --no-cache-dir -r requirements.txt 