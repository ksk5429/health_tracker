#!/bin/bash
# Deploy interactive dashboard to the public health_dashboard repo for GitHub Pages.
# Usage: bash scripts/deploy_pages.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$REPO_ROOT/dashboard/interactive"
DEPLOY_REPO="https://github.com/ksk5429/health_dashboard.git"
TEMP_DIR=$(mktemp -d)

echo "=== Deploying interactive dashboard to GitHub Pages ==="

# Clone the deploy repo
git clone --depth 1 "$DEPLOY_REPO" "$TEMP_DIR"

# Copy dashboard files
cp -r "$DASHBOARD_DIR"/* "$TEMP_DIR/"

cd "$TEMP_DIR"
git add -A
if git diff --cached --quiet; then
    echo "No changes to deploy."
else
    git commit -m "chore: update dashboard $(date +%Y-%m-%d)"
    git push origin main
    echo "=== Deployed to https://ksk5429.github.io/health_dashboard/ ==="
fi

rm -rf "$TEMP_DIR"
