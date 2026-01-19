#!/bin/bash
# Run full test suite with coverage and save results
# Usage: ./scripts/run_tests.sh [pytest-args]
#
# Results saved to:
#   .claude/test_results/latest.txt - Summary with timestamp
#   .claude/test_results/coverage.txt - Detailed coverage report
#   htmlcov/ - HTML coverage report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RESULTS_DIR=".claude/test_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "Running tests at $TIMESTAMP..."
echo "=============================================="

# Run tests with coverage (parallel for speed)
.venv/bin/pytest tests/ \
    --cov=ml4t.diagnostic \
    --cov-report=term-missing \
    --cov-report=html \
    --ignore=tests/test_slow \
    -n auto \
    -q \
    "$@" 2>&1 | tee "$RESULTS_DIR/full_output.txt"

# Extract summary
{
    echo "Test Results - $TIMESTAMP"
    echo "=============================================="
    echo ""
    # Get pass/fail summary
    tail -20 "$RESULTS_DIR/full_output.txt" | grep -E "passed|failed|error|skipped|warning" | tail -5
    echo ""
    # Get coverage percentage
    echo "Coverage Summary:"
    grep "^TOTAL" "$RESULTS_DIR/full_output.txt" || echo "Coverage data not found"
    echo ""
    echo "HTML report: htmlcov/index.html"
} > "$RESULTS_DIR/latest.txt"

# Save detailed coverage
grep -A 1000 "^Name" "$RESULTS_DIR/full_output.txt" | grep -B 1000 "^TOTAL" > "$RESULTS_DIR/coverage.txt" 2>/dev/null || true

echo ""
echo "=============================================="
echo "Results saved to:"
echo "  $RESULTS_DIR/latest.txt"
echo "  $RESULTS_DIR/coverage.txt"
echo "  htmlcov/index.html"
echo ""
cat "$RESULTS_DIR/latest.txt"
