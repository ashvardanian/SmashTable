#!/usr/bin/env bash
# Runs the repository's Git hooks over every commit since a base, as they ran on each commit.
#
#     .github/scripts/check-commits.sh <base>
set -euo pipefail
base=$1
.githooks/selftest
message=$(mktemp)
for commit in $(git rev-list --no-merges "$base..HEAD"); do
    git log -1 --format=%B "$commit" > "$message"
    .githooks/commit-msg "$message"
done
# Staging the whole range lets `pre-commit` judge every added line; the trap puts HEAD back.
head=$(git rev-parse HEAD)
trap 'git reset --quiet --soft "$head"' EXIT
git reset --quiet --soft "$base"
.githooks/pre-commit
