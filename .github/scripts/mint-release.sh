#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 0.3.3          # stable
#     .github/scripts/mint-release.sh 1.0.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
version=$1 core=${1%%-*}
branch=$(git branch --show-current)
previous=$(git describe --tags --abbrev=0)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

echo "$version" > VERSION
sed -i -E "s/^version = \"[^\"]+\"/version = \"$version\"/" pyproject.toml
sed -i -E "s/version=\"[^\"]+\",/version=\"$version\",/" setup.py
sed -i -E "s/(\"__version__\", \")[^\"]+/\1$version/" python/module.cpp
sed -i -E "s/^([[:space:]]+VERSION )[0-9.]+$/\1$core/" CMakeLists.txt
git commit -am "Release: v$version" # commit-msg rejects it if any site was missed
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" "${flags[@]}"
