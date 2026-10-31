#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 0.3.3          # stable
#     .github/scripts/mint-release.sh 1.0.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
export version=$1 core=${1%%-*}
branch=$(git branch --show-current)
previous=$(git describe --tags --abbrev=0)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

echo "$version" > VERSION
perl -pi -e 's/^version = "[^"]+"/version = "$ENV{version}"/' pyproject.toml
perl -pi -e 's/version="[^"]+",/version="$ENV{version}",/' setup.py
perl -pi -e 's/("__version__", ")[^"]+/$1$ENV{version}/' python/module.cpp
perl -pi -e 's/^(\s+VERSION )[\d.]+$/$1$ENV{core}/' CMakeLists.txt
git commit -am "Release: v$version" # commit-msg rejects it if any site was missed
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" "${flags[@]}"
