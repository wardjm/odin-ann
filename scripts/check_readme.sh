#!/usr/bin/env bash
# Builds every ```odin block in README.md that declares `package main` against
# this library and the sibling odin-tensor checkout, and fails if any of them
# does not compile. Run from anywhere:
#
#     scripts/check_readme.sh [README.md]
#
# The placeholder imports "path/to/odin-ann" and "path/to/odin-tensor" are
# rewritten to point at the checkouts. Blocks without `package main` are
# fragments and are skipped.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readme=${1:-"$repo_root/README.md"}
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

awk -v out="$work_dir" '
	/^```odin[[:space:]]*$/ { in_block = 1; dir = out "/line_" NR; system("mkdir -p " dir); file = dir "/main.odin"; next }
	/^```/ && in_block      { in_block = 0; close(file); next }
	in_block                { print > file }
' "$readme"

# A collection rooted at the parent directory reaches both checkouts.
collection="readme_lib=$(dirname "$repo_root")"

blocks=0
failures=0
for dir in $(ls -d "$work_dir"/line_* 2>/dev/null | sort -t_ -k2 -n); do
	line=${dir##*_}
	if ! grep -q '^package main' "$dir/main.odin"; then
		echo "SKIP README.md:$line (fragment)"
		continue
	fi
	blocks=$((blocks + 1))
	sed -e "s|\"path/to/odin-ann\"|\"readme_lib:$(basename "$repo_root")\"|" \
		-e "s|\"path/to/odin-tensor\"|\"readme_lib:odin-tensor\"|" "$dir/main.odin" > "$dir/main.odin.tmp"
	mv "$dir/main.odin.tmp" "$dir/main.odin"
	if output=$(odin build "$dir" -collection:"$collection" -vet -strict-style -out:"$dir/example" 2>&1); then
		echo "PASS README.md:$line"
	else
		failures=$((failures + 1))
		echo "FAIL README.md:$line"
		echo "$output" | sed "s|$dir/main.odin|README.md block at line $line|"
	fi
done

if [ "$blocks" -eq 0 ]; then
	echo "no complete \`\`\`odin programs found in $readme"
	exit 1
fi
echo "$((blocks - failures))/$blocks README examples compiled"
[ "$failures" -eq 0 ]
