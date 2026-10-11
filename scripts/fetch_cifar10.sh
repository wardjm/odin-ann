#!/usr/bin/env bash
# Downloads the binary version of CIFAR-10, about 162 MB, checks it against
# its published MD5 sum, and extracts its five training batches and test batch
# into data/cifar10 (or the directory given). Files already present are kept.
#
#     scripts/fetch_cifar10.sh [directory]
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dir=${1:-"$repo_root/data/cifar10"}
url=https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
md5=c32a1d4ab5d03f1284b67883e8d87530
files=(data_batch_1.bin data_batch_2.bin data_batch_3.bin data_batch_4.bin data_batch_5.bin test_batch.bin batches.meta.txt)

# md5_matches FILE SUM succeeds if FILE exists and has MD5 sum SUM. It uses
# md5sum where present, and macOS's md5 otherwise.
md5_matches() {
	[ -f "$1" ] || return 1
	if command -v md5sum > /dev/null; then
		echo "$2  $1" | md5sum --check --status
	else
		[ "$(md5 -q "$1")" = "$2" ]
	fi
}

mkdir -p "$dir"
missing=0
for name in "${files[@]}"; do
	[ -f "$dir/$name" ] || missing=1
done
if [ "$missing" = 0 ]; then
	echo "CIFAR-10 is in $dir"
	exit 0
fi

archive="$dir/cifar-10-binary.tar.gz"
if ! md5_matches "$archive" "$md5"; then
	echo "fetching cifar-10-binary.tar.gz"
	curl --fail --silent --show-error --location --output "$archive.part" "$url"
	if ! md5_matches "$archive.part" "$md5"; then
		rm -f "$archive.part"
		echo "checksum mismatch for cifar-10-binary.tar.gz" >&2
		exit 1
	fi
	mv "$archive.part" "$archive"
fi

# The archive holds cifar-10-batches-bin/; only the expected files leave it.
extract=$(mktemp -d "$dir/extract.XXXXXX")
trap 'rm -rf "$extract"' EXIT
tar -xzf "$archive" -C "$extract"
for name in "${files[@]}"; do
	mv "$extract/cifar-10-batches-bin/$name" "$dir/$name"
done
rm -f "$archive"
echo "CIFAR-10 is in $dir"
