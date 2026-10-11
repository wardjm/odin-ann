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
if ! { [ -f "$archive" ] && echo "$md5  $archive" | md5sum --check --status; }; then
	echo "fetching cifar-10-binary.tar.gz"
	curl --fail --silent --show-error --location --output "$archive.part" "$url"
	if ! echo "$md5  $archive.part" | md5sum --check --status; then
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
