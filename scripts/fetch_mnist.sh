#!/usr/bin/env bash
# Downloads the four MNIST files into data/mnist (or the directory given) and
# checks them against their published MD5 sums. Files already present and
# intact are kept.
#
#     scripts/fetch_mnist.sh [directory]
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dir=${1:-"$repo_root/data/mnist"}
# yann.lecun.com, the original host, is often unavailable; this is the mirror
# TensorFlow's datasets use.
base_url=https://storage.googleapis.com/cvdf-datasets/mnist

files=(
	"train-images-idx3-ubyte.gz f68b3c2dcbeaaa9fbdd348bbdeb94873"
	"train-labels-idx1-ubyte.gz d53e105ee54ea40749a09fcbcd1e9432"
	"t10k-images-idx3-ubyte.gz 9fb629c4189551a2d022fa330f9573f3"
	"t10k-labels-idx1-ubyte.gz ec29112dd5afa0611ce80d1b7f02629c"
)

mkdir -p "$dir"
for entry in "${files[@]}"; do
	read -r name md5 <<< "$entry"
	path="$dir/$name"
	if [ -f "$path" ] && echo "$md5  $path" | md5sum --check --status; then
		echo "have $name"
		continue
	fi
	echo "fetching $name"
	curl --fail --silent --show-error --location --output "$path.part" "$base_url/$name"
	if ! echo "$md5  $path.part" | md5sum --check --status; then
		rm -f "$path.part"
		echo "checksum mismatch for $name" >&2
		exit 1
	fi
	mv "$path.part" "$path"
done
echo "MNIST is in $dir"
