"""Download and extract the cache tarball from the GitHub Release `cache-latest`.

The repo no longer tracks `embeddings_cache/` or `explainxkcd_cache/` in git;
they are distributed as a single tarball asset on a fixed release tag so the
repository history does not grow with each daily refresh.

Usage:
    python bootstrap_cache.py            # download if cache is missing
    python bootstrap_cache.py --force    # always re-download
    python bootstrap_cache.py --check    # exit 0 if cache present, 1 otherwise

Environment variables:
    XKCD_CACHE_REPO   GitHub "owner/repo" hosting the release (default: JBlitzar/xkcd-rag)
    XKCD_CACHE_TAG    Release tag (default: cache-latest)
    XKCD_CACHE_ASSET  Asset filename (default: cache.tar.gz)
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error

DEFAULT_REPO = "JBlitzar/xkcd-rag"
DEFAULT_TAG = "cache-latest"
DEFAULT_ASSET = "cache.tar.gz"

EMBEDDINGS_DIR = "embeddings_cache"
EXPLAINXKCD_DIR = "explainxkcd_cache"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "index.json")


def cache_present(root: str) -> bool:
    return (
        os.path.isfile(os.path.join(root, EMBEDDINGS_FILE))
        and os.path.isfile(os.path.join(root, INDEX_FILE))
        and os.path.isdir(os.path.join(root, EXPLAINXKCD_DIR))
    )


def asset_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "xkcd-rag-bootstrap"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _is_within(base: str, target: str) -> bool:
    base_abs = os.path.realpath(base)
    target_abs = os.path.realpath(target)
    return target_abs == base_abs or target_abs.startswith(base_abs + os.sep)


def safe_extract(tar: tarfile.TarFile, root: str) -> None:
    allowed_prefixes = (EMBEDDINGS_DIR + "/", EXPLAINXKCD_DIR + "/")
    allowed_exact = (EMBEDDINGS_DIR, EXPLAINXKCD_DIR)
    for member in tar.getmembers():
        name = member.name.lstrip("./")
        if not (name in allowed_exact or name.startswith(allowed_prefixes)):
            raise RuntimeError(f"Refusing to extract unexpected entry: {member.name}")
        dest_path = os.path.join(root, name)
        if not _is_within(root, dest_path):
            raise RuntimeError(f"Refusing path traversal entry: {member.name}")
    tar.extractall(root)


def bootstrap(
    root: str = ".",
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
    asset: str = DEFAULT_ASSET,
    force: bool = False,
) -> bool:
    if not force and cache_present(root):
        print(f"Cache already present in {root}; skipping download.")
        return True

    url = asset_url(repo, tag, asset)
    print(f"Downloading cache from {url} ...")
    with tempfile.NamedTemporaryFile(prefix="xkcd-cache-", suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        try:
            download(url, tmp_path)
        except urllib.error.HTTPError as e:
            print(f"Download failed: HTTP {e.code} {e.reason}", file=sys.stderr)
            return False
        except urllib.error.URLError as e:
            print(f"Download failed: {e.reason}", file=sys.stderr)
            return False

        print("Extracting cache ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            safe_extract(tar, root)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not cache_present(root):
        print("Extracted archive but expected cache files are missing.", file=sys.stderr)
        return False
    print("Cache ready.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root to extract into.")
    parser.add_argument("--repo", default=os.environ.get("XKCD_CACHE_REPO", DEFAULT_REPO))
    parser.add_argument("--tag", default=os.environ.get("XKCD_CACHE_TAG", DEFAULT_TAG))
    parser.add_argument("--asset", default=os.environ.get("XKCD_CACHE_ASSET", DEFAULT_ASSET))
    parser.add_argument("--force", action="store_true", help="Re-download even if cache is present.")
    parser.add_argument("--check", action="store_true", help="Exit 0 if cache present, 1 otherwise.")
    args = parser.parse_args()

    if args.check:
        return 0 if cache_present(args.root) else 1

    ok = bootstrap(
        root=args.root,
        repo=args.repo,
        tag=args.tag,
        asset=args.asset,
        force=args.force,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
