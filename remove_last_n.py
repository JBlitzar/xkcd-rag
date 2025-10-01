import os
import sys
import json
import numpy as np


def find_last_n_comics(n: int) -> list[int]:
    cache_dir = "explainxkcd_cache"
    nums = []
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".json"):
                name = f[:-5]
                if name.isdigit():
                    nums.append(int(name))
    if not nums:
        idx_path = os.path.join("embeddings_cache", "index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            nums = [int(k) for k in idx.keys() if k.isdigit()]
    nums.sort()
    return nums[-n:] if nums else []


def remove_from_explanations(targets: list[int]) -> int:
    removed = 0
    cache_dir = "explainxkcd_cache"
    if not os.path.isdir(cache_dir):
        return 0
    for num in targets:
        p = os.path.join(cache_dir, f"{num}.json")
        if os.path.exists(p):
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass
    return removed


def remove_from_embeddings(targets: list[int]) -> int:
    e_dir = "embeddings_cache"
    idx_path = os.path.join(e_dir, "index.json")
    arr_path = os.path.join(e_dir, "embeddings.npy")
    if not (os.path.exists(idx_path) and os.path.exists(arr_path)):
        return 0

    with open(idx_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    arr = np.load(arr_path, allow_pickle=False)

    targets_set = {str(n) for n in targets}
    keep_items = [(k, r) for k, r in index.items() if k not in targets_set]
    if len(keep_items) == len(index):
        return 0

    keep_items.sort(key=lambda x: x[1])
    keep_rows = [r for _, r in keep_items]
    new_arr = arr[keep_rows]
    new_index = {k: i for i, (k, _) in enumerate(keep_items)}

    tmp_arr = arr_path + ".tmp"
    tmp_idx = idx_path + ".tmp"
    with open(tmp_arr, "wb") as f:
        np.save(f, new_arr)
    with open(tmp_idx, "w", encoding="utf-8") as f:
        json.dump(new_index, f, ensure_ascii=False, indent=2)
    os.replace(tmp_arr, arr_path)
    os.replace(tmp_idx, idx_path)
    return len(index) - len(new_index)


def main():
    n = 10
    if len(sys.argv) > 1:
        try:
            n = max(1, int(sys.argv[1]))
        except ValueError:
            pass

    targets = find_last_n_comics(n)
    if not targets:
        print("No comics found to remove.")
        return

    print(f"Removing last {len(targets)} comics: {targets}")
    exp_removed = remove_from_explanations(targets)
    emb_removed = remove_from_embeddings(targets)
    print(
        f"Removed {exp_removed} explanation files and {emb_removed} embeddings entries."
    )


if __name__ == "__main__":
    main()
