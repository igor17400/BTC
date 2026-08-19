"""Tag every NewsReX model repo on the Hugging Face Hub with a release name.

Hub model repos are git repos, so a tag pins a revision: new weights can keep
landing on ``main`` without moving the numbers a paper points at. Consumers
load a pinned revision with ``revision="<tag>"``.

Dry run (default) lists what would be tagged:

    python scripts/tag_hf_release.py --tag v1.0.0-cikm26

Apply (needs ``huggingface-cli login`` with write access to the org):

    python scripts/tag_hf_release.py --tag v1.0.0-cikm26 --apply
"""

import argparse
import contextlib
import sys

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Tag name, e.g. v1.0.0-cikm26")
    parser.add_argument("--org", default="newsrex", help="Hub org (default: newsrex)")
    parser.add_argument(
        "--message", default=None, help="Tag message (default: derived from --tag)"
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="REPO",
        help="Repo names to leave untagged, e.g. NAML-JAX-MIND-small-glove-random "
        "(org prefix optional). Use for repos whose weights are known stale.",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="Move the tag if it already exists (delete then recreate). Without "
        "this, an existing tag is left pointing at its original revision.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually create tags (default: dry run)"
    )
    args = parser.parse_args()

    api = HfApi()
    repos = sorted(m.id for m in api.list_models(author=args.org))
    if not repos:
        print(f"No model repos found under org '{args.org}'.")
        return 1

    skip = {s.split("/")[-1] for s in args.skip}
    skipped = [r for r in repos if r.split("/")[-1] in skip]
    repos = [r for r in repos if r.split("/")[-1] not in skip]
    for r in skipped:
        print(f"  skipping   {r}")
    unknown = skip - {r.split("/")[-1] for r in skipped}
    if unknown:
        print(f"  WARNING: --skip named repos that do not exist: {sorted(unknown)}")

    message = args.message or f"NewsReX release {args.tag}"
    print(f"{len(repos)} repo(s) under '{args.org}', tag '{args.tag}'")
    if not args.apply:
        print("DRY RUN — pass --apply to create tags\n")

    failed = []
    for repo_id in repos:
        if not args.apply:
            print(f"  would tag  {repo_id}")
            continue
        try:
            if args.retag:
                # The tag may not exist yet; deleting is best-effort.
                with contextlib.suppress(Exception):
                    api.delete_tag(repo_id, tag=args.tag, repo_type="model")
            api.create_tag(
                repo_id,
                tag=args.tag,
                repo_type="model",
                tag_message=message,
                exist_ok=not args.retag,
            )
            print(f"  {'retagged  ' if args.retag else 'tagged    '} {repo_id}")
        except Exception as exc:  # network / auth / permission
            print(f"  FAILED     {repo_id}: {exc}")
            failed.append(repo_id)

    if failed:
        print(f"\n{len(failed)} repo(s) failed:", ", ".join(failed))
        return 1
    if args.apply:
        print(f"\nAll {len(repos)} repos tagged '{args.tag}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
