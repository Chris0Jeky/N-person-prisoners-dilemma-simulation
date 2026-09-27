"""Check Markdown links for dead file/dir/anchor targets (W7 docs gate).

Scope: local links only (``http(s):``, ``mailto:`` and other schemes are
skipped). For each ``[text](target)`` outside fenced code blocks and inline
code spans, the target's file part must exist relative to the linking file
(or to ``--root`` for ``/``-rooted targets), and any ``#anchor`` part must
match a GitHub-style heading slug in the target file.

Usage:
    PYTHONPATH=. python scripts/analysis/check_doc_links.py [PATH ...]

With no PATH arguments, checks the W7 scope: root ``*.md``, ``docs/`` and
``archive/MANIFEST.md``. Exits 0 when clean, 1 listing each dead link.
"""

import os
import re
import sys

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
SLUG_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)

DEFAULT_SCOPE = ("README.md", "docs", "archive/MANIFEST.md")


def repo_root():
    """Return the repo root (this script lives in ``scripts/analysis/``)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def slugify(heading):
    """Return the GitHub-style anchor slug for a heading's text."""
    slug = SLUG_STRIP_RE.sub("", heading.strip().lower())
    return re.sub(r"\s", "-", slug)  # one hyphen per space, like GitHub


def iter_markdown_files(paths):
    """Yield ``*.md`` files for the given files/dirs (sorted, deduped)."""
    seen = set()
    for path in paths:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames.sort()
                for filename in sorted(filenames):
                    if filename.lower().endswith(".md"):
                        full = os.path.join(dirpath, filename)
                        if full not in seen:
                            seen.add(full)
                            yield full
        elif path.lower().endswith(".md") and os.path.isfile(path):
            if path not in seen:
                seen.add(path)
                yield path


def collect_anchors(lines):
    """Return the set of anchor slugs defined by Markdown headings."""
    anchors = set()
    counts = {}
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slugify(match.group(2))
        n = counts.get(base, 0)
        counts[base] = n + 1
        anchors.add(base if n == 0 else "%s-%d" % (base, n))
    return anchors


def iter_links(lines):
    """Yield ``(line_number, target)`` for links outside code spans."""
    in_fence = False
    for lineno, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_RE.finditer(INLINE_CODE_RE.sub("", line)):
            yield lineno, match.group(1)


def check_file(path, root):
    """Return a list of ``(lineno, target, reason)`` dead links in a file."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    own_anchors = collect_anchors(lines)
    anchor_cache = {}
    dead = []
    for lineno, target in iter_links(lines):
        if SCHEME_RE.match(target):
            continue  # external link: out of scope
        file_part, sep, anchor = target.partition("#")
        if file_part:
            if file_part.startswith("/"):
                resolved = os.path.normpath(os.path.join(root, file_part[1:]))
            else:
                resolved = os.path.normpath(
                    os.path.join(os.path.dirname(path), file_part)
                )
            if not os.path.exists(resolved):
                dead.append((lineno, target, "missing file or directory"))
                continue
            if anchor:
                if resolved not in anchor_cache:
                    if os.path.isdir(resolved):
                        anchor_cache[resolved] = set()
                    else:
                        with open(resolved, encoding="utf-8") as f:
                            anchor_cache[resolved] = collect_anchors(
                                f.read().splitlines()
                            )
                if anchor not in anchor_cache[resolved]:
                    dead.append((lineno, target, "missing anchor"))
        elif sep and anchor not in own_anchors:
            dead.append((lineno, target, "missing anchor"))
    return dead


def default_scope(root):
    """Return the default W7 check scope: root ``*.md`` + docs + MANIFEST."""
    scoped = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.lower().endswith(".md")
    ]
    for rel in DEFAULT_SCOPE[1:]:
        scoped.append(os.path.join(root, rel))
    return scoped


def check_paths(paths, root):
    """Return ``{file: [(lineno, target, reason)]}`` for dead links found."""
    results = {}
    for path in iter_markdown_files(paths):
        dead = check_file(path, root)
        if dead:
            results[os.path.relpath(path, root)] = dead
    return results


def main(argv):
    root = repo_root()
    paths = [os.path.join(root, p) if not os.path.isabs(p) else p for p in argv[1:]]
    results = check_paths(paths or default_scope(root), root)
    for rel_path in sorted(results):
        for lineno, target, reason in results[rel_path]:
            print("%s:%d: dead link (%s): %s" % (rel_path, lineno, reason, target))
    if results:
        total = sum(len(v) for v in results.values())
        print("FAIL: %d dead link(s) in %d file(s)" % (total, len(results)))
        return 1
    print("OK: no dead links")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
