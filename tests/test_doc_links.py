"""W7 docs-link tests: checker unit behavior plus a repo-wide clean scan."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

import check_doc_links  # noqa: E402


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TestSlugify:
    def test_basic(self):
        assert check_doc_links.slugify("Hello World") == "hello-world"

    def test_punctuation_removed_keeps_double_hyphen(self):
        # GitHub turns "a & b" into "a--b" (one hyphen per space).
        assert (
            check_doc_links.slugify("Core Mechanisms & Theoretical Framework")
            == "core-mechanisms--theoretical-framework"
        )

    def test_underscores_kept(self):
        assert check_doc_links.slugify("a_b c") == "a_b-c"


class TestCheckFile:
    def test_missing_file_reported(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(doc, "# T\n\nSee [gone](nope.md) and [dir](sub/).\n")
        dead = check_doc_links.check_file(doc, str(tmp_path))
        assert [(t, r) for _, t, r in dead] == [
            ("nope.md", "missing file or directory"),
            ("sub/", "missing file or directory"),
        ]

    def test_existing_file_and_dir_pass(self, tmp_path):
        _write(str(tmp_path / "other.md"), "# Other\n")
        os.mkdir(str(tmp_path / "sub"))
        doc = str(tmp_path / "doc.md")
        _write(doc, "# T\n\nSee [o](other.md) and [d](sub/).\n")
        assert check_doc_links.check_file(doc, str(tmp_path)) == []

    def test_missing_anchor_reported(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(doc, "# Real Section\n\nJump to [x](#no-such-anchor).\n")
        dead = check_doc_links.check_file(doc, str(tmp_path))
        assert dead == [(3, "#no-such-anchor", "missing anchor")]

    def test_existing_anchor_passes(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(doc, "# Real Section\n\nJump to [x](#real-section).\n")
        assert check_doc_links.check_file(doc, str(tmp_path)) == []

    def test_cross_file_anchor_checked(self, tmp_path):
        _write(str(tmp_path / "other.md"), "# Real Section\n")
        doc = str(tmp_path / "doc.md")
        _write(
            doc,
            "# T\n\nGood [g](other.md#real-section), "
            "bad [b](other.md#missing).\n",
        )
        assert check_doc_links.check_file(doc, str(tmp_path)) == [
            (3, "other.md#missing", "missing anchor")
        ]

    def test_duplicate_headings_get_numbered_slugs(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(doc, "# Dup\n\n# Dup\n\n[x](#dup) [y](#dup-1) [z](#dup-2)\n")
        assert check_doc_links.check_file(doc, str(tmp_path)) == [
            (5, "#dup-2", "missing anchor")
        ]

    def test_code_blocks_and_spans_ignored(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(
            doc,
            "# T\n\n`[x](inline.md)`\n\n```\n[gone](fenced.md)\n```\n",
        )
        assert check_doc_links.check_file(doc, str(tmp_path)) == []

    def test_external_links_skipped(self, tmp_path):
        doc = str(tmp_path / "doc.md")
        _write(
            doc,
            "# T\n\n[a](https://example.com/x) "
            "[b](mailto:a@b.c) [c](#t).\n",
        )
        assert check_doc_links.check_file(doc, str(tmp_path)) == []

    def test_rooted_target_resolves_against_root(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(str(tmp_path / "top.md"), "# Top\n")
        doc = str(sub / "doc.md")
        _write(doc, "# T\n\nSee [t](/top.md) and [m](/missing.md).\n")
        assert check_doc_links.check_file(doc, str(tmp_path)) == [
            (3, "/missing.md", "missing file or directory")
        ]


class TestRepoScope:
    def test_default_scope_has_no_dead_links(self):
        results = check_doc_links.check_paths(
            check_doc_links.default_scope(ROOT), ROOT
        )
        assert results == {}
