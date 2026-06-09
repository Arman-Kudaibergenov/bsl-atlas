"""Tests for the eval harness: it must build the sample index, run the tagged
question set green, and produce a grep baseline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from eval import runner


@pytest.fixture
def report(tmp_path):
    questions = json.loads(runner.DEFAULT_QUESTIONS.read_text(encoding="utf-8"))
    store = runner.build_store(runner.SAMPLE_DIR, tmp_path / "eval.db")
    return runner.run_questions(store, questions, source=runner.SAMPLE_DIR)


class TestEvalHarness:
    def test_all_questions_pass(self, report):
        assert report["passed"] == report["total"], [
            r for r in report["results"] if not r["passed"]
        ]
        assert report["pass_rate"] == 1.0

    def test_has_questions(self, report):
        assert report["total"] >= 15

    def test_by_tag_populated(self, report):
        assert "reverse-call" in report["by_tag"]
        assert "triggers-on-write" in report["by_tag"]

    def test_grep_baseline_present(self, report):
        # at least one question records grep hits for the friction comparison
        assert any("grep_hits" in r for r in report["results"])

    def test_grep_baseline_counts(self):
        # ЗначениеРеквизита appears multiple times textually
        hits = runner.grep_baseline(runner.SAMPLE_DIR, "ЗначениеРеквизита")
        assert hits >= 3

    def test_format_report_runs(self, report):
        text = runner.format_report(report)
        assert "eval:" in text
        assert "By tag:" in text
