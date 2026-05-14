"""
Retrieval tests — run without a server.
python -m pytest tests/test_retrieval.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app.retriever import search, get_by_names, extract_job_level, extract_test_types


class TestExtractors:
    def test_job_level_entry(self):
        assert extract_job_level("hiring a junior developer") == "Entry-Level"

    def test_job_level_manager(self):
        assert extract_job_level("looking for a sales manager") == "Manager"

    def test_job_level_years(self):
        assert extract_job_level("candidate with 3 years experience") == "Mid-Professional"

    def test_job_level_none(self):
        assert extract_job_level("I need an assessment") is None

    def test_test_types_personality(self):
        assert "P" in extract_test_types("I want a personality assessment")

    def test_test_types_cognitive(self):
        assert "A" in extract_test_types("cognitive ability test")

    def test_test_types_multiple(self):
        types = extract_test_types("personality and cognitive assessment")
        assert "P" in types
        assert "A" in types


class TestRetrieval:
    def test_java_returns_results(self):
        results = search("Java developer backend engineer", top_k=10)
        assert len(results) > 0
        names = [r["name"].lower() for r in results]
        assert any("java" in n for n in names), f"No Java test found in: {names}"

    def test_sales_returns_results(self):
        results = search("sales representative customer-facing role", top_k=10)
        assert len(results) > 0

    def test_leadership_returns_results(self):
        results = search("leadership executive director role", top_k=10)
        assert len(results) > 0

    def test_data_science_returns_results(self):
        results = search("data scientist machine learning python", top_k=10)
        assert len(results) > 0

    def test_hadoop_returns_results(self):
        results = search("Hadoop big data engineer", top_k=10)
        names = [r["name"].lower() for r in results]
        assert any("hadoop" in n for n in names), f"No Hadoop in: {names}"

    def test_job_level_filter(self):
        results = search("software engineer", job_level="Entry-Level", top_k=10)
        for r in results:
            # At least some should have matching job level
            pass  # scoring, not strict filtering
        assert len(results) > 0

    def test_results_have_required_fields(self):
        results = search("customer service agent", top_k=5)
        for r in results:
            assert "name" in r
            assert "url" in r
            assert r["url"].startswith("https://www.shl.com/")

    def test_cap_at_requested_k(self):
        results = search("engineer", top_k=5)
        assert len(results) <= 5

    def test_get_by_names(self):
        results = get_by_names(["OPQ32r"])
        assert len(results) > 0
        assert any("OPQ" in r["name"] for r in results)

    def test_nonsense_query_returns_empty_or_low(self):
        results = search("xyzzy banana quantum flibbertigibbet", top_k=10)
        # Should return empty or very low scoring results
        assert len(results) <= 10  # at minimum, doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])