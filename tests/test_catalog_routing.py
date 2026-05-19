"""Unit tests for catalog-first routing and group-by extraction."""

from __future__ import annotations

from services.catalog_routing_service import (
    CatalogDataset,
    CatalogRoutingService,
    extract_group_by_before_by,
)
from services.routing_service import RoutingService, _score_target_with_scores


def test_extract_group_by_before_by_plural():
    cols = ["emirate", "period", "value"]
    assert extract_group_by_before_by(
        "top 5 emirates by total revenue", cols
    ) == "emirate"


def test_extract_group_by_before_by_exact():
    cols = ["region", "total_revenue"]
    assert extract_group_by_before_by(
        "top 10 region by total revenue in 2024", cols
    ) == "region"


def test_catalog_score_prefers_display_name():
    svc = CatalogRoutingService(mongo=None)  # type: ignore[arg-type]
    entries = [
        CatalogDataset(
            slug="hajj-package-service",
            dataset_name="Hajj Package Service",
            facts_collection="awqaf_hajj_package_service_facts",
            purpose="Package bookings and transactions",
            metric_phrases=("total_transactions",),
        ),
        CatalogDataset(
            slug="zakat-payment",
            dataset_name="Zakat Payment",
            facts_collection="awqaf_zakat_payment_facts",
            purpose="Zakat collection channels",
            metric_phrases=("funds_collected_aed",),
        ),
    ]
    svc.load_catalog = lambda: entries  # type: ignore[method-assign]

    result = svc.score_question(
        "monthly transactions for hajj package service in 2025",
        available_collections=[
            "awqaf_hajj_package_service_facts",
            "awqaf_zakat_payment_facts",
        ],
    )
    assert result.method == "catalog"
    assert result.facts_collection == "awqaf_hajj_package_service_facts"
    assert result.slug == "hajj-package-service"
    assert result.score >= 3.0


def test_routing_guess_group_by_top_n_pattern():
    router = RoutingService()
    cols = ["emirate", "total_revenues_collected_aed", "period"]
    gb = router._guess_group_by(
        "top 5 emirates by total revenue from occupancy and revenues",
        cols,
    )
    assert gb == "emirate"


def test_score_target_with_scores_overlap():
    candidates = [
        "awqaf_hajj_package_service_facts",
        "awqaf_zakat_payment_facts",
    ]
    chosen, scores = _score_target_with_scores(
        "hajj package service transactions 2025",
        candidates,
    )
    assert chosen == "awqaf_hajj_package_service_facts"
    assert scores["awqaf_hajj_package_service_facts"] >= scores.get(
        "awqaf_zakat_payment_facts", 0
    )
