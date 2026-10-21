"""Figures, HTML report and optional AI anomaly summary."""

from ivchar.report.html import render
from ivchar.report.plots import make_all

__all__ = ["make_all", "render"]
