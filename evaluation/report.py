"""Report generation — delegates to table, chart, and conclusion modules."""

import os
from typing import Dict

from config import RESULTS_DIR
from detectors.base_detector import VideoResult
from .table import build_comparison_table
from .chart import save_chart
from .conclusion import generate_conclusion


def generate_report(all_results: Dict[str, Dict[str, VideoResult]]) -> None:
    """Generate all report artefacts.

    Args:
        all_results: Nested dict  model_name -> test_file_name -> VideoResult
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    table_df = build_comparison_table(all_results)
    save_chart(all_results)
    generate_conclusion(all_results, table_df)
