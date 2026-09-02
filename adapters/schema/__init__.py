# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema adapter — Dataset Search ↔ Data Mining parquet contract translation."""

from adapters.schema.mapper import (
    data_mining_row_to_record,
    dataset_search_row_to_record,
    enrich_dataset_search_meta,
    record_to_data_mining_row,
    record_to_dataset_search_row,
    record_to_tmm_row,
    records_to_data_mining_frame_dicts,
    records_to_dataset_search_frame_dicts,
)

__all__ = [
    "data_mining_row_to_record",
    "dataset_search_row_to_record",
    "enrich_dataset_search_meta",
    "record_to_data_mining_row",
    "record_to_dataset_search_row",
    "record_to_tmm_row",
    "records_to_data_mining_frame_dicts",
    "records_to_dataset_search_frame_dicts",
]
