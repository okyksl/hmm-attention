import pytest

from src.probe_offsets import normalize_offsets_by_level, resolve_probe_offsets


def test_auto_offsets_share_surface_horizons_across_levels():
    assert resolve_probe_offsets([6, 3, 1], surface_burn_in=12) == [
        [-2, -1, 0, 1],
        [-4, -3, -2, -1, 0, 1, 2],
        list(range(-12, 7)),
    ]


def test_explicit_flat_offsets_broadcast_and_nested_offsets_stay_per_level():
    assert resolve_probe_offsets([6, 3, 1], 12, configured=[-1, 0]) == [
        [-1, 0], [-1, 0], [-1, 0],
    ]
    nested = [[-2, -1, 0, 1], [-3, 0, 2], [0, 1, 4]]
    assert resolve_probe_offsets([6, 3, 1], 12, configured=nested) == nested


def test_legacy_mode_preserves_common_base_context_offsets():
    assert resolve_probe_offsets([6, 3, 1], 12, mode="legacy") == [
        [-2, -1, 0, 1],
        [-2, -1, 0, 1],
        [-2, -1, 0, 1],
    ]


def test_per_level_offsets_validate_shape_and_duplicates():
    with pytest.raises(ValueError, match="expected 3"):
        normalize_offsets_by_level([[-1, 0], [0]], 3)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_offsets_by_level([[-1, -1], [0], [0]], 3)
