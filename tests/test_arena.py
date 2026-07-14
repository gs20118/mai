"""The arena config is the coordinate system everything else trusts.

A silently wrong zone map is the worst failure mode available to us: it produces
confident, well-formed, wrong answers that look exactly like a working system and
score zero. So these tests are deliberately picky.
"""

from __future__ import annotations

import pytest

from mai.arena import Arena


@pytest.fixture(scope="module")
def arena() -> Arena:
    return Arena.from_yaml()


def test_loads_and_validates(arena: Arena):
    assert arena.validate() == []
    assert len(arena.zones) == 26


def test_zones_tile_the_arena_exactly(arena: Arena):
    """No gaps (a crater in a gap gets no zone) and no overlaps (it gets an arbitrary one)."""
    covered = sum(zone.w * zone.h for zone in arena.zones)
    assert covered == pytest.approx(arena.width_cm * arena.height_cm)


def test_each_band_spans_the_full_width(arena: Arena):
    for band in ("taxiway_a", "runway", "taxiway_b"):
        width = sum(zone.w for zone in arena.zones_in_band(band))
        assert width == pytest.approx(arena.width_cm), band

    # The two facility rows each span the width: 160 + 180 + 160 = 500.
    facility_rows: dict[float, float] = {}
    for zone in arena.facility_zones:
        facility_rows[zone.y] = facility_rows.get(zone.y, 0.0) + zone.w
    assert sorted(facility_rows) == [0.0, 320.0]
    assert all(width == pytest.approx(500.0) for width in facility_rows.values())


def test_band_order_is_fa_twa_rw_twb_fa(arena: Arena):
    bands_by_depth = sorted({(zone.y, zone.band) for zone in arena.zones})
    assert [band for _, band in bands_by_depth] == [
        "facility",
        "taxiway_a",
        "runway",
        "taxiway_b",
        "facility",
    ]


def test_runway_scales_to_3000m(arena: Arena):
    """TASK.md 13.3: the runway is 3000m real, in ten 300m zones."""
    runway = arena.runway_zones
    assert len(runway) == 10
    for zone in runway:
        assert zone.w * arena.scale / 100.0 == pytest.approx(300.0)


def test_zone_ids_match_the_dashboard_format(arena: Arena):
    """TASK.md 12.4 pads RW and FA to two digits but does not pad TW."""
    ids = {zone.id for zone in arena.zones}
    assert "RW-04" in ids and "RW-10" in ids
    assert "TW-A2" in ids and "TW-B5" in ids
    assert "FA-02" in ids
    assert "RW-4" not in ids and "TW-A02" not in ids


def test_zone_at_boundaries(arena: Arena):
    # Runway band starts at y=160; RW-04 spans x 150..200.
    assert arena.zone_at(175.0, 200.0) == "RW-04"
    # A shared edge belongs to exactly one zone, never both.
    assert arena.zone_at(200.0, 200.0) == "RW-05"
    assert arena.zone_at(150.0, 160.0) == "RW-04"
    # The far arena edge still resolves rather than falling through to None.
    assert arena.zone_at(500.0, 400.0) == "FA-06"
    assert arena.zone_at(0.0, 0.0) == "FA-01"


def test_zone_at_rejects_points_outside(arena: Arena):
    assert arena.zone_at(-1.0, 200.0) is None
    assert arena.zone_at(250.0, 401.0) is None


def test_scoring_groups(arena: Arena):
    """Craters and UXO score in RW+TW only; the counts score in RW only."""
    assert len(arena.runway_zones) == 10
    assert len(arena.taxiway_zones) == 10
    assert len(arena.airfield_zones) == 20
    assert len(arena.facility_zones) == 6


def test_marker_corners_are_square_and_centred(arena: Arena):
    for marker_id, marker in arena.markers.items():
        corners = arena.marker_world_corners(marker_id)
        assert corners.shape == (4, 2)
        centre = corners.mean(axis=0)
        assert centre == pytest.approx(marker.center)
        for index in range(4):
            edge = corners[(index + 1) % 4] - corners[index]
            assert float(abs(edge[0]) + abs(edge[1])) == pytest.approx(
                arena.marker_size_cm, abs=1e-6
            )
