import math

from realgazebo.fire import (
    FireModel, RAY_COUNT, MIN_RADIUS, WIND_SPREAD, BASE_FLAME)
from realgazebo.protocol import pack_fire_state, MAX_FIRE_STATE_BYTES


def _radius(vertex, center):
    return math.hypot(vertex[0] - center[0], vertex[1] - center[1])


def test_ignite_starts_a_ring_of_the_given_radius():
    model = FireModel()
    assert model.ignite(3, (10.0, 5.0, 1.0), 4.0, now=0.0)
    (part,) = model.polygons(3)
    assert len(part) == RAY_COUNT
    for vertex in part:
        assert math.isclose(_radius(vertex, (10.0, 5.0)), 4.0)
        assert vertex[2] == 1.0
        assert vertex[3] == BASE_FLAME


def test_zero_radius_still_has_a_ring():
    model = FireModel()
    model.ignite(0, (0.0, 0.0, 0.0), 0.0, now=0.0)
    assert math.isclose(_radius(model.polygons(0)[0][0], (0, 0)), MIN_RADIUS)


def test_ignite_on_a_live_id_is_ignored():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=0.0)
    assert not model.ignite(1, (50.0, 50.0, 0.0), 9.0, now=1.0)
    assert math.isclose(_radius(model.polygons(1)[0][0], (0, 0)), 2.0)


def test_extinguish():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=0.0)
    assert model.extinguish(1)
    assert model.ids() == []
    assert not model.extinguish(1)


def test_no_wind_no_spread():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=0.0)
    model.step(10.0, (0.0, 0.0))
    for vertex in model.polygons(1)[0]:
        assert math.isclose(_radius(vertex, (0, 0)), 2.0)
        assert vertex[3] == BASE_FLAME


def test_wind_stretches_the_fire_downwind():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=0.0)
    model.step(10.0, (4.0, 0.0))  # 4 m/s toward +x
    part = model.polygons(1)[0]
    head = part[0]                  # ray 0 points along +x
    back = part[RAY_COUNT // 2]     # ray pointing along -x
    assert math.isclose(head[0], 2.0 + WIND_SPREAD * 4.0 * 10.0)
    assert math.isclose(_radius(back, (0, 0)), 2.0)  # upwind side stays put
    assert head[3] > back[3]        # taller flames at the head


def test_time_going_backwards_moves_nothing():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=100.0)
    model.step(5.0, (3.0, 0.0))     # sim reset
    assert math.isclose(_radius(model.polygons(1)[0][0], (0, 0)), 2.0)


def test_state_packet_fits_one_datagram():
    model = FireModel()
    model.ignite(1, (0.0, 0.0, 0.0), 2.0, now=0.0)
    assert len(pack_fire_state(1, model.polygons(1))) <= MAX_FIRE_STATE_BYTES
