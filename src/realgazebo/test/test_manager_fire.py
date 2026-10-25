"""manager_node.py sends a fire lit from the ignite topic to UE at once.

Waiting for the next 2 Hz tick (FIRE_TICK_SEC) sent the first state packet 0-0.5 s
after the crash it stands for: in a 51-drone run the drones went into the road
0.5-0.8 s before the first flames. manager_node imports ROS and its sibling modules,
so those are stubs here; the fire model and the packets (fire.py, protocol.py) are real.
"""
import importlib
import pathlib
import sys
import threading
import types

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "realgazebo"


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _nothing(*a, **k):
    return None


class _Node:
    def __init__(self, *a, **k):
        pass


@pytest.fixture
def overlay():
    """manager_node.py loaded with ROS and its heavy siblings stubbed."""
    pkg = types.ModuleType("rg_manager_under_test")
    pkg.__path__ = [str(PACKAGE)]
    injected = {
        "rg_manager_under_test": pkg,
        "rclpy": _stub("rclpy", init=_nothing, shutdown=_nothing, ok=lambda: False),
        "rclpy.executors": _stub("rclpy.executors", ExternalShutdownException=Exception),
        "rclpy.node": _stub("rclpy.node", Node=_Node),
        "geometry_msgs": _stub("geometry_msgs"),
        "geometry_msgs.msg": _stub("geometry_msgs.msg", PointStamped=object),
        "rosgraph_msgs": _stub("rosgraph_msgs"),
        "rosgraph_msgs.msg": _stub("rosgraph_msgs.msg", Clock=object),
        "std_msgs": _stub("std_msgs"),
        "std_msgs.msg": _stub("std_msgs.msg", UInt8=object),
        "rg_manager_under_test.yaml_config": _stub(
            "rg_manager_under_test.yaml_config", parse_vehicles=_nothing, VehicleSpec=object),
        "rg_manager_under_test.spawn_core": _stub(
            "rg_manager_under_test.spawn_core", render_sdf=_nothing, build_create_argv=_nothing,
            build_remove_argv=_nothing, build_set_pose_argv=_nothing),
        "rg_manager_under_test.backends": _stub(
            "rg_manager_under_test.backends", make_backend=_nothing, HitlBackend=object,
            PilsBackend=object),
        "rg_manager_under_test.entity": _stub("rg_manager_under_test.entity", Entity=object),
        "rg_manager_under_test.registry": _stub("rg_manager_under_test.registry", EntityRegistry=object),
        "rg_manager_under_test.wind": _stub("rg_manager_under_test.wind", WindPublisher=object),
        "rg_manager_under_test.type_codes": _stub(
            "rg_manager_under_test.type_codes", scan_type_codes=_nothing, type_for_code=_nothing),
        "rg_manager_under_test.geometry": _stub("rg_manager_under_test.geometry", quat_to_euler=_nothing),
    }
    loaded = ("rg_manager_under_test.manager_node", "rg_manager_under_test.fire", "rg_manager_under_test.protocol")
    saved = {k: sys.modules.get(k) for k in (*injected, *loaded)}
    sys.modules.update(injected)
    try:
        yield types.SimpleNamespace(
            manager=importlib.import_module("rg_manager_under_test.manager_node"),
            fire=importlib.import_module("rg_manager_under_test.fire"),
            protocol=importlib.import_module("rg_manager_under_test.protocol"))
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class FakeSocket:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))


def _manager(overlay, sock):
    """A ManagerNode with only the fire state, without __init__ (ROS, gz)."""
    node = overlay.manager.ManagerNode.__new__(overlay.manager.ManagerNode)
    params = {"fire_default_radius": 3.0, "fire_default_duration": 0.0,
              "unreal_ip": "127.0.0.1", "unreal_port": 5005}
    node.get_parameter = lambda name: types.SimpleNamespace(value=params[name])
    log = types.SimpleNamespace(info=_nothing, warn=_nothing)
    node.get_logger = lambda: log
    node._fires = overlay.fire.FireModel()
    node._fire_lock = threading.Lock()
    node._fire_target_ip = None
    node._wind_state = (False, (0.0, 0.0, 0.0))
    node._sim_time = 266.6
    node._sock = sock
    return node


def _ignite(x, y, z):
    return types.SimpleNamespace(point=types.SimpleNamespace(x=x, y=y, z=z))


def test_topic_fire_reaches_unreal_at_once(overlay):
    sock = FakeSocket()
    node = _manager(overlay, sock)

    node._on_fire_ignite(_ignite(35.3, -109.0, 6.0))

    state = overlay.protocol.pack_fire_state(0, node._fires.polygons(0))
    assert sock.sent == [(state, ("127.0.0.1", 5005))]


def test_tick_keeps_streaming_a_topic_fire(overlay):
    sock = FakeSocket()
    node = _manager(overlay, sock)
    node._on_fire_ignite(_ignite(35.3, -109.0, 6.0))
    sock.sent.clear()

    node._fire_tick()

    state = overlay.protocol.pack_fire_state(0, node._fires.polygons(0))
    assert sock.sent == [(state, ("127.0.0.1", 5005))]


def test_topic_fire_before_the_socket_waits_for_the_tick(overlay):
    node = _manager(overlay, None)

    node._on_fire_ignite(_ignite(35.3, -109.0, 6.0))

    assert node._fires.ids() == [0]
