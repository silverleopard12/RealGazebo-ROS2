import math
import struct
from collections import namedtuple

# RealGazebo UDP wire protocol (must match the gz plugin / UE side).
# Header: entity_id(u8), type_code(u8), message_id(u8).
# type_code 0..199 = PX4 vehicles, >= 200 = static props/obstacles
# (see type_codes.PROP_CODE_MIN).
HEADER_SIZE = 3
# Pose packet, reused inbound (UE -> manager) as an UPSERT:
#  - id unknown            -> SPAWN (vehicle or prop)
#  - id active, same code  -> prop: MOVE (set_pose); vehicle: ignored
#  - id active, other code -> dropped (sender bug; ids are global)
MSG_POSE = 1
# Destroy packet, reused as the inbound DESPAWN command. The same
# code-vs-holder validation applies: an id active under another code
# is dropped, never despawned.
MSG_DESTROY = 4
# World wind (UE <-> manager). A WORLD-level message: the header ids carry
# nothing and are ignored (the sender puts 0, 0). Payload: op(u8) +
# 3 little-endian float32 = wind linear velocity in the Gazebo world frame
# (m/s, right-handed, no UE-frame conversion) - 16 bytes in total.
#   UE -> manager: op 0 = wind OFF (vector ignored, parsed as zeros),
#                  1 = wind ON, 2 = QUERY the current state (vector ignored).
#   manager -> UE: the answer to a QUERY, same layout with op 0/1 = the
#                  current enable flag + vector, sent to the asker's IP on
#                  the UE port (unreal_port).
MSG_WIND = 6
WIND_OP_OFF = 0
WIND_OP_ON = 1
WIND_OP_QUERY = 2
# Fire (UE <-> manager). WORLD-level like wind: header ids are 0, 0 and
# ignored; the fire is addressed by fire_id (u8, 0-255, one range shared by
# UE and the manager like vehicle ids). Coordinates are the Gazebo world
# frame (meters, right-handed), little-endian float32.
#   MSG_FIRE_IGNITE (both ways, 20 B): fire_id + x, y, z, radius.
#       UE -> manager: start a fire; ignored if fire_id is already alive.
#       The fire's first MSG_FIRE_STATE is the confirmation.
#   MSG_FIRE_STATE (manager -> UE): fire_id, part_count(u8), then per part
#       vertex_count(u16) + vertex_count * (x, y, z, flame_len). Polygons in
#       boundary order, either winding, no repeated first vertex. One
#       datagram per fire, at most MAX_FIRE_STATE_BYTES.
#   MSG_FIRE_REMOVE (both ways, 4 B): fire_id.
#       UE -> manager: extinguish. manager -> UE: the fire is gone (the
#       answer to an extinguish, also sent for an id that was not alive).
MSG_FIRE_IGNITE = 7
MSG_FIRE_STATE = 8
MSG_FIRE_REMOVE = 9
# Keep a state packet inside one Ethernet MTU so it never fragments:
# 5 B + 2 B + 16 B * 87 = 1399 B for a single-part fire.
MAX_FIRE_STATE_BYTES = 1400

# Pose payload: 7 little-endian float32 = pos(x, y, z) + quat(x, y, z, w),
# expressed in the Gazebo frame (meters, right-handed).
_POSE = struct.Struct('<7f')
# Wind payload: enable(u8) + velocity(x, y, z) as float32.
_WIND = struct.Struct('<B3f')
# Fire ignite payload: fire_id(u8) + position(x, y, z) + radius, float32.
_FIRE_IGNITE = struct.Struct('<B4f')
_FIRE_VERTEX = struct.Struct('<4f')

SpawnCommand = namedtuple(
    'SpawnCommand', ['entity_id', 'type_code', 'position', 'quaternion'])
DespawnCommand = namedtuple('DespawnCommand', ['entity_id', 'type_code'])
# Deliberately have NO entity_id/type_code: type_code 0 is a real vehicle
# (x500), so a wind message carrying its header bytes would be mistaken
# for x500_0 by any entity-keyed path. Dispatch on the class instead.
WindCommand = namedtuple('WindCommand', ['enable', 'velocity'])
# "Tell me the current world wind" - answered with a MSG_WIND packet.
WindQuery = namedtuple('WindQuery', [])
# Fire commands carry fire_id, never entity_id/type_code, for the same reason.
FireIgnite = namedtuple('FireIgnite', ['fire_id', 'position', 'radius'])
FireExtinguish = namedtuple('FireExtinguish', ['fire_id'])


def parse_packet(data: bytes):
    """Parse a wire packet into a command.

    Returns a SpawnCommand (MessageID=1), a DespawnCommand (MessageID=4), a
    WindCommand or WindQuery (MessageID=6), a FireIgnite (MessageID=7), a
    FireExtinguish (MessageID=9), or None for other message ids.
    Raises ValueError on a malformed packet.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError("packet shorter than header")
    entity_id, type_code, message_id = data[0], data[1], data[2]
    if message_id == MSG_POSE:
        payload = data[HEADER_SIZE:]
        if len(payload) < _POSE.size:
            raise ValueError("MessageID=1 payload too short for pose")
        x, y, z, qx, qy, qz, qw = _POSE.unpack(payload[:_POSE.size])
        return SpawnCommand(entity_id, type_code, (x, y, z), (qx, qy, qz, qw))
    if message_id == MSG_DESTROY:
        return DespawnCommand(entity_id, type_code)
    if message_id == MSG_WIND:
        payload = data[HEADER_SIZE:]
        if len(payload) < _WIND.size:
            raise ValueError("MessageID=6 payload too short for wind")
        op, vx, vy, vz = _WIND.unpack(payload[:_WIND.size])
        if op == WIND_OP_QUERY:
            return WindQuery()
        if op == WIND_OP_OFF:
            # The vector means nothing when the wind is off: normalise it,
            # so the state reported back is unambiguous and a garbage
            # vector can never stop the OFF command from landing.
            return WindCommand(False, (0.0, 0.0, 0.0))
        if op != WIND_OP_ON:
            raise ValueError(f"MessageID=6 unknown wind op {op}")
        # A NaN/inf wind would poison every enable_wind link in the world
        # (and UE text boxes are free-form), so refuse it here.
        if not all(math.isfinite(v) for v in (vx, vy, vz)):
            raise ValueError("MessageID=6 wind velocity is not finite")
        return WindCommand(True, (vx, vy, vz))
    if message_id == MSG_FIRE_IGNITE:
        payload = data[HEADER_SIZE:]
        if len(payload) < _FIRE_IGNITE.size:
            raise ValueError("MessageID=7 payload too short for fire ignite")
        fire_id, x, y, z, radius = _FIRE_IGNITE.unpack(
            payload[:_FIRE_IGNITE.size])
        if not all(math.isfinite(v) for v in (x, y, z, radius)) or radius < 0:
            raise ValueError("MessageID=7 position/radius not finite or negative")
        return FireIgnite(fire_id, (x, y, z), radius)
    if message_id == MSG_FIRE_REMOVE:
        if len(data) < HEADER_SIZE + 1:
            raise ValueError("MessageID=9 payload too short for fire_id")
        return FireExtinguish(data[HEADER_SIZE])
    return None


def pack_pose(entity_id, type_code, position, quaternion) -> bytes:
    """Build a MessageID=1 pose packet (used by tests and golden vectors)."""
    return bytes([entity_id, type_code, MSG_POSE]) + _POSE.pack(
        *position, *quaternion)


def pack_destroy(entity_id, type_code) -> bytes:
    """Build a MessageID=4 destroy packet (used by tests)."""
    return bytes([entity_id, type_code, MSG_DESTROY])


def pack_wind(enable, velocity) -> bytes:
    """Build a MessageID=6 wind packet: the UE->manager ON/OFF command and,
    with the same layout, the manager->UE state answer."""
    return bytes([0, 0, MSG_WIND]) + _WIND.pack(
        WIND_OP_ON if enable else WIND_OP_OFF, *velocity)


def pack_wind_query() -> bytes:
    """Build a MessageID=6 QUERY packet (UE -> manager, vector unused)."""
    return bytes([0, 0, MSG_WIND]) + _WIND.pack(WIND_OP_QUERY, 0.0, 0.0, 0.0)


def pack_fire_ignite(fire_id, position, radius) -> bytes:
    """Build a MessageID=7 ignite packet (same layout both ways)."""
    return bytes([0, 0, MSG_FIRE_IGNITE]) + _FIRE_IGNITE.pack(
        fire_id, *position, radius)


def pack_fire_state(fire_id, polygons) -> bytes:
    """Build a MessageID=8 state packet.

    polygons: list of parts, each a list of (x, y, z, flame_len) vertices.
    Raises ValueError when the packet would exceed MAX_FIRE_STATE_BYTES -
    the caller must simplify the fire line first.
    """
    out = bytearray([0, 0, MSG_FIRE_STATE, fire_id, len(polygons)])
    for part in polygons:
        out += struct.pack('<H', len(part))
        for vertex in part:
            out += _FIRE_VERTEX.pack(*vertex)
    if len(out) > MAX_FIRE_STATE_BYTES:
        raise ValueError(
            f"fire state packet {len(out)} B exceeds {MAX_FIRE_STATE_BYTES} B")
    return bytes(out)


def pack_fire_remove(fire_id) -> bytes:
    """Build a MessageID=9 extinguish/removed packet (same layout both ways)."""
    return bytes([0, 0, MSG_FIRE_REMOVE, fire_id])


def triage_batch(items):
    """Classify one receive batch of (command, sender_ip) pairs.

    Returns (wind, askers, fires, rest):
      wind   - the NEWEST WindCommand, or None. A world-level setting is
               idempotent, so earlier ones in the same batch are superseded.
      askers - sender IPs that sent a WindQuery, deduplicated, arrival order.
      fires  - (FireIgnite | FireExtinguish, sender_ip) in arrival order;
               not collapsed, an ignite then an extinguish both matter.
      rest   - the entity commands in arrival order, for the entity
               supersede/dispatch logic, which keys on entity_id/type_code
               that the world-level messages deliberately do not have.
    None commands (packets the manager does not act on) are skipped.
    """
    wind = None
    askers = []
    fires = []
    rest = []
    for cmd, sender_ip in items:
        if isinstance(cmd, WindCommand):
            wind = cmd
        elif isinstance(cmd, (FireIgnite, FireExtinguish)):
            fires.append((cmd, sender_ip))
        elif isinstance(cmd, WindQuery):
            if sender_ip not in askers:
                askers.append(sender_ip)
        elif cmd is not None:
            rest.append(cmd)
    return wind, askers, fires, rest
