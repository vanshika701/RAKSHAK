"""
capture.py

Live packet capture and flow reconstruction for RAKSHAK's real-time
intrusion detection. Sniffs raw packets with Scapy, groups them into
network flows (a flow = one conversation between two endpoints, identified
by the 5-tuple: source IP, destination IP, source port, destination port,
protocol) - the same unit of analysis CICIDS2017 was built from, so a live
flow's reconstructed statistics line up with what the trained model
learned from.

This is layer 1: capture + flow grouping only. Feature extraction (turning
a finished flow into the 25 columns models/selected_features.json expects)
is a separate, later step.

Run via: sudo python src/capture.py
(sudo is required - raw packet capture needs elevated privileges.)
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scapy.all import IP, TCP, UDP, sniff

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# A flow is considered finished once this many seconds pass with no new
# packets - the same convention CICFlowMeter uses to decide when to stop
# aggregating a conversation's statistics and emit its feature row.
FLOW_TIMEOUT_SECONDS = 120

# A gap between consecutive packets longer than this counts as an "idle"
# period rather than just a slow-but-active one - used for Idle Max.
IDLE_THRESHOLD_SECONDS = 1.0

# Floor for flow duration when computing rate features (bytes_per_sec,
# Flow Packets/s), so a very short or single-packet flow can't divide by
# (near) zero - matches preprocess.py's MIN_DURATION_SEC exactly, since
# bytes_per_sec is one of preprocess.py's own derived features and must be
# computed identically live or the model sees an out-of-distribution value.
MIN_DURATION_SEC = 1e-3


@dataclass
class Packet:
    """One captured packet's relevant fields, boiled down to what flow
    feature computation will need later - not the full raw packet.

    Direction ("fwd"/"bwd") is deliberately not stored here - it's decided
    later, once the flow finishes, by extract_features()'s SYN-based
    logic. Deciding it eagerly at capture time (the original approach)
    got the very first packet of an already-in-progress TCP connection
    wrong whenever capture.py started sniffing mid-conversation - see
    extract_features()'s docstring.
    """

    timestamp: float
    src_ip: str
    src_port: int
    length: int
    payload_length: int
    header_length: int
    flags: str  # e.g. "PA" for PSH+ACK, "" if not TCP


@dataclass
class Flow:
    """Accumulates packets belonging to one conversation between two
    endpoints, until the flow is considered finished.

    "Forward" is fixed at flow creation to whichever endpoint sent the
    very first packet - every later packet's direction is decided by
    comparing its source against that original sender, matching
    CICFlowMeter's convention (the convention the training data used).
    """

    forward_ip: str
    forward_port: int
    backward_ip: str
    backward_port: int
    protocol: str  # "TCP" or "UDP"
    start_time: float
    last_seen: float = field(default=0.0)
    packets: list[Packet] = field(default_factory=list)

    def add_packet(self, packet: Packet) -> None:
        self.packets.append(packet)
        self.last_seen = packet.timestamp

    def is_expired(self, now: float) -> bool:
        return (now - self.last_seen) > FLOW_TIMEOUT_SECONDS


class FlowManager:
    """Tracks all currently-active flows, keyed by a direction-independent
    5-tuple so packets from either side of a conversation land in the same
    Flow object.
    """

    def __init__(self):
        self.flows: dict[tuple, Flow] = {}

    def _flow_key(
        self, ip_a: str, port_a: int, ip_b: str, port_b: int, protocol: str
    ) -> tuple:
        """Build a key that's identical regardless of which direction a
        packet is travelling, by always ordering the two endpoints the
        same way - so a reply packet's (src, dst) maps to the same key
        as the original request's (src, dst).
        """
        endpoint_a = (ip_a, port_a)
        endpoint_b = (ip_b, port_b)
        if endpoint_a <= endpoint_b:
            return (endpoint_a, endpoint_b, protocol)
        return (endpoint_b, endpoint_a, protocol)

    def process_packet(self, scapy_packet) -> None:
        """Extract the fields we care about from a raw Scapy packet, find
        or create its Flow, and append it.
        """
        if IP not in scapy_packet:
            return  # not an IP packet - nothing to bucket

        ip_layer = scapy_packet[IP]
        if TCP in scapy_packet:
            transport = scapy_packet[TCP]
            protocol = "TCP"
            flags = str(transport.flags)
            # dataofs is the TCP header length in 32-bit words; fall back
            # to the standard 20-byte header if it's missing.
            header_length = transport.dataofs * 4 if transport.dataofs else 20
        elif UDP in scapy_packet:
            transport = scapy_packet[UDP]
            protocol = "UDP"
            flags = ""
            header_length = 8
        else:
            return  # only TCP/UDP conversations are meaningful here

        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        src_port, dst_port = transport.sport, transport.dport
        now = time.time()

        key = self._flow_key(src_ip, src_port, dst_ip, dst_port, protocol)
        flow = self.flows.get(key)

        if flow is None:
            flow = Flow(
                forward_ip=src_ip,
                forward_port=src_port,
                backward_ip=dst_ip,
                backward_port=dst_port,
                protocol=protocol,
                start_time=now,
            )
            self.flows[key] = flow

        flow.add_packet(
            Packet(
                timestamp=now,
                src_ip=src_ip,
                src_port=src_port,
                # len(ip_layer), not len(scapy_packet): CICFlowMeter's packet
                # lengths are IP-packet sizes, excluding the Ethernet framing
                # Scapy's raw capture length would otherwise include.
                length=len(ip_layer),
                payload_length=len(bytes(transport.payload)),
                header_length=header_length,
                flags=flags,
            )
        )

    def pop_expired_flows(self) -> list[Flow]:
        """Remove and return every flow that's been idle past
        FLOW_TIMEOUT_SECONDS - these are ready for feature extraction.

        Known limitation: expiry is only checked when a new packet
        arrives (see main()'s handle_packet), so on a quiet interface a
        finished flow won't actually be popped until some other packet
        triggers the check. Fine for a first pass; a production version
        would check on a periodic timer instead.
        """
        now = time.time()
        expired_keys = [key for key, flow in self.flows.items() if flow.is_expired(now)]
        return [self.flows.pop(key) for key in expired_keys]


def _true_forward_endpoint(flow: Flow) -> tuple[str, int]:
    """Decide which of the flow's two endpoints actually initiated it.

    A bare TCP SYN (flag "S" without "A") is unambiguous proof of who
    started the connection - only the true initiator ever sends one, and
    a SYN-ACK reply always carries both flags, so checking for "A" not in
    flags rules that out. This overrides Flow.forward_ip/forward_port
    (set eagerly, at flow creation, to whoever sent the first packet
    *this program happened to observe*) for the common case where
    capture.py starts sniffing mid-conversation - e.g. a TCP connection
    that was already open before the sniffer launched. Without this
    check, such a flow gets its forward/backward endpoints swapped,
    which silently corrupts Destination Port and every Fwd/Bwd feature -
    caught by testing against real traffic, not a hypothetical.

    Falls back to Flow's original first-packet-observed assignment for
    UDP (no SYN concept) or a TCP flow where the true SYN was never
    captured at all (connection open before capture.py started AND no
    retransmitted SYN happened to occur during capture).
    """
    for packet in flow.packets:
        if "S" in packet.flags and "A" not in packet.flags:
            if (packet.src_ip, packet.src_port) == (flow.forward_ip, flow.forward_port):
                return flow.forward_ip, flow.forward_port
            return flow.backward_ip, flow.backward_port
    return flow.forward_ip, flow.forward_port


def extract_features(flow: Flow) -> dict:
    """Turn a finished Flow's packet list into the 25 feature values
    models/selected_features.json expects, keyed by feature name.

    Two deliberate approximations, documented rather than hidden:
    - Subflow Fwd/Bwd Bytes normally come from CICFlowMeter's own
      sub-flow-splitting logic, which is involved enough to not be worth
      replicating exactly here - approximated as equal to the flow's
      total forward/backward bytes (a common simplification).
    - Standard deviation uses numpy's default (population, ddof=0) rather
      than pandas' default (sample, ddof=1), so single-packet flows get a
      well-defined 0.0 instead of NaN - CICFlowMeter's own convention
      isn't recoverable from the training CSVs alone, and the difference
      is negligible once a flow has more than a few packets.

    Returns a plain dict, not ordered to match selected_features.json -
    the caller must reindex by that file's column order before scaling,
    the same way scale_features() expects a fixed column order.
    """
    packets = flow.packets
    true_fwd_endpoint = _true_forward_endpoint(flow)
    true_bwd_endpoint = (
        (flow.backward_ip, flow.backward_port)
        if true_fwd_endpoint == (flow.forward_ip, flow.forward_port)
        else (flow.forward_ip, flow.forward_port)
    )
    fwd_packets = [p for p in packets if (p.src_ip, p.src_port) == true_fwd_endpoint]
    bwd_packets = [p for p in packets if (p.src_ip, p.src_port) == true_bwd_endpoint]

    all_lengths = np.array([p.length for p in packets], dtype=float)
    fwd_lengths = np.array([p.length for p in fwd_packets], dtype=float)
    bwd_lengths = np.array([p.length for p in bwd_packets], dtype=float)

    duration = max(flow.last_seen - flow.start_time, 0.0)
    duration_clipped = max(duration, MIN_DURATION_SEC)

    total_fwd_bytes = float(fwd_lengths.sum()) if len(fwd_lengths) else 0.0
    total_bwd_bytes = float(bwd_lengths.sum()) if len(bwd_lengths) else 0.0
    total_bytes = total_fwd_bytes + total_bwd_bytes

    # Inter-arrival times: gaps between consecutive packets, sorted by
    # when they actually arrived. Flow IAT uses both directions; Fwd IAT
    # uses forward packets only.
    timestamps = sorted(p.timestamp for p in packets)
    gaps = np.diff(timestamps) if len(timestamps) > 1 else np.array([])

    fwd_timestamps = sorted(p.timestamp for p in fwd_packets)
    fwd_gaps = np.diff(fwd_timestamps) if len(fwd_timestamps) > 1 else np.array([])

    # Idle periods are the subset of gaps long enough to count as the
    # flow "going quiet" rather than just briefly pausing between packets
    # of the same burst.
    idle_gaps = gaps[gaps > IDLE_THRESHOLD_SECONDS] if len(gaps) else np.array([])

    return {
        "Packet Length Std": float(all_lengths.std()) if len(all_lengths) else 0.0,
        "Bwd Packet Length Std": float(bwd_lengths.std()) if len(bwd_lengths) else 0.0,
        "Packet Length Variance": float(all_lengths.var()) if len(all_lengths) else 0.0,
        "Bwd Packet Length Mean": float(bwd_lengths.mean()) if len(bwd_lengths) else 0.0,
        "Average Packet Size": float(all_lengths.mean()) if len(all_lengths) else 0.0,
        "Bwd Packet Length Max": float(bwd_lengths.max()) if len(bwd_lengths) else 0.0,
        "Packet Length Mean": float(all_lengths.mean()) if len(all_lengths) else 0.0,
        "Max Packet Length": float(all_lengths.max()) if len(all_lengths) else 0.0,
        "Subflow Bwd Bytes": total_bwd_bytes,
        "Destination Port": true_bwd_endpoint[1],
        "fwd_bwd_ratio": len(fwd_packets) / (len(bwd_packets) + 1),
        "Total Fwd Packets": len(fwd_packets),
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Subflow Fwd Bytes": total_fwd_bytes,
        "Total Length of Bwd Packets": total_bwd_bytes,
        "Flow IAT Max": float(gaps.max()) if len(gaps) else 0.0,
        "Idle Max": float(idle_gaps.max()) if len(idle_gaps) else 0.0,
        "PSH Flag Count": sum(1 for p in packets if "P" in p.flags),
        "bytes_per_sec": total_bytes / duration_clipped,
        "Flow Packets/s": len(packets) / duration_clipped,
        "Fwd Header Length": sum(p.header_length for p in fwd_packets),
        "act_data_pkt_fwd": sum(1 for p in fwd_packets if p.payload_length > 0),
        "Fwd IAT Std": float(fwd_gaps.std()) if len(fwd_gaps) else 0.0,
        "Fwd IAT Mean": float(fwd_gaps.mean()) if len(fwd_gaps) else 0.0,
        "Fwd IAT Max": float(fwd_gaps.max()) if len(fwd_gaps) else 0.0,
    }


def main() -> None:
    """Sniff packets indefinitely, printing each finished flow's extracted
    feature row - the same 25 columns models/selected_features.json
    expects, ready for scaling and inference in detector.py (next step).
    """
    manager = FlowManager()

    def handle_packet(packet) -> None:
        manager.process_packet(packet)
        for flow in manager.pop_expired_flows():
            features = extract_features(flow)
            fwd_ip, fwd_port = _true_forward_endpoint(flow)
            bwd_ip, bwd_port = (
                (flow.backward_ip, flow.backward_port)
                if (fwd_ip, fwd_port) == (flow.forward_ip, flow.forward_port)
                else (flow.forward_ip, flow.forward_port)
            )
            print(
                f"\nFlow finished: {fwd_ip}:{fwd_port} -> "
                f"{bwd_ip}:{bwd_port} ({flow.protocol}), "
                f"{len(flow.packets)} packets"
            )
            for name, value in features.items():
                print(f"  {name}: {value}")

    print("Capturing traffic - press Ctrl+C to stop.")
    sniff(prn=handle_packet, store=False)


if __name__ == "__main__":
    main()
