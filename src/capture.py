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

from scapy.all import IP, TCP, UDP, sniff

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# A flow is considered finished once this many seconds pass with no new
# packets - the same convention CICFlowMeter uses to decide when to stop
# aggregating a conversation's statistics and emit its feature row.
FLOW_TIMEOUT_SECONDS = 120


@dataclass
class Packet:
    """One captured packet's relevant fields, boiled down to what flow
    feature computation will need later - not the full raw packet.
    """

    timestamp: float
    direction: str  # "fwd" or "bwd"
    length: int
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

        direction = (
            "fwd" if (src_ip, src_port) == (flow.forward_ip, flow.forward_port) else "bwd"
        )
        flow.add_packet(
            Packet(
                timestamp=now,
                direction=direction,
                length=len(scapy_packet),
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


def main() -> None:
    """Sniff packets indefinitely, printing a summary each time a flow
    finishes - proves capture and flow-grouping work correctly before
    feature extraction is added on top.
    """
    manager = FlowManager()

    def handle_packet(packet) -> None:
        manager.process_packet(packet)
        for flow in manager.pop_expired_flows():
            print(
                f"Flow finished: {flow.forward_ip}:{flow.forward_port} -> "
                f"{flow.backward_ip}:{flow.backward_port} ({flow.protocol}), "
                f"{len(flow.packets)} packets"
            )

    print("Capturing traffic - press Ctrl+C to stop.")
    sniff(prn=handle_packet, store=False)


if __name__ == "__main__":
    main()
