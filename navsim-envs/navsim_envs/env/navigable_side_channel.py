import struct
import uuid
from typing import List

from mlagents_envs.side_channel import SideChannel, IncomingMessage, \
    OutgoingMessage


class NavigableSideChannel(SideChannel):
    """
    This is the SideChannel for requesting or checking a navigable point in Unity.
    You can send requests to Unity using send_request.
    """
    resolution = []

    def __init__(self) -> None:
        channel_id = uuid.UUID("fbae7da3-76e8-4c37-86c9-ad647c74fd69")
        super().__init__(channel_id)
        self.point = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        IncomingMessage is a list of floats
        IncomingMessage is empty if there was no point that satisfied the request,
        otherwise it will contain the requested navigable point in Unity's world space
        """
        if self.resolution is None:
            self.point = None
            return

        self.point = msg.read_float32_list()


    def send_request(self, key: str, value: List[float]) -> None:
        """
        Sends a request to Unity
        The arguments for the request are ("navigable", [POINT]), where POINT can be one of the following:
            1. []        - requests random navigable point
            2. [x, z]    - check if there is a navigable point (x, y, z) at any height y
            3. [x, y, z] - check if (x, y, z) is a navigable point
        """
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        super().queue_message_to_send(msg)

    def build_immediate_request(self, key: str, value: List[float]) -> bytearray:
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)

        result = bytearray()
        result += self.channel_id.bytes_le
        result += struct.pack("<i", len(msg.buffer))
        result += msg.buffer
        return result
