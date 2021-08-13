import struct
import uuid
from typing import List

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


class SetAgentPositionSideChannel(SideChannel):
    """
    This is the SideChannel for setting an agent's position in Unity.
    """
    def __init__(self) -> None:
        channel_id = uuid.UUID("821d1b06-5035-4518-9e67-a34946637260")
        super().__init__(channel_id)
        self.success = None 

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.success = msg.read_bool()

    def send_request(self, key: str, value: List[float]) -> None:
        """
        Sends a request to Unity
        The arguments for setting the agent position are are ("agentPosition", [AGENT_ID, POSITION_X, POSITION_Y, POSITION_Z OPTIONAL(ROTATION_X, ROTATION_Y, ROTATION_Z, ROTATION_W)])
        """
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        super().queue_message_to_send(msg)

    def build_immediate_request(self, key: str, value: List[float]) -> bytearray:
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)

        result = bytearray()
        result += self.channel_id.bytes_le
        result += struct.pack("<i", len(msg.buffer))
        result += msg.buffer
        return result
