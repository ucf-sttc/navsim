from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from typing import List, Optional
import numpy as np
import struct

import uuid

class MapSideChannel(SideChannel):
    """
    This is the SideChannel for retrieving map data from Unity.
    You can send map requests to Unity using send_request.
    """
    def __init__(self, ) -> None:
        channel_id = uuid.UUID("24b099f1-b184-407c-af72-f3d439950bdb")
        super().__init__(channel_id)
        self.requested_map = None
        self.resolution = None
        self.navmap_max_x = None
        self.navmap_max_y = None
        self.unity_max_x = None
        self.unity_max_z = None

    def on_message_received(self, msg: IncomingMessage) -> np.ndarray:
        if self.resolution is None:
            raise ValueError('resolution to full map is not set')

        raw_bytes = msg.get_raw_bytes()
        self.requested_map = np.unpackbits(raw_bytes)[
                             0:self.resolution[0] * self.resolution[1]]
        self.requested_map = self.requested_map.reshape((self.resolution[1],
                                                         self.resolution[0]))
        return self.requested_map

    def _request_helper(self, key: Optional[str]='binaryMap', value: Optional[List[float]] = None ):
        if key == 'binaryMap':
            if value is None:
                value = []
                self.resolution = [self.navmap_max_x,self.navmap_max_y]
        elif key == 'binaryMapZoom':
            if value is None:
                raise ValueError('[x,y] not provided')
            self.resolution = [100, 100]  # resolution at cm scale for 1 square meter tile
        else:
            raise ValueError('invalid key')
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        return msg
        
    def send_request(self, key: Optional[str]='binaryMap', value: Optional[List[float]] = None) -> None:
        """
        Sends a request to Unity
        The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
        Or ("binaryMapZoom", [ROW, COL])
        """
        msg = self._request_helper(key=key,value=value)
        super().queue_message_to_send(msg)

    def build_immediate_request(self, key: Optional[str]='binaryMap', value: Optional[List[float]] = None) -> bytearray:
        msg = self._request_helper(key=key,value=value)

        result = bytearray()
        result += self.channel_id.bytes_le
        result += struct.pack("<i", len(msg.buffer))
        result += msg.buffer
        return result
        
