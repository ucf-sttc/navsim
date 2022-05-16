from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from typing import List, Optional
import numpy as np
import struct
import uuid

class AroraSideChannelBase(SideChannel):
    #def __init__(self, channel_id) -> None:
    #    super().__init__(channel_id)

    def _request_helper(self, key:Optional[str]=None, value: Optional[List[float]]=None ) -> OutgoingMessage:
        msg = OutgoingMessage()
        if key is not None:
            msg.write_string(key)
        if value is not None:
            msg.write_float32_list(value)
        return msg

    def send_request(self, key:Optional[str]=None, value: Optional[List[float]]=None) -> None:
        """Sends a request to Unity
        """
        msg = self._request_helper(key=key,value=value)
        super().queue_message_to_send(msg)

    def build_immediate_request(self, key:Optional[str]=None, value: Optional[List[float]]=None) -> bytearray:
        msg = self._request_helper(key=key,value=value)

        result = bytearray()
        result += self.channel_id.bytes_le
        result += struct.pack("<i", len(msg.buffer))
        result += msg.buffer
        return result

class MapSideChannel(AroraSideChannelBase):
    """This is the SideChannel for retrieving map data from Unity.
    You can send map requests to Unity using send_request.
    The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
    """

    def __init__(self) -> None:
        super().__init__(channel_id=uuid.UUID("24b099f1-b184-407c-af72-f3d439950bdb"))
        self.requested_map = None
        self.resolution = None
        self.navmap_max_x = None
        self.navmap_max_y = None
        self.unity_max_x = None
        self.unity_max_z = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        if self.resolution is None:
            raise ValueError('resolution to full map is not set')

        raw_bytes = msg.get_raw_bytes()
        self.requested_map = np.unpackbits(raw_bytes)[
                             0:self.resolution[0] * self.resolution[1]]
        self.requested_map = self.requested_map.reshape((self.resolution[1],
                                                         self.resolution[0]))

    def _request_helper(self, key: Optional[str]='binaryMap', value: Optional[List[float]] = None ) -> OutgoingMessage:
        if key == 'binaryMap':
            if value is None:
                value = []
                self.resolution = [self.navmap_max_x, self.navmap_max_y] # how to determine resolution??
        elif key == 'binaryMapZoom':
            if value is None:
                raise ValueError('[x,y] not provided')
            self.resolution = [100, 100]  # resolution at cm scale for 1 square meter tile
        else:
            raise ValueError('invalid key')

        return super()._request_helper(key=key,value=value)

    def send_request(self, key: Optional[str] = 'binaryMap', value: Optional[List[float]] = None) -> None:
        """Sends a request to Unity
        The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
        Or ("binaryMapZoom", [ROW, COL])
        """
        super().send_request(key=key,value=value)

    def build_immediate_request(self, key: Optional[str] = 'binaryMap',
                                value: Optional[List[float]] = None) -> bytearray:
        return super().build_immediate_request(key=key,value=value)


class NavigableSideChannel(AroraSideChannelBase):
    """
    This is the SideChannel for requesting or checking a navigable point in Unity.
    You can send requests to Unity using send_request.
    """

    def __init__(self) -> None:
        super().__init__(channel_id = uuid.UUID("fbae7da3-76e8-4c37-86c9-ad647c74fd69"))
        self.point = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        IncomingMessage is a list of floats
        IncomingMessage is empty if there was no point that satisfied the request,
        otherwise it will contain the requested navigable point in Unity's world space
        """
        self.point = msg.read_float32_list()


    #def send_request(self, key: str, value: List[float]) -> None:
        """
        Sends a request to Unity
        The arguments for the request are ("navigable", [POINT]), where POINT can be one of the following:
            1. []        - requests random navigable point
            2. [x, z]    - check if there is a navigable point (x, y, z) at any height y
            3. [x, y, z] - check if (x, y, z) is a navigable point
        """
    #    super.send_request(key=key,value=value)

    #def build_immediate_request(self, key: str, value: List[float]) -> bytearray:
    #    return super.build_immediate_request(key=key,value=value)


class ShortestPathSideChannel(AroraSideChannelBase):
    """
    This is the SideChannel for requesting the shortest path as a list of floats
    """
    def __init__(self) -> None:
        super().__init__(channel_id = uuid.UUID("dc4b7d9a-774e-49bc-b73e-4a221070d716"))
        self.path=None

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        IncomingMessage is a list of floats
        Reshaped to n x 3 array (array of 3D points)
        """
        path = msg.read_float32_list()
        #path_length = int(len(path)/3)
        print('inside channel code 1:',path)
        self.path = np.reshape(path, (-1,3))  # reshape -1,3
        print('inside channel code 2:',self.path)
    
    def send_request(self) -> None:
        """
        Sends a request to Unity for the shortest path points
        """
        super().send_request()

    def build_immediate_request(self)-> bytearray:
        return super().build_immediate_request()


class SetAgentPositionSideChannel(AroraSideChannelBase):
    """
    This is the SideChannel for setting an agent's position in Unity.
    The arguments for setting the agent position are are ("agentPosition", [AGENT_ID, POSITION_X, POSITION_Y, POSITION_Z OPTIONAL(ROTATION_X, ROTATION_Y, ROTATION_Z, ROTATION_W)])

    """
    def __init__(self) -> None:
        super().__init__(channel_id = uuid.UUID("821d1b06-5035-4518-9e67-a34946637260"))
        self.success = None 

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.success = msg.read_bool()

    def send_request(self, key:str, value: List[float]) -> None:

        super().send_request(key=key,value=value)

    def build_immediate_request(key:str, value: List[float]) -> bytearray:
        return super().build_immediate_request(key=key,value=value)


