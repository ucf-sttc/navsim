import struct
import uuid
from typing import List

import numpy as np
from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


class NavSimUnityEnv(UnityEnvironment):
    """NavSimGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface

    Read the **NavSim Environment Tutorial** on how to use this class.
    """

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        env_path = env_config.get('env_path', None)
        log_folder = str(env_config.get('log_folder', './env_log'))
        seed = env_config.get('seed', 0)
        timeout = self.env_config.get('timeout', 600) + (0.5 * (
                self.start_from_episode - 1))
        worker_id = env_config.get('worker_id', 0)
        base_port = env_config.get('base_port', None)

        super().__init__(file_name=env_path,
                         log_folder=str(log_folder),
                         seed=seed,
                         timeout_wait=timeout,
                         worker_id=worker_id,
                         base_port=base_port,
                         no_graphics=False,
                         side_channels=[],
                         additional_args="")


class MapSideChannel(SideChannel):
    """This is the SideChannel for retrieving map data from Unity.
    You can send map requests to Unity using send_request.
    The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
    """
    resolution = None

    def __init__(self) -> None:
        channel_id = uuid.UUID("24b099f1-b184-407c-af72-f3d439950bdb")
        super().__init__(channel_id)
        self.requested_map = None

    def on_message_received(self, msg: IncomingMessage) -> np.ndarray:
        if self.resolution is None:
            return None

        raw_bytes = msg.get_raw_bytes()
        self.requested_map = np.unpackbits(raw_bytes)[
                             0:self.resolution[0] * self.resolution[1]]
        self.requested_map = self.requested_map.reshape((self.resolution[1],
                                                         self.resolution[0]))
        return self.requested_map

    def send_request(self, key: str, value: List[float]) -> None:
        """Sends a request to Unity
        The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
        """
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        super().queue_message_to_send(msg)

    def build_immediate_request(self, key: str,
                                value: List[float]) -> bytearray:
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)

        result = bytearray()
        result += self.channel_id.bytes_le
        result += struct.pack("<i", len(msg.buffer))
        result += msg.buffer
        return result
