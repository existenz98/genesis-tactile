from typing import Iterator, Optional, Tuple
import numpy as np

class FrameSource:
    """
    Abstract streaming frame source: yields BGR uint8 frames.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
    
    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError
