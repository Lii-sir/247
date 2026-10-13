import mmap


class SharedImage:
    """Read-only raw image view; owner must outlive every NumPy view of its pixels."""
    def __init__(self, descriptor):
        import numpy as np
        name = str(descriptor.get("shm_name") or "")
        size = int(descriptor.get("image_size", 0))
        offset = int(descriptor.get("image_offset", 0))
        width, height = int(descriptor.get("image_width", 0)), int(descriptor.get("image_height", 0))
        channels = int(descriptor.get("image_channels", 3))
        stride = int(descriptor.get("image_stride", 0)) or width * channels
        if (not name or min(size, width, height) <= 0 or offset < 0 or channels not in (1, 3)
                or stride < width * channels or size < stride * (height - 1) + width * channels):
            raise ValueError("invalid raw shared-memory image layout")
        self.mapping = mmap.mmap(-1, offset + size, tagname=name, access=mmap.ACCESS_READ)
        try:
            self.image = np.ndarray((height, width, channels), dtype=np.uint8,
                                    buffer=self.mapping, offset=offset,
                                    strides=(stride, channels, 1))
            self.image.flags.writeable = False
        except Exception:
            self.mapping.close()
            raise

    def close(self):
        self.image = None
        self.mapping.close()


def read_shared_memory(shm_name: str, data_size: int) -> bytes:
    # transport 层负责按名称读取共享内存字节
    with mmap.mmap(
            -1,
            data_size,
            tagname=shm_name,
            access=mmap.ACCESS_READ) as mm:
        return mm.read(data_size)
