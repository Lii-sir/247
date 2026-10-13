import mmap


def read_shared_memory(shm_name: str, data_size: int) -> bytes:
    # transport 层负责按名称读取共享内存字节
    with mmap.mmap(
            -1,
            data_size,
            tagname=shm_name,
            access=mmap.ACCESS_READ) as mm:
        return mm.read(data_size)
