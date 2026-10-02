# SPDX-License-Identifier: Apache-2.0
"""HIXL implementation of the LMCache MP transfer channel.

The ZMQ handshake matches the Mooncake channel. The bytes move through
``lmcache_ascend.hixl_npu_comms`` instead of Mooncake.
"""

from __future__ import annotations

import os
import socket
import threading
from typing import Union

import msgspec
import torch
import torch_npu  # noqa: F401
import zmq

import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hixl_npu_comms as hixl_comms
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

logger = init_logger(__name__)

_HANDSHAKE_TIMEOUT_MS = 60_000


def _parse_url(url: str) -> tuple[str, int]:
    """Parse ``host:port`` or ``tcp://host:port``."""
    stripped = url.split("://", 1)[-1]
    host, _, port = stripped.rpartition(":")
    if not host or not port:
        raise ValueError(f"Invalid transfer channel url: {url!r}")
    return host, int(port)


def _free_port() -> int:
    """Return an unused TCP port on this host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _set_npu() -> None:
    """Select the first visible NPU so HIXL has a device context."""
    # The first entry in ASCEND_RT_VISIBLE_DEVICES is torch device 0.
    torch.npu.set_device(0)


class HandshakeMsgBase(msgspec.Struct, tag=True):
    """Base tagged handshake message."""


class InitReq(HandshakeMsgBase):
    """Client hello: advertise url, HIXL engine id, L1 base pointer."""

    advertise_url: str
    session_id: str
    buffer_base_ptr: int


class InitResp(HandshakeMsgBase):
    """Server reply with the same fields."""

    advertise_url: str
    session_id: str
    buffer_base_ptr: int


HandshakeMsg = Union[InitReq, InitResp]


class HixlTransferChannelClient(TransferChannelClient):
    """Reads remote L1 bytes through one HIXL peer."""

    def __init__(
        self,
        context: HixlTransferChannelContext,
        remote_session_id: str,
        remote_buffer_ptr: int,
    ) -> None:
        """Store the peer HIXL id and remote L1 base pointer."""
        self._ctx = context
        self._remote_session_id = remote_session_id
        self._remote_buffer_ptr = remote_buffer_ptr
        self._connected = False
        self._task_counter = 0
        self._tasks: dict[int, tuple[object, list[TransferChannelAddress]]] = {}
        self._lock = threading.Lock()

    def _ensure_connected(self) -> None:
        """Open the HIXL connection to this peer once."""
        if self._connected:
            return
        logger.info("HIXL connect %s", self._remote_session_id)
        self._ctx.engine.connect(self._remote_session_id)
        self._connected = True

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit an async HIXL read into the local L1 buffer."""
        if len(local_addresses) != len(remote_addresses):
            raise ValueError("local and remote address lists differ in length")
        self._ensure_connected()
        ops = []
        for local, remote in zip(local_addresses, remote_addresses, strict=True):
            if local.size != remote.size:
                raise ValueError("local and remote sizes must match")
            ops.append(
                hixl_comms.TransferOpDesc(
                    local_addr=self._ctx.l1_memory_desc.ptr + local.offset,
                    remote_addr=self._remote_buffer_ptr + remote.offset,
                    len=local.size,
                )
            )
        request = self._ctx.engine.transfer_async(
            self._remote_session_id, hixl_comms.READ, ops
        )
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (request, list(remote_addresses))
        return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Poll one submitted HIXL read."""
        with self._lock:
            request, remote_addresses = self._tasks[task_id]
        status = self._ctx.engine.get_transfer_status(request)
        if status == hixl_comms.TransferStatus.COMPLETED:
            with self._lock:
                self._tasks.pop(task_id, None)
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[True] * len(remote_addresses)
            )
        if status in (
            hixl_comms.TransferStatus.FAILED,
            hixl_comms.TransferStatus.TIMEOUT,
        ):
            with self._lock:
                self._tasks.pop(task_id, None)
            logger.error("HIXL read %s status %s", self._remote_session_id, status)
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[False] * len(remote_addresses)
            )
        return TransferChannelReadResult(finished=False, succeeded_mask=[])

    def close(self) -> None:
        """Drop pending task ids. HIXL has no per-client handle to release."""
        with self._lock:
            self._tasks.clear()


class HixlTransferChannelServer(TransferChannelServer):
    """ZMQ REP handshake that exchanges HIXL engine ids and L1 base pointers."""

    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        context: HixlTransferChannelContext,
    ) -> None:
        """Bind the handshake socket and start the serve thread."""
        self._ctx = context
        self._advertise_url = advertise_url
        self._running = True
        self._socket = context.zmq_context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        host, port = _parse_url(listen_url)
        self._socket.bind(f"tcp://{host}:{port}")
        self._thread = threading.Thread(
            target=self._serve_loop, name="tc-hixl-server", daemon=True
        )
        self._thread.start()

    def _serve_loop(self) -> None:
        """Answer handshake requests until ``close``."""
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=1000))
                if self._socket not in events:
                    continue
                req = msgspec.msgpack.decode(self._socket.recv(), type=HandshakeMsg)
                self._socket.send(msgspec.msgpack.encode(self._handle_msg(req)))
            except Exception:
                if self._running:
                    logger.exception("HIXL transfer channel server loop failed")

    def _handle_msg(self, req: HandshakeMsg) -> HandshakeMsg:
        """Register the peer and return this server's HIXL id."""
        if not isinstance(req, InitReq):
            raise ValueError(f"Unexpected handshake message: {type(req)}")
        self._ctx.register_client(
            req.advertise_url,
            HixlTransferChannelClient(
                context=self._ctx,
                remote_session_id=req.session_id,
                remote_buffer_ptr=req.buffer_base_ptr,
            ),
        )
        return InitResp(
            advertise_url=self._ctx.advertise_url,
            session_id=self._ctx.engine_id,
            buffer_base_ptr=self._ctx.l1_memory_desc.ptr,
        )

    def close(self) -> None:
        """Stop the serve thread and close the socket."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._socket.close(linger=0)


class HixlTransferChannelContext(TransferChannelContext):
    """One HIXL engine registered over the whole L1 buffer."""

    def __init__(
        self,
        l1_memory_desc: L1MemoryDesc,
        listen_url: str,
        advertise_url: str,
    ) -> None:
        """Initialize HIXL, register L1, and start the handshake server."""
        _set_npu()
        self._l1_memory_desc = l1_memory_desc
        self.listen_url = listen_url
        self.advertise_url = advertise_url
        host, _port = _parse_url(advertise_url)
        self._advertise_host = host
        self.engine = hixl_comms.Hixl()
        hixl_port = _free_port()
        self.engine_id = f"{host}:{hixl_port}"
        pool = os.environ.get("ASCEND_BUFFER_POOL", "4:8")
        self.engine.initialize(self.engine_id, {"BufferPool": pool})
        mem_type = (
            hixl_comms.MEM_DEVICE
            if hixl_comms.is_device_memory(l1_memory_desc.ptr)
            else hixl_comms.MEM_HOST
        )
        # aclrtMallocHost memory is already mapped. HIXL's connect path
        # calls aclrtHostRegister and fails with 507899 unless that mapping
        # is dropped first. Restore a device VA afterwards for GPU copies.
        if lmc_ops.get_device_ptr(l1_memory_desc.ptr) is not None:
            lmc_ops.unregister_ptr(l1_memory_desc.ptr)
        self._mem_handle = self.engine.register_mem(
            l1_memory_desc.ptr, l1_memory_desc.size, mem_type
        )
        dev_ptr = hixl_comms.get_dev_va(
            torch.npu.current_device(), l1_memory_desc.ptr, l1_memory_desc.size
        )
        if dev_ptr is not None:
            lmc_ops.register_mapping(l1_memory_desc.ptr, dev_ptr, l1_memory_desc.size)
        logger.info(
            "HIXL engine %s registered L1 ptr=%#x size=%d type=%s pool=%s",
            self.engine_id,
            l1_memory_desc.ptr,
            l1_memory_desc.size,
            mem_type,
            pool,
        )
        self.zmq_context = zmq.Context.instance()
        self._clients: dict[str, HixlTransferChannelClient] = {}
        self._lock = threading.Lock()
        self._server = HixlTransferChannelServer(listen_url, advertise_url, self)

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        """Return the registered L1 region."""
        return self._l1_memory_desc

    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate offsets against L1 and return channel addresses."""
        size = self._l1_memory_desc.size
        out: list[TransferChannelAddress] = []
        for offset, obj_size in lmcache_addresses:
            if offset < 0 or offset + obj_size > size:
                raise ValueError(
                    f"Object [{offset:#x}, {offset + obj_size:#x}) is outside L1"
                )
            out.append(TransferChannelAddress(offset=offset, size=obj_size))
        return out

    def get_transfer_channel_server(self) -> HixlTransferChannelServer:
        """Return the handshake server."""
        return self._server

    def get_transfer_channel_client(
        self, peer_advertise_url: str
    ) -> HixlTransferChannelClient:
        """Return the peer client, dialing the handshake if needed."""
        with self._lock:
            client = self._clients.get(peer_advertise_url)
            if client is not None:
                return client
        return self.register_client(peer_advertise_url, self._connect(peer_advertise_url))

    def get_num_connected_clients(self) -> int:
        """Return how many peers have completed the handshake."""
        with self._lock:
            return len(self._clients)

    def register_client(
        self, key: str, client: HixlTransferChannelClient
    ) -> HixlTransferChannelClient:
        """Keep the first client registered for ``key``."""
        with self._lock:
            existing = self._clients.get(key)
            if existing is None:
                self._clients[key] = client
                return client
            if existing is client:
                return client
        client.close()
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Drop the client for one peer."""
        with self._lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is not None:
            client.close()

    def close(self) -> None:
        """Stop the handshake server and deregister L1."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        self._server.close()
        for client in clients:
            client.close()
        try:
            self.engine.deregister_mem(self._mem_handle)
        except Exception:
            logger.exception("HIXL deregister failed")
        self.engine.finalize()

    def _connect(self, server_url: str) -> HixlTransferChannelClient:
        """Dial the peer handshake and build a client from the reply."""
        sock = self.zmq_context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)
        host, port = _parse_url(server_url)
        sock.connect(f"tcp://{host}:{port}")
        try:
            logger.info("HIXL handshake to %s as %s", server_url, self.engine_id)
            sock.send(
                msgspec.msgpack.encode(
                    InitReq(
                        advertise_url=self.advertise_url,
                        session_id=self.engine_id,
                        buffer_base_ptr=self._l1_memory_desc.ptr,
                    )
                )
            )
            try:
                raw = sock.recv()
            except zmq.Again as err:
                raise LMCacheTimeoutError(
                    f"HIXL handshake to {server_url!r} timed out"
                ) from err
            reply = msgspec.msgpack.decode(raw, type=HandshakeMsg)
            if not isinstance(reply, InitResp):
                raise RuntimeError(f"Unexpected HIXL handshake reply {type(reply)}")
        finally:
            sock.close(linger=0)
        return HixlTransferChannelClient(
            context=self,
            remote_session_id=reply.session_id,
            remote_buffer_ptr=reply.buffer_base_ptr,
        )


def create_hixl_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs: object,
) -> HixlTransferChannelContext:
    """Register the HIXL factory entry used by ``--p2p-transfer-engine hixl``."""
    del kwargs
    return HixlTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
    )


register_transfer_channel_factory("hixl", create_hixl_transfer_channel_context)
