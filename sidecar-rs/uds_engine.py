"""Decision Index client engine that talks to the PQNLD readout over a Unix socket.

Used as ``--engine uds_engine:UdsSystemOne --option uds=/tmp/pqnld.sock`` so the
kit's local hop never crosses the TCP stack. The outbound engine call remains
HTTP because vLLM does not serve a Unix socket.
"""
from __future__ import annotations

import httpx

from decision_index.engines.http import HttpSystemOne


class UdsSystemOne(HttpSystemOne):
    name = "uds"

    def __init__(self, uds=None, **options):
        base_url = options.pop("base_url", "http://localhost")
        super().__init__(base_url=base_url, **options)
        if not uds:
            raise ValueError("uds engine requires the 'uds' option")
        timeout = options.get("timeout", 600)
        headers = dict(self.client.headers)
        self.client.close()
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds=uds),
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        )
        self.provenance["kind"] = "uds"
        self.provenance["uds"] = uds
