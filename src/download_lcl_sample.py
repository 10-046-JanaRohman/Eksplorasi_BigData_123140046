"""Ambil satu potongan CSV Low Carbon London tanpa mengunduh ZIP 700+ MB penuh.

Sumber asli: UK Power Networks / London Datastore. ZIP memakai 168 CSV; skrip
memakai HTTP Range untuk mengambil direktori ZIP dan satu CSV pertama saja.
"""
from __future__ import annotations

import struct
import urllib.request
import zlib
from pathlib import Path

URL = (
    "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/"
    "04feba67-f1a3-4563-98d0-f3071e3d56d1/"
    "Power-Networks-LCL-June2015(withAcornGps).csv_Pieces.zip"
)
OUT = Path("data/raw/lcl_sample.csv")
USER_AGENT = "Mozilla/5.0 (educational Spark replication)"


def fetch(start: int, end: int) -> bytes:
    request = urllib.request.Request(
        URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def remote_size() -> int:
    request = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def first_member() -> tuple[str, int, int]:
    """Return (name, compressed_size, local_header_offset) for first CSV member."""
    size = remote_size()
    tail_start = max(0, size - 131_072)
    tail = fetch(tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("End of central directory ZIP tidak ditemukan.")
    _, _, _, _, entries, directory_size, directory_offset, _ = struct.unpack(
        "<4s4H2LH", tail[eocd : eocd + 22]
    )
    if entries == 0 or directory_offset < tail_start:
        raise RuntimeError("Direktori ZIP tidak berada pada bagian akhir yang diambil.")
    directory = tail[directory_offset - tail_start : directory_offset - tail_start + directory_size]
    pos = 0
    while pos < len(directory):
        if directory[pos : pos + 4] != b"PK\x01\x02":
            break
        fields = struct.unpack("<4s6H3L5H2L", directory[pos : pos + 46])
        name_len, extra_len, comment_len = fields[10:13]
        compressed_size, local_offset = fields[8], fields[16]
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8")
        if name.lower().endswith(".csv"):
            return name, compressed_size, local_offset
        pos += 46 + name_len + extra_len + comment_len
    raise RuntimeError("Tidak ada anggota CSV di dalam ZIP.")


def main() -> None:
    name, compressed_size, local_offset = first_member()
    # Header lokal ZIP panjangnya 30 byte, kemudian nama dan extra field.
    local_header = fetch(local_offset, local_offset + 30 + 1024)
    if local_header[:4] != b"PK\x03\x04":
        raise RuntimeError("Header lokal ZIP tidak valid.")
    name_len, extra_len = struct.unpack("<HH", local_header[26:30])
    payload_start = local_offset + 30 + name_len + extra_len
    compressed = fetch(payload_start, payload_start + compressed_size - 1)
    data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"Tersimpan: {OUT} | anggota: {name} | {len(data):,} byte")


if __name__ == "__main__":
    main()
