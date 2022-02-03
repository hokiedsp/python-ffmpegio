import numpy as np
import fractions, re
from ..utils import get_pixel_config, get_audio_format, spec_stream

from struct import Struct
from collections import namedtuple

# https://docs.microsoft.com/en-us/previous-versions//dd183376(v=vs.85)?redirectedfrom=MSDN


class FlagProcessor:
    def __init__(self, name, flags, masks, defaults):
        self.template = namedtuple(
            name,
            flags,
            defaults=defaults,
        )
        self.masks = self.template._make(masks)

    def default(self):
        return self.template()

    def unpack(self, flags):
        return self.template._make((bool(flags & mask) for mask in self.masks))

    def pack(self, flags):
        return sum((mask if flag else 0 for flag, mask in zip(flags, self.masks)))


from itertools import accumulate


class StructProcessor:
    def __init__(self, name, format, fields, defaults, **flags):
        if "S" in format or "C" in format:
            # expand the format
            m = re.match(r"([<>!=])?(.+)", format)
            fmt_items = [
                (int(m[1]) if m[1] else 1, m[2])
                for m in re.finditer(r"(\d*)([xcCbB?hHiIlLqQnNefdsSpP])", m[2])
            ]
            fmt_counts = [1 if f in "sSp" else count for count, f in fmt_items]
            fmt_offsets = list((0, *accumulate(fmt_counts)))
            is_str = [False] * fmt_offsets[-1]
            for itm, offset in zip(fmt_items, fmt_offsets[:-1]):
                is_str[offset] = itm[1] in "SC"
            self.is_str = [fields[i] for i, tf in enumerate(is_str) if tf]
            format = format.replace("C", "c").replace("S", "s")
        self.struct = Struct(format)
        self.template = namedtuple(name, fields, defaults=defaults)
        self.flags = ((k, FlagProcessor(*v)) for k, v in flags.items())

    def default(self):
        data = self.template()
        return data._replace(**{k: proc.default() for k, proc in self.flags})

    def unpack(self, buffer):
        data = self.template._make(super().unpack(buffer))
        return data._replace(
            **{k: proc.unpack(getattr(data, k)) for k, proc in self.flags}
        )

    def unpack_from(self, buffer, offset=0):
        data = self.template._make(super().unpack_from(buffer, offset))
        return data._replace(
            **{k: proc.unpack(getattr(data, k)) for k, proc in self.flags}
        )

    def pack(self, ntuple):
        ntuple = ntuple._replace(
            **{k: proc.pack(getattr(ntuple, k)) for k, proc in self.flags}
        )
        return super().pack(*ntuple)

    def pack_into(self, buffer, offset, ntuple):
        ntuple = ntuple._replace(
            **{k: proc.pack(getattr(ntuple, k)) for k, proc in self.flags}
        )
        super().pack_into(buffer, offset, *ntuple)

    @property
    def size(self):
        return self.struct.size


AVIMainHeader = StructProcessor(
    "Avih",
    "<10I",
    (
        "micro_sec_per_frame",
        "max_bytes_per_sec",
        "padding_granularity",
        "flags",
        "total_frames",
        "initial_frames",
        "streams",
        "suggested_buffer_size",
        "width",
        "height",
    ),
    (0,) * 10,
    flags=(
        "AvihFlags",
        (
            "copyrighted",
            "has_index",
            "is_interleaved",
            "must_use_index",
            "was_capture_file",
        ),
        (
            int("0x00020000", 0),
            int("0x00000010", 0),
            int("0x00000100", 0),
            int("0x00000020", 0),
            int("0x00010000", 0),
        ),
        (False,) * 5,
    ),
)


def decode_avih(data, prev_chunk):
    return AVIMainHeader.unpack(data)


AVIStreamHeader = StructProcessor(
    "Strh",
    "<4S4SI2H8I4h",
    (
        "fcc_type",  # 'auds','mids','txts','vids'
        "fcc_handler",
        "flags",
        "priority",
        "language",
        "initial_frame",
        "scale",
        "rate",
        "start",
        "length",
        "suggested_buffer_size",
        "quality",
        "sample_size",
        "frame_left",
        "frame_top",
        "frame_right",
        "frame_bottom",
    ),
    (b"auds", b"\0" * 4, *((0,) * 15)),
    flags=(
        "StrhFlags",
        (
            "video_pal_changes",
            "disabled",
        ),
        (
            int("0x00000001", 0),
            int("0x00010000", 0),
        ),
        (False,) * 2,
    ),
)


def decode_strh(data, prev_chunk):
    return AVIStreamHeader.unpack(data)


# PCM audio
WAVE_FORMAT_PCM = 1
# IEEE floating-point audio
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = int("FFFE", 16)  # /* Microsoft */

BitmapInfoHeaer = StructProcessor(
    "AVISTREAMHEADER",
    "IiiHHIIiiII",
    (
        "size",
        "width",
        "height",
        "planes",
        "bit_count",
        "compression",
        "size_image",
        "x_pels_per_meter",
        "y_pels_per_meter",
        "clr_used",
        "clr_important",
    ),
    (0,) * 11,
)

WaveFormatEx = StructProcessor(
    "WAVEFORMATEX",
    "HHIIHHH",
    (
        "format_tag",
        "channels",
        "samples_per_sec",
        "avg_bytes_per_sec",
        "block_align",
        "bits_per_sample",
        "size",
    ),
    (0,) * 7,
)

WaveFormatExtensible = StructProcessor(
    "WAVEFORMATEXTENSIBLE",
    "HHIIHHHHI16s",
    (
        "format_tag",
        "channels",
        "samples_per_sec",
        "avg_bytes_per_sec",
        "block_align",
        "bits_per_sample",
        "size",
        "samples",
        "channel_mask",
        "sub_format",
    ),
    (*((0,) * 9), "\0" * 16),
)


def decode_strf(data, prev_chunk):
    fcc_type = prev_chunk[1].fcc_type
    if fcc_type == "vids":  # BITMAPINFO
        return BitmapInfoHeaer(data)
    elif fcc_type == "auds":  # WAVEFORMATEX
        chunk = WaveFormatEx.unpack_from(data)
        if chunk.format_tag == WAVE_FORMAT_EXTENSIBLE:
            chunk = WaveFormatExtensible.unpack(data)
        return chunk
    return data


def decode_zstr(data, prev_chunk):
    return data[:-1].decode("utf-8")


VideoPropHeader = StructProcessor(
    "VPRP",
    "9I",
    (
        "video_format_token",
        "video_standard",
        "vertical_refresh_rate",
        "h_total_in_t",
        "v_total_in_lines",
        "frame_aspect_ratio",
        "frame_width_in_pixels",
        "frame_height_in_lines",
        "field_per_frame",
        "field_info",
    ),
    (((0,) * 9), ()),
)

VPRP_VideoField = StructProcessor(
    "VPRP_VIDEO_FIELD_DESC",
    "8I",
    (
        "compressed_bm_height",
        "compressed_bm_width",
        "valid_bm_height",
        "valid_bm_width",
        "valid_bm_x_offset",
        "valid_bm_y_offset",
        "video_x_offset_in_t",
        "video_y_valid_start_line",
    ),
    ((0,) * 8),
)


def decode_vprp(data, prev_chunk):
    chunk = VideoPropHeader.unpack_from(data)
    offset = VideoPropHeader.size
    ninfo = VPRP_VideoField.size

    return chunk._replace(
        field_info=(
            VPRP_VideoField.unpack_from(data, i)
            for i in range(offset, offset + ninfo * chunk.field_per_frame, ninfo)
        )
    )


ODMLExtendedAVIHeader = StructProcessor("dmlh", "8I", ("total_frames",), (0,))


def decode_dmlh(data, prev_chunk):
    return ODMLExtendedAVIHeader.unpack(data)


decoders = dict(
    avih=decode_avih,
    strh=decode_strh,
    strf=decode_strf,
    strn=decode_zstr,
    vprp=decode_vprp,
    ISMP=decode_zstr,
    IDIT=decode_zstr,
    IARL=decode_zstr,
    IART=decode_zstr,
    ICMS=decode_zstr,
    ICMT=decode_zstr,
    ICOP=decode_zstr,
    ICRD=decode_zstr,
    ICRP=decode_zstr,
    IDIM=decode_zstr,
    IDPI=decode_zstr,
    IENG=decode_zstr,
    IGNR=decode_zstr,
    IKEY=decode_zstr,
    ILGT=decode_zstr,
    IMED=decode_zstr,
    INAM=decode_zstr,
    IPLT=decode_zstr,
    IPRD=decode_zstr,
    ISBJ=decode_zstr,
    ISFT=decode_zstr,
    ISHP=decode_zstr,
    ISRC=decode_zstr,
    ISRF=decode_zstr,
    ITCH=decode_zstr,
)

# tcdl
# time
# indx


def next_chunk(f, resolve_sublist=False, prev_item=None):
    b = f.read(4)
    if not len(b):
        return None

    id = b.decode("utf-8")
    datasize = int.from_bytes(f.read(4), byteorder="little", signed=False)
    size = datasize + 1 if datasize % 2 else datasize

    if id == "LIST":
        data = f.read(4).decode("utf-8")
        listsize = size - 4
        if resolve_sublist or data == "INFO":
            items = []
            while listsize:
                item = next_chunk(f, resolve_sublist, prev_item)
                if item[0] != "JUNK":
                    items.append(item[:-1])
                    prev_item = item
                listsize -= item[2] + 8
            id = data
            data = items
    elif id == "JUNK":
        f.read(size)
        if size > datasize:
            f.read(size - datasize)
        data = None
    else:
        data = f.read(datasize)
        if size > datasize:
            f.read(size - datasize)
        decoder = decoders.get(id, None)
        if decoder:
            data = decoder(data, prev_item)
    return id, data, size


fcc_types = dict(vids="v", auds="a", txts="s")  # , mids="midi")


def read_header(f, use_ya8):
    f.read(12)  # ignore the 'RIFF SIZE AVI ' entry of the top level chunk
    hdr = next_chunk(f, resolve_sublist=True)[1]
    ch = next_chunk(f)
    while ch and ch[0] != "LIST" and ch[1] != "movi":
        ch = next_chunk(f)

    def get_stream_info(i, data, use_ya8):
        strh = data[0][1]
        strf = data[1][1]
        type = fcc_types[strh.fcc_type]  # raises if not valid type
        info = dict(index=i, type=type)
        if type == fcc_types["vids"]:
            info.frame_rate = fractions.Fraction(strh.rate, strh.scale)
            info["width"] = strf.width
            info["height"] = abs(strf.height)
            bpp = strf.bit_count
            compression = strf.compression
            # force unsupported pixel formats
            info["pix_fmt"] = (
                {"Y800": "gray", "RGBA": "rgba"}.get(compression, None)
                if isinstance(compression, str)
                else (compression, bpp)
                if compression
                else "rgb48le"
                if bpp == 48
                else "grayf32le"
                if bpp == 32
                else "rgb24"
                if bpp == 24
                else "ya8"
                if use_ya8
                else "gray16le"
                if bpp == 16
                else None
            )
            vprp = next((d[1] for d in data[2:] if d[0] == "vprp"), None)
            info["dar"] = vprp.frame_aspect_ratio if vprp else None
        elif type == fcc_types["auds"]:  #'audio'
            info["sample_rate"] = strf.samples_per_sec
            info["channels"] = strf.channels

            strf_format = (
                strf.format_tag
                if strf.format_tag != WAVE_FORMAT_EXTENSIBLE
                else strf.sub_format,
                strf.bits_per_sample,
            )

            info["sample_fmt"] = {
                (WAVE_FORMAT_PCM, 8): "u8",
                (WAVE_FORMAT_PCM, 16): "s16",
                (WAVE_FORMAT_PCM, 32): "s32",
                (WAVE_FORMAT_IEEE_FLOAT, 32): "flt",
                (WAVE_FORMAT_IEEE_FLOAT, 64): "dbl",
            }.get(strf_format, strf_format)
            # TODO: if need arises, resolve more formats, need to include codec names though
        return info

    strl = [hdr[i][1] for i in range(len(hdr)) if hdr[i][0] == "strl"]
    return [get_stream_info(i, strl[i], use_ya8) for i in range(len(strl))]


def read_frame(f):
    chunk = next_chunk(f)
    if chunk is None or not re.match(r"ix..|\d{2}(?:wb|db|dc|tx|pc)", chunk[0]):
        return None
    hdr = chunk[0]
    return (int(hdr[:2]) if hdr[2:] in ("wb", "db", "dc", "tx") else None), chunk[1]


#######################################################################################################


class AviReader:
    def __init__(self, use_ya8=False):
        self._f = None
        self.use_ya8 = use_ya8  #: bool: True to interpret 16-bit pixel as 'ya8' pix_fmt, False for 'gray16le'

        self.ready = False  #:bool: True if AVI headers has been processed
        self.streams = None  #:dict: Stream headers keyed by stream id (int key)
        self.converters = None  #:dict : Stream to numpy ndarray conversion functions keyed by stream id

    def start(self, f):
        self._f = f
        hdr = read_header(self._f, self.use_ya8)

        cnt = {"v": 0, "a": 0, "s": 0}

        def set_stream_info(hdr):
            st_type = hdr["type"]
            id = cnt[st_type]
            cnt[st_type] += 1
            if st_type == "v":
                _, ncomp, dtype, _ = get_pixel_config(hdr["pix_fmt"])
                shape = (hdr["height"], hdr["width"], ncomp)
            elif st_type == "a":
                _, dtype = get_audio_format(hdr["sample_fmt"])
                shape = (hdr["channels"],)
            return {
                "spec": spec_stream(id, st_type),
                "shape": shape,
                "dtype": dtype,
                **hdr,
            }

        self.streams = {v["index"]: set_stream_info(v) for v in hdr}

        def get_converter(stream):
            return lambda b: np.frombuffer(b, dtype=stream["dtype"]).reshape(
                -1, *stream["shape"]
            )

        self.converters = {k: get_converter(v) for k, v in self.streams.items()}

        self.ready = True

    def __next__(self):
        i = d = None
        while i is None:  # None if unknown frame format, skip
            frame = read_frame(self._f)
            if frame is None:  # likely eof
                raise StopIteration
            i, d = frame
        return i, self.converters[i](d)

    def __iter__(self):
        return self


# (
#     "hdrl",
#     [
#         (
#             "avih",
#             {
#                 "micro_sec_per_frame": 66733,
#                 "max_bytes_per_sec": 3974198,
#                 "padding_granularity": 0,
#                 "flags": 0,
#                 "total_frames": 0,
#                 "initial_frames": 0,
#                 "streams": 2,
#                 "suggested_buffer_size": 1048576,
#                 "width": 352,
#                 "height": 240,
#             },
#         ),
#         (
#             "strl",
#             [
#                 (
#                     "strh",
#                     {
#                         "fcc_type": "vids",
#                         "fcc_handler": "\x00\x00\x00\x00",
#                         "flags": 0,
#                         "priority": 0,
#                         "language": 0,
#                         "initial_frames": 0,
#                         "scale": 200,
#                         "rate": 2997,
#                         "start": 0,
#                         "length": 1073741824,
#                         "suggested_buffer_size": 1048576,
#                         "quality": 4294967295,
#                         "sample_size": 0,
#                         "frame_left": 0,
#                         "frame_top": 0,
#                         "frame_right": 352,
#                         "frame_bottom": 240,
#                     },
#                 ),
#                 (
#                     "strf",
#                     {
#                         "size": 40,
#                         "width": 352,
#                         "height": -240,
#                         "planes": 1,
#                         "bit_count": 24,
#                         "compression": "rgb24",
#                         "size_image": 253440,
#                         "x_pels_per_meter": 0,
#                         "y_pels_per_meter": 0,
#                         "clr_used": 0,
#                         "clr_important": 0,
#                     },
#                 ),
#                 (
#                     "vprp",
#                     {
#                         "video_format_token": 0,
#                         "video_standard": 0,
#                         "vertical_refresh_rate": 15,
#                         "h_total_in_t": 352,
#                         "v_total_in_lines": 240,
#                         "frame_aspect_ratio": Fraction(15, 22),
#                         "frame_width_in_pixels": 352,
#                         "frame_height_in_lines": 240,
#                         "field_per_frame": 1,
#                         "field_info": (
#                             {
#                                 "compressed_bm_height": 240,
#                                 "compressed_bm_width": 352,
#                                 "valid_bm_height": 240,
#                                 "valid_bm_width": 352,
#                                 "valid_bmx_offset": 0,
#                                 "valid_bmy_offset": 0,
#                                 "video_x_offset_in_t": 0,
#                                 "video_y_valid_start_line": 0,
#                             },
#                         ),
#                     },
#                 ),
#             ],
#         ),
#         (
#             "strl",
#             [
#                 (
#                     "strh",
#                     {
#                         "fcc_type": "auds",
#                         "fcc_handler": "\x01\x00\x00\x00",
#                         "flags": 0,
#                         "priority": 0,
#                         "language": 0,
#                         "initial_frames": 0,
#                         "scale": 1,
#                         "rate": 44100,
#                         "start": 0,
#                         "length": 1073741824,
#                         "suggested_buffer_size": 12288,
#                         "quality": 4294967295,
#                         "sample_size": 4,
#                         "frame_left": 0,
#                         "frame_top": 0,
#                         "frame_right": 0,
#                         "frame_bottom": 0,
#                     },
#                 ),
#                 (
#                     "strf",
#                     {
#                         "format_tag": 1,
#                         "channels": 2,
#                         "samples_per_sec": 44100,
#                         "avg_bytes_per_sec": 176400,
#                         "block_align": 4,
#                         "bits_per_sample": 16,
#                     },
#                 ),
#             ],
#         ),
#     ],
#     368,
# )
