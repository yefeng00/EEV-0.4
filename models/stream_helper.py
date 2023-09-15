# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - width - padding_left
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_rounded_q(q_scale):
    q_scale = np.clip(q_scale, 0.01, 655.)
    q_index = int(np.round(q_scale * 100))
    q_scale = q_index / 100
    return q_scale, q_index


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError('Invalid file.')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        header = read_uints(f, 2)
        y_length = header[0]
        z_length = header[1]
        y = read_bytes(f, y_length)
        z = read_bytes(f, z_length)
        shape = height, width
    return [[y], [z]], shape


def encode_i(string, shape, output):
    with Path(output).open("wb") as f:
        h, w = shape
        write_uints(f, (h, w,))
        y = string[0][0]
        z = string[1][0]
        string_length_y = len(y)
        string_length_z = len(z)
        write_uints(f, (string_length_y, string_length_z))
        write_bytes(f, y)
        write_bytes(f, z)

def encode_p(result, output):
    with Path(output).open("wb") as f:
        s = {"byte_stream_mv_hyper_hat", "byte_stream_mv_feature_hat", "byte_stream_y_hat", "byte_stream_z_hat"}
        for i in s:
            byte_stream, h, w, miny, maxy = get_decoder_inf(result[i])
            write_uints(f, (h, w, -miny, maxy,))
            string_length = len(byte_stream)
            write_uints(f, (string_length,))
            write_bytes(f, byte_stream)

def get_decoder_inf(result):
        byte_stream = result["byte_stream"]
        h = result["h"]
        w = result["w"]
        miny = result["min"]
        maxy = result["max"]
        return byte_stream, h, w, miny, maxy

def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        s = {"byte_stream_mv_hyper_hat", "byte_stream_mv_feature_hat", "byte_stream_y_hat", "byte_stream_z_hat"}
        output = {}
        for i in s:
            header = read_uints(f, 4)
            h = header[0]
            w = header[1]
            miny = -header[2]
            maxy = header[3]
            header = read_uints(f, 1)
            string_length = header[0]
            byte_stream = read_bytes(f, string_length)
            output0 = {
                "h": h,
                "w": w,
                "miny": miny,
                "maxy": maxy,
                "byte_stream": byte_stream,
            }
            output[i] = output0
        
        return output