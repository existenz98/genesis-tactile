# -----------------------------------------------------------------------------
# SPDX-License-Identifier: LicenseRef-YF-Research-NC-1.0
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# Licensed for academic research and non-commercial use only.
# Any Commercial Use (including production use or any use for commercial
# advantage) requires a separate written license from the copyright holder.
# See LICENSE-SRC-RESEARCH-NC for definitions and terms.
#
# Patent Notice:
#   No patent license is granted or implied. Users are responsible for
#   third-party patent clearance.
#
# Citation:
#   Please cite associated publications when available.
# -----------------------------------------------------------------------------


from __future__ import annotations
import torch
import torch.nn as nn

def norm_layer(c: int, kind: str = "group") -> nn.Module:
    """
    default is groupnorm with 8 or min(c,8) groups
    """
    if kind == "batch":
        return nn.BatchNorm2d(c)    
    g = min(8, c)
    return nn.GroupNorm(g, c)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch, norm),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        
        # Method 1. AvgPool2d
        #  is not good enough, better choice is 'strided conv + norm+SiLU'
        #self.pool = nn.AvgPool2d(2, 2)
        #self.conv = DoubleConv(in_ch, out_ch, norm)

        # Method 2. Conv2D
        # downsampling using Conv 3x3, stride=2, padding=1 -> protect  edge, and feeling
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False),   # floor(H/2)
            norm_layer(out_ch, norm),
            nn.SiLU(inplace=True),
            # feature extraction：2nd Conv 3x3
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_ch, norm),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Method 1.
        #return self.conv(self.pool(x))

        # Method 2.
        return self.block(x)
    
class Up(nn.Module):
    
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()

        # Method 1. ConvTranspose2d. ConvTranspose2d is not good enough, better choice is 'bilinear upsample + Conv3×3'
        #self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        # Method 2. Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv = DoubleConv(in_ch, out_ch, norm)

    def _match_to(self, x, ref):
        import torch.nn.functional as F

        dh = ref.size(2) - x.size(2)
        dw = ref.size(3) - x.size(3)
        # pad right/bottom if x is smaller
        if dh > 0 or dw > 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
        # crop right/bottom if x is larger
        if dh < 0 or dw < 0:
            x = x[:, :, :ref.size(2), :ref.size(3)]
        return x
    
    def forward(self, x, skip):
        x = self.up(x)

        # before cat, pad if needed to align size
        x = self._match_to(x, skip)

        #dh = skip.size(2) - x.size(2)
        #dw = skip.size(3) - x.size(3)
        #if dh != 0 or dw != 0:
        #    x = nn.functional.pad(x, (0, max(0, dw), 0, max(0, dh)))
        #x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)

class UNetBasic(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 3,
                 base_ch: list[int] = [16, 32, 64, 128],
                 norm: str = "group", nonneg_tz_head: bool = False):
        super().__init__()
        c1, c2, c3, c4 = base_ch
        self.inc = DoubleConv(in_ch, c1, norm)
        self.d1  = Down(c1, c2, norm)
        self.d2  = Down(c2, c3, norm)
        self.d3  = Down(c3, c4, norm)
        self.u1  = Up(c4, c3, norm)
        self.u2  = Up(c3, c2, norm)
        self.u3  = Up(c2, c1, norm)
        self.outc = nn.Conv2d(c1, out_ch, kernel_size=1)
        self.nonneg_tz_head = nonneg_tz_head

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        u1 = self.u1(x4, x3)
        u2 = self.u2(u1, x2)
        u3 = self.u3(u2, x1)
        y  = self.outc(u3)    # (B,3,H,W) in order [tz, tx, ty]
        if self.nonneg_tz_head:
            tz = nn.functional.softplus(y[:, 0:1, ...], beta=1.0)
            y  = torch.cat([tz, y[:, 1:2, ...], y[:, 2:3, ...]], dim=1)
        return y
    
