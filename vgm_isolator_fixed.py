#!/usr/bin/env python3
"""
VGM Instrument Isolator - Fixed Version
Separates individual instruments within FM channels by tracking instrument changes
and note events, creating isolated files for each unique instrument instance.

FIXES:
1. Fixed missing get_current_frequency method
2. Fixed create_vgm_file method call
3. Improved channel mapping for key on/off events
4. Added proper DAC enable/disable tracking
5. Fixed indentation and method structure
6. Improved error handling and debugging output
"""

import struct
import os
import sys
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

@dataclass
class InstrumentState:
    """Represents the complete state of an FM instrument"""
    operators: Dict[int, Dict[int, int]]  # operator -> {register: value}
    channel_regs: Dict[int, int]          # channel registers
    active: bool = False
    last_note_time: int = 0
    
    def get_hash(self) -> str:
        """Generate a hash representing this instrument configuration"""
        state_str = ""
        for op in sorted(self.operators.keys()):
            for reg in sorted(self.operators[op].keys()):
                state_str += f"{reg:02X}:{self.operators[op][reg]:02X},"
        for reg in sorted(self.channel_regs.keys()):
            state_str += f"{reg:02X}:{self.channel_regs[reg]:02X},"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

@dataclass
class NoteEvent:
    """Represents a note on/off event"""
    time: int
    channel: int
    note_on: bool
    instrument_hash: str
    frequency: Optional[int] = None

@dataclass
class DACEvent:
    """Represents a DAC sample event"""
    time: int
    sample_value: int
    sample_group: str  # Group similar sample values together

class VGMInstrumentProcessor:
    def __init__(self):
        # YM2612 register mappings
        self.OPERATOR_REGS = {
            0x30: "DT_MUL", 0x40: "TL", 0x50: "KS_AR",
            0x60: "AM_DR", 0x70: "SR", 0x80: "SL_RR", 0x90: "SSG_EG"
        }
        self.CHANNEL_REGS = {
            0xA0: "FREQ_LOW", 0xA4: "FREQ_HIGH",
            0xB0: "FB_ALG", 0xB4: "LR_AMS_FMS"
        }
        self.channel_instruments: Dict[int, InstrumentState] = {}
        self.discovered_instruments: Dict[str, InstrumentState] = {}
        self.note_events: List[NoteEvent] = []
        self.dac_events: List[DACEvent] = []
        self.discovered_dac_samples: Dict[str, List[int]] = {}
        self.current_time = 0
        self.dac_enabled = False
        self.debug = True  # Enable debug output

        for ch in range(6):
            self.channel_instruments[ch] = InstrumentState({}, {})

    def read_vgm_header(self, data: bytes) -> Dict:
        """Parse VGM file header"""
        if len(data) < 64 or data[:4] != b'Vgm ':
            raise ValueError("Invalid VGM file")
        header = {
            'version':     struct.unpack('<I', data[8:12])[0],
            'ym2612_clock':struct.unpack('<I', data[44:48])[0],
            'data_offset': struct.unpack('<I', data[52:56])[0]
        }
        header['data_start'] = header['data_offset'] + 52 if header['data_offset'] else 64
        if self.debug:
            print(f"VGM Version: 0x{header['version']:08X}")
            print(f"YM2612 Clock: {header['ym2612_clock']}")
            print(f"Data starts at: {header['data_start']}")
        return header

    def get_channel_from_port_reg(self, port: int, reg: int) -> Optional[int]:
        """Determine which channel a register write affects"""
        if port == 0x52:
            if reg == 0x28: return None
            channel = ((reg - 0xA0) % 4) if reg >= 0xA0 else (reg & 0x03)
            return channel if channel < 3 else None
        elif port == 0x53:
            if reg == 0x28: return None
            channel = (((reg - 0xA0) % 4) + 3) if reg >= 0xA0 else ((reg & 0x03) + 3)
            return channel if channel < 6 else None
        return None

    def get_operator_from_reg(self, reg: int) -> Optional[int]:
        if reg < 0x30: return None
        base = reg & 0xF0
        if base in self.OPERATOR_REGS:
            return (reg & 0x0C) >> 2
        return None

    def update_instrument_state(self, channel: int, reg: int, value: int, port: int):
        if channel is None or channel < 0 or channel > 5:
            return
        inst = self.channel_instruments[channel]
        op = self.get_operator_from_reg(reg)
        if op is not None:
            base = reg & 0xF0
            inst.operators.setdefault(op, {})[base] = value
            if self.debug and reg == 0x40:
                print(f"CH{channel} OP{op} TL={value:02X}")
        elif reg in self.CHANNEL_REGS:
            inst.channel_regs[reg] = value
            if self.debug and reg == 0xB0:
                print(f"CH{channel} FB/ALG={value:02X}")

    def handle_key_onoff(self, value: int):
        raw = value & 0x07
        if raw > 6 or raw == 3: return
        channel = raw if raw <= 2 else raw - 1
        key_on = (value & 0xF0) != 0
        if self.debug:
            print(f"Key {'ON' if key_on else 'OFF'} CH{channel} at {self.current_time}")
        inst = self.channel_instruments[channel]
        if key_on:
            inst.active = True
            inst.last_note_time = self.current_time
            h = inst.get_hash()
            if h not in self.discovered_instruments:
                self.discovered_instruments[h] = InstrumentState(
                    operators={o: r.copy() for o,r in inst.operators.items()},
                    channel_regs=inst.channel_regs.copy()
                )
                if self.debug: print(f"New instrument: {h}")
            freq = self.get_current_frequency(channel)
            self.note_events.append(NoteEvent(self.current_time, channel, True, h, freq))
        else:
            if inst.active:
                h = inst.get_hash()
                self.note_events.append(NoteEvent(self.current_time, channel, False, h))
            inst.active = False

    def get_current_frequency(self, channel: int) -> Optional[int]:
        inst = self.channel_instruments[channel]
        lo = inst.channel_regs.get(0xA0, 0)
        hi = inst.channel_regs.get(0xA4, 0)
        return (hi << 8) | lo if (lo or hi) else None

    def group_dac_samples(self, values: List[int], window_size: int = 10) -> str:
        if not values: return "empty"
        avg = sum(values)/len(values)
        mx, mn = max(values), min(values)
        rng = mx - mn
        if rng < 20:   grp = "sustained"
        elif mx > 200: grp = "loud"
        elif avg < 50: grp = "quiet"
        else:          grp = "mid"
        cat = int(avg/32)
        return f"{grp}_{cat:02d}"

    def handle_dac_sample(self, value: int):
        if not self.dac_enabled: return
        if not hasattr(self, '_dac_seq'):
            self._dac_seq = []; self._dac_start = self.current_time
        self._dac_seq.append(value)
        if len(self._dac_seq) >= 5:
            g = self.group_dac_samples(self._dac_seq)
            self.discovered_dac_samples.setdefault(g, [])
            self.dac_events.append(DACEvent(self._dac_start, value, g))
            self._dac_seq = []

    def finalize_dac_sequence(self):
        if hasattr(self, '_dac_seq') and self._dac_seq:
            g = self.group_dac_samples(self._dac_seq)
            self.discovered_dac_samples.setdefault(g, [])
            self.dac_events.append(DACEvent(self._dac_start, self._dac_seq[-1], g))
            self._dac_seq = []

    def analyze_instruments(self, data: bytes, header: Dict) -> Tuple[Dict[str, List[NoteEvent]], Dict[str, List[DACEvent]]]:
        pos = header['data_start']; self.current_time = 0
        print("Analyzing instruments and DAC samples...")
        while pos < len(data):
            cmd = data[pos]
            if   cmd == 0x66: break
            elif cmd == 0x52 and pos+2 < len(data):
                r,v = data[pos+1], data[pos+2]
                if   r==0x28: self.handle_key_onoff(v)
                else:
                    ch = self.get_channel_from_port_reg(0x52, r)
                    if ch is not None: self.update_instrument_state(ch, r, v, 0x52)
                pos+=3
            elif cmd == 0x53 and pos+2 < len(data):
                r,v = data[pos+1], data[pos+2]
                if   r==0x28: self.handle_key_onoff(v)
                elif r==0x2A:
                    self.dac_enabled = (v & 0x80)!=0
                    if self.debug: print(f"DAC {'EN' if self.dac_enabled else 'DIS'}")
                else:
                    ch = self.get_channel_from_port_reg(0x53, r)
                    if ch is not None: self.update_instrument_state(ch, r, v, 0x53)
                pos+=3
            elif cmd == 0x2A and pos+1 < len(data):
                self.handle_dac_sample(data[pos+1]); pos+=2
            elif cmd & 0xF0 == 0x80:
                self.handle_dac_sample(cmd&0x0F); self.current_time+=1; pos+=1
            elif cmd == 0x61 and pos+2 < len(data):
                wt = struct.unpack('<H', data[pos+1:pos+3])[0]
                self.current_time += wt; self.finalize_dac_sequence(); pos+=3
            elif cmd == 0x62:
                self.current_time += 735; self.finalize_dac_sequence(); pos+=1
            elif cmd == 0x63:
                self.current_time += 882; self.finalize_dac_sequence(); pos+=1
            elif cmd & 0xF0 == 0x70:
                self.current_time += (cmd&0x0F)+1; self.finalize_dac_sequence(); pos+=1
            else:
                if self.debug: print(f"Skip cmd 0x{cmd:02X} @ {pos}")
                pos+=1
        self.finalize_dac_sequence()

        inst_events = defaultdict(list)
        for e in self.note_events: inst_events[e.instrument_hash].append(e)
        dac_events = defaultdict(list)
        for e in self.dac_events:    dac_events[e.sample_group].append(e)

        print(f"Discovered {len(self.discovered_instruments)} FM inst, {len(self.discovered_dac_samples)} DAC groups")
        print(f"Recorded {len(self.note_events)} note evts, {len(self.dac_events)} DAC evts")
        return dict(inst_events), dict(dac_events)

    def create_vgm_file(self, original_data: bytes, processed_data: bytes, output_path: str):
        header_data = bytearray(original_data[:64])
        struct.pack_into('<I', header_data, 52, len(header_data)-52)
        with open(output_path, 'wb') as f:
            f.write(header_data)
            f.write(processed_data)

    def create_instrument_vgm(self, original_data: bytes, header: Dict,
                              instrument_hash: str, events: List[NoteEvent],
                              output_path: str):
        pos = header['data_start']; out = bytearray(); t = 0; active = set()
        inst = self.discovered_instruments[instrument_hash]
        print(f"Creating VGM for inst {instrument_hash}...")
        # initial reg dumps
        for ch in range(6):
            for op, regs in inst.operators.items():
                for br,val in regs.items():
                    reg = br | (op<<2)
                    out.extend([0x52 if ch<3 else 0x53, reg, val])
            for r,v in inst.channel_regs.items():
                out.extend([0x52 if ch<3 else 0x53, r, v])
        ev_by_t = defaultdict(list)
        for e in events: ev_by_t[e.time].append(e)

        while pos < len(original_data):
            cmd = original_data[pos]
            if cmd == 0x66:
                out.append(cmd); break
            cur = ev_by_t.get(t, [])
            if cmd in (0x52, 0x53) and pos+2< len(original_data):
                r,v = original_data[pos+1], original_data[pos+2]
                inc = False
                if r==0x28:
                    raw = v&0x07
                    ch  = raw if raw<=2 else raw-1
                    for e in cur:
                        if e.channel==ch and e.instrument_hash==instrument_hash:
                            inc = True
                            if e.note_on: active.add(ch)
                            else: active.discard(ch)
                            break
                else:
                    ch = self.get_channel_from_port_reg(cmd, r)
                    if ch in active: inc=True
                if inc: out.extend([cmd,r,v])
                pos+=3
            elif cmd==0x61 and pos+2< len(original_data):
                wt=struct.unpack('<H', original_data[pos+1:pos+3])[0]
                t+=wt; out.extend([cmd, original_data[pos+1], original_data[pos+2]]); pos+=3
            elif cmd in (0x62,0x63) or (cmd&0xF0)==0x70:
                if cmd==0x62: t+=735
                elif cmd==0x63: t+=882
                else: t+=(cmd&0x0F)+1
                out.append(cmd); pos+=1
            else:
                pos+=1

        self.create_vgm_file(original_data, bytes(out), output_path)

    def create_dac_vgm(self, original_data: bytes, header: Dict,
                       sample_group: str, events: List[DACEvent],
                       output_path: str):
        pos = header['data_start']; out=bytearray(); t=0
        print(f"Creating VGM for DAC group {sample_group}...")
        ev_by_t = defaultdict(list)
        for e in events: ev_by_t[e.time].append(e)

        while pos < len(original_data):
            cmd = original_data[pos]
            if cmd==0x66:
                out.append(cmd); break
            cur = ev_by_t.get(t, [])
            if cmd==0x53 and pos+2< len(original_data):
                r,v=original_data[pos+1],original_data[pos+2]
                if r==0x2A: out.extend([cmd,r,v])
                pos+=3
            elif cmd==0x2A and pos+1< len(original_data):
                v=original_data[pos+1]
                if any(e.sample_group==sample_group for e in cur):
                    out.extend([cmd,v])
                pos+=2
            elif (cmd&0xF0)==0x80:
                inc = any(e.sample_group==sample_group for e in cur)
                if inc: out.append(cmd)
                else:
                    wt=cmd&0x0F
                    out.extend([0x61, wt or 1, 0x00] if wt==0 else [0x70+wt-1])
                t+=1; pos+=1
            elif cmd==0x61 and pos+2< len(original_data):
                wt=struct.unpack('<H', original_data[pos+1:pos+3])[0]
                t+=wt; out.extend([cmd, original_data[pos+1], original_data[pos+2]]); pos+=3
            elif cmd in (0x62,0x63) or (cmd&0xF0)==0x70:
                if cmd==0x62: t+=735
                elif cmd==0x63: t+=882
                else: t+=(cmd&0x0F)+1
                out.append(cmd); pos+=1
            else:
                pos+=1

        self.create_vgm_file(original_data, bytes(out), output_path)

    def create_summary_report(self, output_dir: str, base_name: str,
                              instrument_events: Dict[str, List], dac_events: Dict[str, List]):
        """Create a summary report of discovered instruments and samples"""
        summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("VGM Instrument Isolation Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Source file: {base_name}\n\n")
            f.write(f"FM Instruments Found: {len(instrument_events)}\n")
            f.write("-"*30 + "\n")
            for i,(h,evs) in enumerate(instrument_events.items()):
                inst = self.discovered_instruments[h]
                note_ct = len([e for e in evs if e.note_on])
                f.write(f"Instrument {i+1:02d} ({h}):\n")
                f.write(f"  - Note events: {note_ct}\n")
                f.write(f"  - Operators: {len(inst.operators)}\n")
                f.write(f"  - Channel regs: {len(inst.channel_regs)}\n")
                if inst.channel_regs.get(0xB0):
                    fb_alg=inst.channel_regs[0xB0]
                    f.write(f"  - Algorithm: {fb_alg&0x07}, Feedback: {(fb_alg>>3)&0x07}\n")
                f.write("\n")
            f.write(f"DAC Sample Groups Found: {len(dac_events)}\n")
            f.write("-"*30 + "\n")
            for i,(g,evs) in enumerate(dac_events.items()):
                f.write(f"DAC Group {i+1:02d} ({g}):\n")
                f.write(f"  - Events: {len(evs)}\n")
                if evs:
                    vals = [e.sample_value for e in evs[:10]]
                    f.write(f"  - Sample values: {vals}\n")
                f.write("\n")
        print(f"Summary report saved to: {summary_file}")

    def isolate_instruments_complete(self, input_file: str, output_dir: str):
        """Complete version of isolate_instruments with summary report"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        os.makedirs(output_dir, exist_ok=True)
        with open(input_file, 'rb') as f:
            data = f.read()
        header = self.read_vgm_header(data)
        base = os.path.splitext(os.path.basename(input_file))[0]
        print(f"Processing {input_file}...")
        inst_ev, dac_ev = self.analyze_instruments(data, header)
        if not inst_ev and not dac_ev:
            print("No instruments or DAC samples found!"); return
        for i,(h,evs) in enumerate(inst_ev.items()):
            out = os.path.join(output_dir, f"{base}_fm_inst_{i+1:02d}_{h}.vgm")
            self.create_instrument_vgm(data, header, h, evs, out)
        for i,(g,evs) in enumerate(dac_ev.items()):
            out = os.path.join(output_dir, f"{base}_dac_{i+1:02d}_{g}.vgm")
            self.create_dac_vgm(data, header, g, evs, out)
        self.create_summary_report(output_dir, base, inst_ev, dac_ev)
        print("\nProcessing complete!")
        print(f"Files in: {output_dir} - {len(inst_ev)} FM files, {len(dac_ev)} DAC files, 1 summary")

def main():
    """Main function to process VGM files"""
    if len(sys.argv) < 2:
        print("Usage: python vgm_isolator_fixed.py <input_vgm_file> [output_directory]")
        sys.exit(1)
    inp = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else "./isolated_instruments"
    try:
        proc = VGMInstrumentProcessor()
        proc.isolate_instruments_complete(inp, outd)
    except Exception as e:
        print(f"Error processing VGM file: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
