#!/usr/bin/env python3
"""
VGM Instrument Isolator
Separates individual instruments within FM channels by tracking instrument changes
and note events, creating isolated files for each unique instrument instance.
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
        # Create a sorted representation of the instrument state
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
            # DT/MUL, TL, KS/AR, AM/DR, SR, SL/RR, SSG-EG
            0x30: "DT_MUL", 0x40: "TL", 0x50: "KS_AR", 
            0x60: "AM_DR", 0x70: "SR", 0x80: "SL_RR", 0x90: "SSG_EG"
        }
        
        self.CHANNEL_REGS = {
            0xA0: "FREQ_LOW", 0xA4: "FREQ_HIGH", 
            0xB0: "FB_ALG", 0xB4: "LR_AMS_FMS"
        }
        
        # Track instrument states for each channel
        self.channel_instruments: Dict[int, InstrumentState] = {}
        self.discovered_instruments: Dict[str, InstrumentState] = {}
        self.note_events: List[NoteEvent] = []
        self.dac_events: List[DACEvent] = []
        self.discovered_dac_samples: Dict[str, List[int]] = {}  # sample_group -> [values]
        self.current_time = 0
        
        # Initialize channel states
        for ch in range(6):
            self.channel_instruments[ch] = InstrumentState({}, {})

    def read_vgm_header(self, data: bytes) -> Dict:
        """Parse VGM file header"""
        if len(data) < 64 or data[:4] != b'Vgm ':
            raise ValueError("Invalid VGM file")
        
        header = {}
        header['version'] = struct.unpack('<I', data[8:12])[0]
        header['ym2612_clock'] = struct.unpack('<I', data[44:48])[0]
        header['data_offset'] = struct.unpack('<I', data[52:56])[0]
        header['data_start'] = header['data_offset'] + 52 if header['data_offset'] != 0 else 64
        return header

    def get_channel_from_port_reg(self, port: int, reg: int) -> Optional[int]:
        """Determine which channel a register write affects"""
        if port == 0x52:  # Port 0 (channels 0-2)
            if reg == 0x28:  # Key on/off affects all channels
                return None  # Special case, handle separately
            # Extract channel from register
            if reg >= 0xA0:
                return (reg - 0xA0) % 4 if (reg - 0xA0) % 4 < 3 else None
            else:
                return (reg & 0x03) if (reg & 0x03) < 3 else None
        elif port == 0x53:  # Port 1 (channels 3-5)
            if reg == 0x28:
                return None
            if reg >= 0xA0:
                return ((reg - 0xA0) % 4) + 3 if (reg - 0xA0) % 4 < 3 else None
            else:
                return (reg & 0x03) + 3 if (reg & 0x03) < 3 else None
        return None

    def get_operator_from_reg(self, reg: int) -> Optional[int]:
        """Get operator number (0-3) from register"""
        if reg < 0x30:
            return None
        base_reg = (reg & 0xF0)
        if base_reg in self.OPERATOR_REGS:
            return (reg & 0x0C) >> 2
        return None

    def update_instrument_state(self, channel: int, reg: int, value: int, port: int):
        """Update the instrument state for a channel"""
        if channel is None or channel < 0 or channel > 5:
            return
        
        instrument = self.channel_instruments[channel]
        
        # Handle operator registers
        operator = self.get_operator_from_reg(reg)
        if operator is not None:
            base_reg = reg & 0xF0
            if operator not in instrument.operators:
                instrument.operators[operator] = {}
            instrument.operators[operator][base_reg] = value
        
        # Handle channel registers
        elif reg in self.CHANNEL_REGS:
            instrument.channel_regs[reg] = value

    def handle_key_onoff(self, value: int):
        """Handle key on/off register (0x28)"""
        channel = value & 0x07
        if channel > 5:
            return
        
        # Adjust channel number for port mapping
        if channel >= 4:
            channel = channel - 1  # Channels 4,5,6 -> 3,4,5
        
        key_on = (value & 0xF0) != 0
        
        instrument = self.channel_instruments[channel]
        
        if key_on:
            # Note on - capture current instrument state
            instrument.active = True
            instrument.last_note_time = self.current_time
            
            # Generate instrument hash and store if new
            inst_hash = instrument.get_hash()
            if inst_hash not in self.discovered_instruments:
                # Deep copy the instrument state
                new_inst = InstrumentState(
                    operators={op: regs.copy() for op, regs in instrument.operators.items()},
                    channel_regs=instrument.channel_regs.copy()
                )
                self.discovered_instruments[inst_hash] = new_inst
            
            # Record note event
            freq = self.get_current_frequency(channel)
            self.note_events.append(NoteEvent(
                time=self.current_time,
                channel=channel,
                note_on=True,
                instrument_hash=inst_hash,
                frequency=freq
            ))
        else:
            # Note off
            if instrument.active:
                self.note_events.append(NoteEvent(
                    time=self.current_time,
                    channel=channel,
                    note_on=False,
                    instrument_hash=instrument.get_hash()
                ))
            instrument.active = False

    def group_dac_samples(self, sample_values: List[int], window_size: int = 10) -> str:
        """Group similar DAC sample values together to identify instruments"""
        if not sample_values:
            return "empty"
        
        # For drum samples, group by value ranges or patterns
        avg_value = sum(sample_values) / len(sample_values)
        max_value = max(sample_values)
        min_value = min(sample_values)
        value_range = max_value - min_value
        
        # Create a signature based on characteristics
        if value_range < 20:  # Low variation - might be a sustained sound
            group_type = "sustained"
        elif max_value > 200:  # High amplitude - likely kick or snare
            group_type = "loud"
        elif avg_value < 50:  # Low amplitude - likely hi-hat or cymbal
            group_type = "quiet" 
        else:
            group_type = "mid"
        
        # Sub-categorize by average value ranges
        avg_category = int(avg_value / 32)  # 0-7 categories
        
        return f"{group_type}_{avg_category:02d}"

    def handle_dac_sample(self, value: int):
        """Handle DAC sample data"""
        # Track consecutive DAC samples to group them
        if not hasattr(self, '_current_dac_sequence'):
            self._current_dac_sequence = []
            self._dac_sequence_start = self.current_time
        
        self._current_dac_sequence.append(value)
        
        # If we have a sequence of samples, group them
        if len(self._current_dac_sequence) >= 5:  # Minimum sequence length
            sample_group = self.group_dac_samples(self._current_dac_sequence)
            
            # Store the sample group
            if sample_group not in self.discovered_dac_samples:
                self.discovered_dac_samples[sample_group] = []
            
            # Record this DAC event
            self.dac_events.append(DACEvent(
                time=self._dac_sequence_start,
                sample_value=value,
                sample_group=sample_group
            ))
            
            # Reset for next sequence
            self._current_dac_sequence = []

    def finalize_dac_sequence(self):
        """Finalize any remaining DAC sequence"""
        if hasattr(self, '_current_dac_sequence') and self._current_dac_sequence:
            sample_group = self.group_dac_samples(self._current_dac_sequence)
            if sample_group not in self.discovered_dac_samples:
                self.discovered_dac_samples[sample_group] = []
            
            self.dac_events.append(DACEvent(
                time=self._dac_sequence_start,
                sample_value=self._current_dac_sequence[-1],
                sample_group=sample_group
            ))
        """Get current frequency setting for a channel"""
        instrument = self.channel_instruments[channel]
        freq_low = instrument.channel_regs.get(0xA0, 0)
        freq_high = instrument.channel_regs.get(0xA4, 0)
        return (freq_high << 8) | freq_low if freq_low or freq_high else None

    def analyze_instruments(self, data: bytes, header: Dict) -> Tuple[Dict[str, List[NoteEvent]], Dict[str, List[DACEvent]]]:
        """Analyze VGM data to discover instruments and DAC samples"""
        pos = header['data_start']
        self.current_time = 0
        
        print("Analyzing instruments and DAC samples...")
        
        while pos < len(data):
            cmd = data[pos]
            
            if cmd == 0x66:  # End of data
                break
            elif cmd == 0x52:  # YM2612 port 0 write
                if pos + 2 >= len(data):
                    break
                reg, value = data[pos + 1], data[pos + 2]
                
                if reg == 0x28:  # Key on/off
                    self.handle_key_onoff(value)
                else:
                    channel = self.get_channel_from_port_reg(0x52, reg)
                    if channel is not None:
                        self.update_instrument_state(channel, reg, value, 0x52)
                
                pos += 3
            elif cmd == 0x53:  # YM2612 port 1 write
                if pos + 2 >= len(data):
                    break
                reg, value = data[pos + 1], data[pos + 2]
                
                if reg == 0x28:  # Key on/off
                    self.handle_key_onoff(value)
                elif reg == 0x2A:  # DAC enable/disable
                    # Track DAC state changes
                    pass
                else:
                    channel = self.get_channel_from_port_reg(0x53, reg)
                    if channel is not None:
                        self.update_instrument_state(channel, reg, value, 0x53)
                
                pos += 3
            elif cmd == 0x2A:  # DAC data write
                if pos + 1 >= len(data):
                    break
                value = data[pos + 1]
                self.handle_dac_sample(value)
                pos += 2
            elif cmd & 0xF0 == 0x80:  # DAC write + wait
                value = cmd & 0x0F
                self.handle_dac_sample(value)
                self.current_time += 1
                pos += 1
            elif cmd == 0x61:  # Wait n samples
                if pos + 2 >= len(data):
                    break
                wait_time = struct.unpack('<H', data[pos + 1:pos + 3])[0]
                self.current_time += wait_time
                # Finalize any DAC sequence on wait
                self.finalize_dac_sequence()
                pos += 3
            elif cmd == 0x62:  # Wait 735 samples
                self.current_time += 735
                self.finalize_dac_sequence()
                pos += 1
            elif cmd == 0x63:  # Wait 882 samples
                self.current_time += 882
                self.finalize_dac_sequence()
                pos += 1
            elif cmd & 0xF0 == 0x70:  # Wait 1-16 samples
                self.current_time += (cmd & 0x0F) + 1
                self.finalize_dac_sequence()
                pos += 1
            else:
                pos += 1
        
        # Finalize any remaining DAC sequence
        self.finalize_dac_sequence()
        
        # Group note events by instrument
        instrument_events = defaultdict(list)
        for event in self.note_events:
            instrument_events[event.instrument_hash].append(event)
        
        # Group DAC events by sample type
        dac_events = defaultdict(list)
        for event in self.dac_events:
            dac_events[event.sample_group].append(event)
        
        print(f"Discovered {len(self.discovered_instruments)} unique FM instruments")
        print(f"Discovered {len(self.discovered_dac_samples)} unique DAC sample groups")
        print(f"Recorded {len(self.note_events)} FM note events")
        print(f"Recorded {len(self.dac_events)} DAC events")
        
        return dict(instrument_events), dict(dac_events)

    def create_instrument_vgm(self, original_data: bytes, header: Dict, 
                            instrument_hash: str, events: List[NoteEvent], 
                            output_path: str):
        """Create a VGM file containing only the specified instrument"""
        pos = header['data_start']
        output_data = bytearray()
        current_time = 0
        active_notes = set()  # Track which channels have active notes for this instrument
        
        # Get the instrument definition
        instrument = self.discovered_instruments[instrument_hash]
        
        print(f"Creating VGM for instrument {instrument_hash}...")
        
        # Create event lookup for quick access
        events_by_time = defaultdict(list)
        for event in events:
            events_by_time[event.time].append(event)
        
        while pos < len(data):
            cmd = original_data[pos]
            
            # Check if we have events at current time
            current_events = events_by_time.get(current_time, [])
            
            if cmd == 0x66:  # End of data
                output_data.append(cmd)
                break
            elif cmd == 0x52:  # YM2612 port 0 write
                if pos + 2 >= len(data):
                    break
                reg, value = original_data[pos + 1], original_data[pos + 2]
                
                include_command = False
                
                if reg == 0x28:  # Key on/off
                    # Only include if it's for our instrument
                    for event in current_events:
                        if event.channel == (value & 0x07) and event.instrument_hash == instrument_hash:
                            include_command = True
                            if event.note_on:
                                active_notes.add(event.channel)
                            else:
                                active_notes.discard(event.channel)
                            break
                else:
                    # Include register writes for channels that have active notes
                    channel = self.get_channel_from_port_reg(0x52, reg)
                    if channel in active_notes:
                        include_command = True
                
                if include_command:
                    output_data.extend([cmd, reg, value])
                
                pos += 3
            elif cmd == 0x53:  # YM2612 port 1 write
                if pos + 2 >= len(data):
                    break
                reg, value = original_data[pos + 1], original_data[pos + 2]
                
                include_command = False
                
                if reg == 0x28:  # Key on/off
                    for event in current_events:
                        channel_mapped = event.channel
                        if channel_mapped >= 3:
                            channel_mapped = channel_mapped + 1  # Map back to register format
                        if channel_mapped == (value & 0x07) and event.instrument_hash == instrument_hash:
                            include_command = True
                            if event.note_on:
                                active_notes.add(event.channel)
                            else:
                                active_notes.discard(event.channel)
                            break
                else:
                    channel = self.get_channel_from_port_reg(0x53, reg)
                    if channel in active_notes:
                        include_command = True
                
                if include_command:
                    output_data.extend([cmd, reg, value])
                
                pos += 3
            elif cmd == 0x61:  # Wait n samples
                if pos + 2 >= len(data):
                    break
                wait_time = struct.unpack('<H', original_data[pos + 1:pos + 3])[0]
                current_time += wait_time
                output_data.extend([cmd, original_data[pos + 1], original_data[pos + 2]])
                pos += 3
            elif cmd in [0x62, 0x63] or (cmd & 0xF0 == 0x70):
                # Wait commands
                if cmd == 0x62:
                    current_time += 735
                elif cmd == 0x63:
                    current_time += 882
                else:
                    current_time += (cmd & 0x0F) + 1
                output_data.append(cmd)
                pos += 1
            else:
                pos += 1
        
        # Create the VGM file
        self.create_vgm_file(original_data, bytes(output_data), output_path)

    def create_dac_vgm(self, original_data: bytes, header: Dict, 
                      sample_group: str, events: List[DACEvent], 
                      output_path: str):
        """Create a VGM file containing only the specified DAC sample group"""
        pos = header['data_start']
        output_data = bytearray()
        current_time = 0
        
        print(f"Creating VGM for DAC sample group {sample_group}...")
        
        # Create event lookup for quick access
        events_by_time = defaultdict(list)
        for event in events:
            events_by_time[event.time].append(event)
        
        while pos < len(original_data):
            cmd = original_data[pos]
            
            # Check if we have DAC events at current time
            current_events = events_by_time.get(current_time, [])
            
            if cmd == 0x66:  # End of data
                output_data.append(cmd)
                break
            elif cmd == 0x2A:  # DAC data write
                if pos + 1 >= len(original_data):
                    break
                value = original_data[pos + 1]
                
                # Include if this sample belongs to our group
                include_sample = False
                for event in current_events:
                    if event.sample_group == sample_group:
                        include_sample = True
                        break
                
                if include_sample:
                    output_data.extend([cmd, value])
                
                pos += 2
            elif cmd & 0xF0 == 0x80:  # DAC write + wait
                # Check if this DAC sample belongs to our group
                include_sample = False
                for event in current_events:
                    if event.sample_group == sample_group:
                        include_sample = True
                        break
                
                if include_sample:
                    output_data.append(cmd)
                else:
                    # Convert to regular wait if DAC not included
                    wait_time = cmd & 0x0F
                    if wait_time == 0:
                        output_data.extend([0x61, 0x01, 0x00])  # Wait 1 sample
                    else:
                        output_data.append(0x70 + wait_time - 1)
                
                current_time += 1
                pos += 1
            elif cmd == 0x61:  # Wait n samples
                if pos + 2 >= len(original_data):
                    break
                wait_time = struct.unpack('<H', original_data[pos + 1:pos + 3])[0]
                current_time += wait_time
                output_data.extend([cmd, original_data[pos + 1], original_data[pos + 2]])
                pos += 3
            elif cmd in [0x62, 0x63] or (cmd & 0xF0 == 0x70):
                # Other wait commands
                if cmd == 0x62:
                    current_time += 735
                elif cmd == 0x63:
                    current_time += 882
                else:
                    current_time += (cmd & 0x0F) + 1
                output_data.append(cmd)
                pos += 1
            else:
                # Skip other commands for DAC-only file
                pos += 1
        
        # Create the VGM file
        self.create_vgm_file(original_data, bytes(output_data), output_path)
        """Create a new VGM file with processed data"""
        header_data = bytearray(original_data[:64])
        data_offset = len(header_data) - 52
        struct.pack_into('<I', header_data, 52, data_offset)
        
        with open(output_path, 'wb') as f:
            f.write(header_data)
            f.write(processed_data)

    def isolate_instruments(self, input_file: str, output_dir: str):
        """Create isolated files for each discovered instrument"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, 'rb') as f:
            original_data = f.read()
        
        header = self.read_vgm_header(original_data)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        print(f"Processing {input_file}...")
        
        # Analyze the file to discover instruments and DAC samples
        instrument_events, dac_events = self.analyze_instruments(original_data, header)
        
        # Create a file for each FM instrument
        for i, (inst_hash, events) in enumerate(instrument_events.items()):
            instrument = self.discovered_instruments[inst_hash]
            note_count = len([e for e in events if e.note_on])
            
            print(f"FM Instrument {i+1:02d} ({inst_hash}): {note_count} notes")
            
            output_file = os.path.join(output_dir, f"{base_name}_fm_inst_{i+1:02d}_{inst_hash}.vgm")
            self.create_instrument_vgm(original_data, header, inst_hash, events, output_file)
        
        # Create a file for each DAC sample group
        for i, (sample_group, events) in enumerate(dac_events.items()):
            event_count = len(events)
            
            print(f"DAC Sample {i+1:02d} ({sample_group}): {event_count} events")
            
            output_file = os.path.join(output_dir, f"{base_name}_dac_{i+1:02d}_{sample_group}.vgm")
            self.create_dac_vgm(original_data, header, sample_group, events, output_file)
        
        # Create summary file
        summary_path = os.path.join(output_dir, f"{base_name}_instrument_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Instrument Analysis for {input_file}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FM INSTRUMENTS:\n")
            f.write("-" * 20 + "\n")
            for i, (inst_hash, events) in enumerate(instrument_events.items()):
                instrument = self.discovered_instruments[inst_hash]
                note_events = [e for e in events if e.note_on]
                
                f.write(f"FM Instrument {i+1:02d} ({inst_hash}):\n")
                f.write(f"  Notes played: {len(note_events)}\n")
                f.write(f"  Channels used: {list(set(e.channel for e in note_events))}\n")
                f.write(f"  Time range: {min(e.time for e in events)} - {max(e.time for e in events)}\n")
                f.write("\n")
            
            f.write("\nDAC SAMPLES:\n")
            f.write("-" * 20 + "\n")
            for i, (sample_group, events) in enumerate(dac_events.items()):
                f.write(f"DAC Sample {i+1:02d} ({sample_group}):\n")
                f.write(f"  Events: {len(events)}\n")
                if events:
                    f.write(f"  Time range: {min(e.time for e in events)} - {max(e.time for e in events)}\n")
                f.write("\n")
        
        print(f"\nInstrument isolation complete!")
        print(f"Files saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python vgm_instrument_isolator.py <input_vgm_file> <output_directory>")
        print("Example: python vgm_instrument_isolator.py song.vgm ./isolated_instruments/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        processor = VGMInstrumentProcessor()
        processor.isolate_instruments(input_file, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
