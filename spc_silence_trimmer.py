#!/usr/bin/env python3
"""
SPC Direct Silence Trimmer
Analyzes SPC700 sound state directly to detect when audio has effectively ended,
then adds timing tags to preserve the original SPC format and interpolation.
"""

import os
import sys
import struct
import time
from pathlib import Path
import argparse
import subprocess
import tempfile
import re

class SPC700Analyzer:
    """Analyzes SPC700 sound processor state to detect audio activity"""
    
    def __init__(self):
        # SPC700 voice/channel parameters
        self.num_voices = 8
        self.voice_size = 16  # Each voice uses 16 bytes in voice table
        
        # Key SPC700 registers for audio analysis
        self.dsp_registers = {
            'MVOLL': 0x0C,    # Main volume left
            'MVOLR': 0x1C,    # Main volume right
            'EVOLL': 0x2C,    # Echo volume left
            'EVOLR': 0x3C,    # Echo volume right
            'KON': 0x4C,      # Key on flags
            'KOFF': 0x5C,     # Key off flags
            'FLG': 0x6C,      # DSP flags
            'ENDX': 0x7C,     # Voice end flags
        }
    
    def extract_spc_state(self, spc_data):
        """Extract SPC700 CPU and DSP state from SPC file"""
        if len(spc_data) < 0x10200:  # Minimum SPC file size
            raise ValueError("SPC file too small")
        
        # SPC file format:
        # 0x0000-0x0020: Header
        # 0x0021-0x0022: PC register
        # 0x0023: A register  
        # 0x0024: X register
        # 0x0025: Y register
        # 0x0026: PSW register
        # 0x0027: SP register
        # 0x0028-0x0029: Reserved
        # 0x002A-0x002B: ID666 tag info
        # 0x002C-0x002D: Reserved
        # 0x002E: Start of optional ID666 tag
        # 0x0100-0x00FF: DSP register data (128 bytes)
        # 0x0100-0x10000: 64KB SPC700 RAM
        # 0x10000-0x10080: 128 DSP registers
        
        state = {
            'pc': struct.unpack('<H', spc_data[0x25:0x27])[0],
            'a': spc_data[0x27],
            'x': spc_data[0x28], 
            'y': spc_data[0x29],
            'psw': spc_data[0x2A],
            'sp': spc_data[0x2B],
            'ram': spc_data[0x100:0x10100],  # 64KB RAM
            'dsp': spc_data[0x10100:0x10180] if len(spc_data) >= 0x10180 else spc_data[0x10100:0x10100+128]
        }
        
        return state
    
    def analyze_voice_activity(self, dsp_data):
        """Analyze individual voice activity from DSP registers"""
        voices = []
        
        for voice in range(self.num_voices):
            base_reg = voice * 0x10
            
            voice_info = {
                'id': voice,
                'vol_left': dsp_data[base_reg + 0x00] if base_reg + 0x00 < len(dsp_data) else 0,
                'vol_right': dsp_data[base_reg + 0x01] if base_reg + 0x01 < len(dsp_data) else 0,
                'pitch_low': dsp_data[base_reg + 0x02] if base_reg + 0x02 < len(dsp_data) else 0,
                'pitch_high': dsp_data[base_reg + 0x03] if base_reg + 0x03 < len(dsp_data) else 0,
                'source': dsp_data[base_reg + 0x04] if base_reg + 0x04 < len(dsp_data) else 0,
                'adsr1': dsp_data[base_reg + 0x05] if base_reg + 0x05 < len(dsp_data) else 0,
                'adsr2': dsp_data[base_reg + 0x06] if base_reg + 0x06 < len(dsp_data) else 0,
                'gain': dsp_data[base_reg + 0x07] if base_reg + 0x07 < len(dsp_data) else 0,
                'envx': dsp_data[base_reg + 0x08] if base_reg + 0x08 < len(dsp_data) else 0,
                'outx': dsp_data[base_reg + 0x09] if base_reg + 0x09 < len(dsp_data) else 0,
            }
            
            # Calculate pitch (frequency)
            voice_info['pitch'] = voice_info['pitch_low'] | (voice_info['pitch_high'] << 8)
            
            # Determine if voice is potentially active
            voice_info['has_volume'] = voice_info['vol_left'] > 0 or voice_info['vol_right'] > 0
            voice_info['has_pitch'] = voice_info['pitch'] > 0
            voice_info['has_envelope'] = voice_info['envx'] > 0
            voice_info['has_output'] = abs(voice_info['outx']) > 0
            
            voice_info['potentially_active'] = (
                voice_info['has_volume'] and 
                voice_info['has_pitch'] and
                (voice_info['has_envelope'] or voice_info['has_output'])
            )
            
            voices.append(voice_info)
        
        return voices
    
    def get_global_audio_state(self, dsp_data):
        """Get global DSP state information"""
        if len(dsp_data) < 128:
            return {'active': False, 'reason': 'insufficient_dsp_data'}
        
        state = {
            'main_vol_left': dsp_data[self.dsp_registers['MVOLL']],
            'main_vol_right': dsp_data[self.dsp_registers['MVOLR']],
            'echo_vol_left': dsp_data[self.dsp_registers['EVOLL']],
            'echo_vol_right': dsp_data[self.dsp_registers['EVOLR']],
            'key_on': dsp_data[self.dsp_registers['KON']],
            'key_off': dsp_data[self.dsp_registers['KOFF']],
            'flags': dsp_data[self.dsp_registers['FLG']],
            'voice_end': dsp_data[self.dsp_registers['ENDX']],
        }
        
        # Check if main volume is set
        state['has_main_volume'] = state['main_vol_left'] > 0 or state['main_vol_right'] > 0
        
        # Check if any voices are keyed on
        state['voices_keyed_on'] = state['key_on'] > 0
        
        # Check if echo is active
        state['echo_active'] = (state['echo_vol_left'] > 0 or state['echo_vol_right'] > 0) and not (state['flags'] & 0x20)
        
        # Overall activity assessment
        state['potentially_active'] = state['has_main_volume'] and (state['voices_keyed_on'] or state['echo_active'])
        
        return state

class SPCFile:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = None
        self.analyzer = SPC700Analyzer()
        self.load()
    
    def load(self):
        """Load SPC file data"""
        with open(self.filepath, 'rb') as f:
            self.data = f.read()
        
        # Validate SPC file
        if len(self.data) < 33 or not self.data.startswith(b'SNES-SPC700 Sound File Data'):
            raise ValueError("Not a valid SPC file")
    
    def analyze_audio_state(self):
        """Analyze the current audio state of the SPC"""
        try:
            state = self.analyzer.extract_spc_state(self.data)
            voices = self.analyzer.analyze_voice_activity(state['dsp'])
            global_state = self.analyzer.get_global_audio_state(state['dsp'])
            
            # Count active voices
            active_voices = sum(1 for v in voices if v['potentially_active'])
            
            analysis = {
                'voices': voices,
                'active_voice_count': active_voices,
                'global_state': global_state,
                'likely_silent': not global_state['potentially_active'] and active_voices == 0,
                'silence_confidence': self.calculate_silence_confidence(voices, global_state)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'likely_silent': False, 'silence_confidence': 0.0}
    
    def calculate_silence_confidence(self, voices, global_state):
        """Calculate confidence that the SPC is in a silent state"""
        confidence = 0.0
        
        # Check global volume
        if not global_state['has_main_volume']:
            confidence += 0.4
        
        # Check if no voices are keyed on
        if not global_state['voices_keyed_on']:
            confidence += 0.3
        
        # Check individual voice states
        active_voices = sum(1 for v in voices if v['potentially_active'])
        if active_voices == 0:
            confidence += 0.2
        else:
            confidence -= 0.1 * active_voices
        
        # Check for envelope activity
        envelope_activity = sum(1 for v in voices if v['has_envelope'])
        if envelope_activity == 0:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_timing_info(self):
        """Extract existing timing information from ID666 tag"""
        timing = {'play_time': None, 'fade_time': None}
        
        if len(self.data) >= 0x2E + 210:
            try:
                tag_offset = 0x2E
                
                # Play time (offset 0x6E from tag start, 4 bytes little endian)
                play_time_offset = tag_offset + 0x6E
                if play_time_offset + 4 <= len(self.data):
                    play_time_raw = struct.unpack('<I', self.data[play_time_offset:play_time_offset + 4])[0]
                    if play_time_raw > 0:
                        timing['play_time'] = play_time_raw / 64000.0
                
                # Fade time (offset 0x72 from tag start, 4 bytes little endian)  
                fade_time_offset = tag_offset + 0x72
                if fade_time_offset + 4 <= len(self.data):
                    fade_time_raw = struct.unpack('<I', self.data[fade_time_offset:fade_time_offset + 4])[0]
                    if fade_time_raw > 0:
                        timing['fade_time'] = fade_time_raw
                        
            except:
                pass
        
        return timing
    
    def set_timing_info(self, play_time_seconds, fade_time_ms=5000):
        """Add timing information to the SPC file"""
        # Ensure we have space for ID666 tag
        data = bytearray(self.data)
        if len(data) < 0x2E + 210:
            data.extend(b'\x00' * (0x2E + 210 - len(data)))
        
        tag_offset = 0x2E
        
        # Set play time
        play_time_raw = int(play_time_seconds * 64000)
        play_time_offset = tag_offset + 0x6E
        data[play_time_offset:play_time_offset + 4] = struct.pack('<I', play_time_raw)
        
        # Set fade time
        fade_time_offset = tag_offset + 0x72
        data[fade_time_offset:fade_time_offset + 4] = struct.pack('<I', fade_time_ms)
        
        self.data = bytes(data)
    
    def save(self, output_path):
        """Save the SPC file"""
        with open(output_path, 'wb') as f:
            f.write(self.data)

def check_existing_length_tags(spc_file):
    """Check for existing SPC_LENGTH or similar tags that might indicate duration"""
    data = spc_file.data
    tags_found = {}
    
    # Look for common SPC length tags in the file
    length_patterns = [
        b'SPC_LENGTH=',
        b'Length=',
        b'Song Length=',
        b'Play Time=',
        b'Duration='
    ]
    
    for pattern in length_patterns:
        pos = data.find(pattern)
        if pos != -1:
            # Try to extract the value after the equals sign
            try:
                start = pos + len(pattern)
                end = start
                while end < len(data) and data[end:end+1] not in [b'\x00', b'\n', b'\r', b' ']:
                    end += 1
                value_str = data[start:end].decode('ascii', errors='ignore')
                tags_found[pattern.decode('ascii')] = value_str
            except:
                pass
    
    return tags_found

def analyze_spc_with_emulation(spc_file, max_check_time=300):
    """
    Use external SPC emulator to actually run the SPC and detect silence.
    This is the proper way to detect when a sound effect or jingle ends.
    """
    
    # Check for existing length tags first
    existing_tags = check_existing_length_tags(spc_file)
    if existing_tags:
        print(f"  - Found existing tags: {existing_tags}")
        # Try to parse duration from tags
        for tag_name, tag_value in existing_tags.items():
            try:
                # Parse various time formats (seconds, mm:ss, etc.)
                if ':' in tag_value:
                    # mm:ss format
                    parts = tag_value.split(':')
                    if len(parts) == 2:
                        minutes, seconds = int(parts[0]), float(parts[1])
                        duration = minutes * 60 + seconds
                        print(f"  - Parsed duration from {tag_name}: {duration} seconds")
                        return duration
                else:
                    # Try as plain seconds
                    duration = float(tag_value)
                    if 1 <= duration <= 600:  # Reasonable range
                        print(f"  - Using duration from {tag_name}: {duration} seconds")
                        return duration
            except:
                continue
    
    # Try to use external SPC player for emulation
    print(f"  - Attempting emulation-based analysis...")
    
    # List of common SPC players that support batch processing
    spc_players = [
        'spc123',      # Command line SPC player
        'openspc',     # OpenSPC
        'snes_spc',    # Game_Music_Emu based
        'in_spc',      # Winamp plugin (if available via command line)
    ]
    
    temp_dir = Path(tempfile.gettempdir())
    temp_spc = temp_dir / f"temp_analysis_{os.getpid()}.spc"
    
    try:
        # Write SPC to temp file
        spc_file.save(temp_spc)
        
        for player in spc_players:
            try:
                # Try to get duration info from the player
                result = subprocess.run([player, '--length', str(temp_spc)], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout:
                    # Try to parse duration from output
                    duration = parse_player_output(result.stdout)
                    if duration:
                        print(f"  - Duration detected by {player}: {duration} seconds")
                        return duration
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print(f"  - No SPC player available for emulation analysis")
        
    finally:
        # Clean up temp file
        if temp_spc.exists():
            temp_spc.unlink()
    
    # Fallback to heuristic analysis
    return analyze_spc_heuristic(spc_file)

def parse_player_output(output):
    """Parse duration from SPC player output"""
    import re
    
    # Common patterns for duration output
    patterns = [
        r'Length:\s*(\d+(?:\.\d+)?)\s*s',
        r'Duration:\s*(\d+(?:\.\d+)?)\s*s', 
        r'Time:\s*(\d+):(\d+)',
        r'(\d+(?:\.\d+)?)\s*seconds',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                return float(match.group(1))
            elif len(match.groups()) == 2:
                # mm:ss format
                minutes, seconds = int(match.group(1)), int(match.group(2))
                return minutes * 60 + seconds
    
    return None

def analyze_spc_heuristic(spc_file):
    """Fallback heuristic analysis when emulation is not available"""
    print(f"  - Using heuristic analysis...")
    
    # Analyze initial state
    initial_analysis = spc_file.analyze_audio_state()
    
    if 'error' in initial_analysis:
        print(f"  - Analysis error: {initial_analysis['error']}")
        return 120.0
    
    print(f"  - Active voices: {initial_analysis['active_voice_count']}")
    print(f"  - Silence confidence: {initial_analysis['silence_confidence']:.2f}")
    
    # More aggressive detection for likely sound effects
    if initial_analysis['silence_confidence'] > 0.7:
        return 5.0  # Very likely silent or very short
    elif initial_analysis['active_voice_count'] == 0:
        return 10.0  # No active voices, probably short
    elif initial_analysis['active_voice_count'] <= 2:
        # Check voice characteristics for sound effects
        voices = initial_analysis['voices']
        active_voices = [v for v in voices if v['potentially_active']]
        
        # Sound effects often have high pitch or simple envelopes
        if active_voices:
            avg_pitch = sum(v['pitch'] for v in active_voices) / len(active_voices)
            if avg_pitch > 2000:  # High pitch suggests sound effect
                return 15.0
            else:
                return 30.0  # Low pitch, might be longer
        return 30.0
    else:
        return 120.0  # Multiple voices, probably music

def simulate_spc_playback(spc_file, max_duration=600, check_interval=5):
    """
    Analyze SPC file to determine appropriate playback duration.
    Uses emulation when possible, falls back to heuristics.
    """
    return analyze_spc_with_emulation(spc_file, max_duration)

def process_spc_file(input_path, output_dir=None, dry_run=False, force=False):
    """Process a single SPC file"""
    spc_path = Path(input_path)
    
    if output_dir:
        output_path = Path(output_dir) / spc_path.name
    else:
        # Create output with .trimmed suffix
        output_path = spc_path.with_stem(spc_path.stem + '.trimmed')
    
    print(f"Processing: {spc_path}")
    
    try:
        # Load SPC file
        spc = SPCFile(spc_path)
        
        # Check existing timing
        timing = spc.get_timing_info()
        if timing['play_time'] is not None and not force:
            print(f"  - Already has timing info: {timing['play_time']:.1f}s play, {timing['fade_time']}ms fade")
            return True
        
        # Analyze and determine duration
        suggested_duration = simulate_spc_playback(spc)
        print(f"  - Suggested duration: {suggested_duration:.1f} seconds")
        
        if not dry_run:
            # Apply timing and save
            spc.set_timing_info(suggested_duration)
            spc.save(output_path)
            print(f"  - Saved: {output_path}")
        else:
            print(f"  - Would save: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  - Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Trim silence from SPC files by analyzing SPC700 state")
    parser.add_argument("input", nargs="+", help="SPC file(s) or directory to process")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("-f", "--force", action="store_true", help="Process files even if they have timing info")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    
    args = parser.parse_args()
    
    # Collect SPC files
    spc_files = []
    for input_path in args.input:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.spc':
            spc_files.append(path) 
        elif path.is_dir():
            pattern = "**/*.spc" if args.recursive else "*.spc"
            spc_files.extend(path.glob(pattern))
    
    if not spc_files:
        print("No SPC files found!")
        return 1
    
    print(f"Found {len(spc_files)} SPC files")
    
    # Create output directory if specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Process files
    success_count = 0
    for spc_file in spc_files:
        if process_spc_file(spc_file, output_dir, args.dry_run, args.force):
            success_count += 1
        print()  # Empty line between files
    
    print(f"Completed: {success_count}/{len(spc_files)} files processed")
    return 0 if success_count == len(spc_files) else 1

if __name__ == "__main__":
    sys.exit(main())
