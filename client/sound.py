import logging
from pathlib import Path
import random

from server.blocks import ASSETS
from server.sound import (
    get_block_break_sound,
    get_block_breaking_sound,
    get_block_walk_sound,
)

logger = logging.getLogger(__name__)


class SoundManager:
    """Audio playback manager for block breaking and walking sound effects."""

    def __init__(self, sounds_dir=None, volume=1.0, enabled=True):
        if sounds_dir is None:
            sounds_dir = ASSETS / "sounds"
        self.sounds_dir = Path(sounds_dir)
        self.volume = float(volume)
        self.enabled = bool(enabled)
        self._sources = {}  # name -> pyglet.media.Source
        self._breaking_player = None
        self._breaking_sound_name = None
        self._audio_available = True
        self._init_sounds()

    def _init_sounds(self):
        """Scan sounds directory and load available audio sources."""
        if not self.sounds_dir.is_dir():
            return

        import pyglet

        for ext in ("*.ogg", "*.wav", "*.mp3"):
            for file_path in self.sounds_dir.glob(ext):
                stem = file_path.stem
                try:
                    source = pyglet.media.load(str(file_path), streaming=False)
                    self._sources[stem] = source
                except Exception as e:
                    logger.debug("Could not load sound %s: %s", file_path.name, e)

    def play(self, sound_name: str, volume=1.0):
        """Play a loaded sound by name."""
        if not self.enabled or not self._audio_available or not sound_name:
            return None

        source = self._sources.get(sound_name)
        if source is None:
            # Try reloading or checking if path exists with extension
            stem = Path(sound_name).stem
            source = self._sources.get(stem)
            if source is None:
                return None

        try:
            player = source.play()
            if player is not None:
                player.volume = max(0.0, min(1.0, self.volume * float(volume)))
            return player
        except Exception as e:
            logger.debug("Playback error for sound '%s': %s", sound_name, e)
            return None

    def play_walk_sound(self, block_or_id, volume=0.85, rng=None):
        """Play a footstep sound for the given block type."""
        sound_name = get_block_walk_sound(block_or_id, rng=rng)
        if sound_name:
            # Slight random volume jitter for natural footstep sound variation
            r = rng or random
            vol = volume * r.uniform(0.9, 1.1)
            return self.play(sound_name, volume=vol)
        return None

    def start_breaking_sound(self, block_or_id, volume=0.8, loop=True, rng=None):
        """Start playing a block breaking sound once (looping if breaking continues)."""
        if not self.enabled or not self._audio_available:
            return None

        sound_name = get_block_breaking_sound(block_or_id, rng=rng)
        if not sound_name:
            self.stop_breaking_sound()
            return None

        # If already playing this exact breaking sound, keep it playing/looping
        if self._breaking_player is not None and self._breaking_sound_name == sound_name:
            if getattr(self._breaking_player, "playing", False):
                return self._breaking_player

        self.stop_breaking_sound()

        source = self._sources.get(sound_name)
        if source is None:
            stem = Path(sound_name).stem
            source = self._sources.get(stem)
            if source is None:
                return None

        try:
            import pyglet

            player = pyglet.media.Player()
            player.queue(source)
            player.loop = bool(loop)
            r = rng or random
            vol = volume * r.uniform(0.9, 1.1)
            player.volume = max(0.0, min(1.0, self.volume * float(vol)))
            player.play()
            self._breaking_player = player
            self._breaking_sound_name = sound_name
            return player
        except Exception as e:
            logger.debug("Playback error for breaking sound '%s': %s", sound_name, e)
            return None

    def stop_breaking_sound(self):
        """Immediately stop and dismiss the active breaking sound midway."""
        if self._breaking_player is not None:
            try:
                self._breaking_player.pause()
                if hasattr(self._breaking_player, "delete"):
                    self._breaking_player.delete()
            except Exception:
                pass
            self._breaking_player = None
            self._breaking_sound_name = None

    def play_breaking_sound(self, block_or_id, volume=0.8, loop=True, rng=None):
        """Play a block breaking sound once, looping if needed."""
        return self.start_breaking_sound(block_or_id, volume=volume, loop=loop, rng=rng)

    def play_break_sound(self, block_or_id, volume=1.0, rng=None):
        """Play a block breaking sound for the given block type."""
        sound_name = get_block_break_sound(block_or_id, rng=rng)
        if sound_name:
            return self.play(sound_name, volume=volume)
        return None

