# Gate 2 synthetic video fixture

`gate2-three-frame.avi` is a deterministic, code-generated acceptance fixture.

- Generate: `python scripts/generate_gate2_fixture.py`
- Dimensions: 640 x 480 pixels
- Frames: 3 visually distinct frames
- Frame rate: 5 FPS
- Codec/container: Motion JPEG (`MJPG`) in AVI
- License: CC0-1.0
- SHA-256: `dd607536bed2fb9493bedfea68600e6a10af88c6073a5f94b2c4711e38bd2080`

The generator reopens the file and requires exactly three decodable BGR frames before returning the checksum.
