"""Join H3 segments and pass their last frame without writing media files."""

import io
from fractions import Fraction

import av


FPS = 24
AUDIO_RATE = 48000


def last_frame_png(mp4):
    """The next H3 segment begins on the preceding segment's final frame."""
    last = None
    with av.open(io.BytesIO(mp4), mode="r") as container:
        for frame in container.decode(video=0):
            last = frame
    if last is None:
        raise RuntimeError("H3 segment contains no video frames.")
    with io.BytesIO() as output:
        last.to_image().save(output, format="PNG")
        return output.getvalue()


def join_segments(parts, seconds, progress=None):
    """Decode and join MP4s in RAM, trimming to the requested whole second."""
    if not parts or not 1 <= seconds <= 30:
        raise ValueError("Invalid H3 segment request")
    target_frames = seconds * FPS
    target_samples = seconds * AUDIO_RATE
    output = io.BytesIO()
    video_count = audio_count = 0
    with av.open(output, mode="w", format="mp4") as destination:
        with av.open(io.BytesIO(parts[0]), mode="r") as first:
            source_video = first.streams.video[0]
            width, height = source_video.width, source_video.height
            with_audio = bool(first.streams.audio)
        video_out = destination.add_stream("libx264", rate=FPS)
        video_out.width, video_out.height = width, height
        video_out.pix_fmt = "yuv420p"
        video_out.options = {"preset": "veryfast", "crf": "18"}
        audio_out = None
        resampler = None
        if with_audio:
            audio_out = destination.add_stream("aac", rate=AUDIO_RATE)
            audio_out.layout = "stereo"
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
        for index, part in enumerate(parts):
            if progress:
                progress(index, len(parts))
            with av.open(io.BytesIO(part), mode="r") as source:
                streams = [source.streams.video[0]]
                if audio_out and source.streams.audio:
                    streams.append(source.streams.audio[0])
                for packet in source.demux(streams):
                    for frame in packet.decode():
                        if isinstance(frame, av.VideoFrame):
                            if video_count >= target_frames:
                                continue
                            frame = frame.reformat(width=width, height=height, format="yuv420p")
                            frame.pts = video_count
                            frame.time_base = Fraction(1, FPS)
                            for encoded in video_out.encode(frame):
                                destination.mux(encoded)
                            video_count += 1
                        elif isinstance(frame, av.AudioFrame) and audio_out:
                            for sample in resampler.resample(frame):
                                if audio_count >= target_samples:
                                    continue
                                sample.pts = audio_count
                                sample.time_base = Fraction(1, AUDIO_RATE)
                                for encoded in audio_out.encode(sample):
                                    destination.mux(encoded)
                                audio_count += sample.samples
        for encoded in video_out.encode():
            destination.mux(encoded)
        if audio_out:
            for sample in resampler.resample(None):
                if audio_count >= target_samples:
                    break
                sample.pts = audio_count
                sample.time_base = Fraction(1, AUDIO_RATE)
                for encoded in audio_out.encode(sample):
                    destination.mux(encoded)
                audio_count += sample.samples
            for encoded in audio_out.encode():
                destination.mux(encoded)
    if video_count != target_frames:
        raise RuntimeError("H3 segments are shorter than the requested duration.")
    return output.getvalue()
