import contextlib
import os
import random
import typing
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
import torch
import torchaudio

from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MusicGen

first_run = True
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
MODEL = None

try:

    def fake_news_get(url, **kwargs):
        kwargs.setdefault("allow_redirects", True)
        return requests.api.request("get", "http://127.0.0.1/", **kwargs)

    original_get = requests.get
    requests.get = fake_news_get
    import gradio as gr
    from gradio.themes.utils import colors

    requests.get = original_get
except Exception:
    raise RuntimeError("Ran into an error bypassing gradio's tracking...")


def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def set_seed(seed: int = 0):
    original_seed = seed
    if seed == -1:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("seed: " + str(seed))
    if seed <= 0:
        seed = np.random.default_rng().integers(1, 2**32 - 1)
    assert 0 < seed < 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return original_seed if original_seed != 0 else seed


def generate_cmelody(
    descriptions: typing.List[str],
    melody_wavs: typing.Union[torch.Tensor, typing.List[typing.Optional[torch.Tensor]]],
    msr: int,
    prompt: torch.Tensor,
    psr: int,
    MODEL,
    progress: bool = False,
) -> torch.Tensor:
    if isinstance(melody_wavs, torch.Tensor):
        if melody_wavs.dim() == 2:
            melody_wavs = melody_wavs[None]
        if melody_wavs.dim() != 3:
            raise ValueError("melody_wavs should have a shape [B, C, T].")
        melody_wavs = list(melody_wavs)
    else:
        for melody in melody_wavs:
            if melody is not None:
                assert melody.dim() == 2, "one melody in the list has the wrong number of dims."

    melody_wavs = [
        convert_audio(wav, msr, MODEL.sample_rate, MODEL.audio_channels) if wav is not None else None
        for wav in melody_wavs
    ]

    if prompt.dim() == 2:
        prompt = prompt[None]
    if prompt.dim() != 3:
        raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
    prompt = convert_audio(prompt, psr, MODEL.sample_rate, MODEL.audio_channels)
    if descriptions is None:
        descriptions = [None] * len(prompt)
    attributes, prompt_tokens = MusicGen._prepare_tokens_and_attributes(
        MODEL, descriptions=descriptions, prompt=prompt, melody_wavs=melody_wavs
    )
    assert prompt_tokens is not None
    return MusicGen._generate_tokens(MODEL, attributes, prompt_tokens, progress)


def initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff, sc_text):
    wav = None
    if continue_file:
        data_waveform, cfsr = torchaudio.load(continue_file)
        wav = data_waveform.cuda()
        cf_len = 0
        with contextlib.closing(wave.open(continue_file, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            cf_len = frames / float(rate)
        if wav.dim() == 2:
            wav = wav[None]
        wav = wav[:, :, int(-cfsr * min(29, cf_len, duration - 1, cf_cutoff)) :]
        new_chunk = None
        if not melody_boolean:
            if not sc_text:
                new_chunk = MODEL.generate_continuation(wav, prompt_sample_rate=cfsr, progress=False)
            else:
                new_chunk = MODEL.generate_continuation(
                    wav, descriptions=[text], prompt_sample_rate=cfsr, progress=False
                )
            wav = new_chunk
        else:
            new_chunk = generate_cmelody([text], melody, msr, wav, cfsr, MODEL, progress=False)
            wav = new_chunk
    else:
        if melody_boolean:
            wav = MODEL.generate_with_chroma(
                descriptions=[text], melody_wavs=melody, melody_sample_rate=msr, progress=False
            )
        else:
            wav = MODEL.generate(descriptions=[text], progress=False)
    return wav


def generate(
    model,
    text,
    melody,
    duration,
    topk,
    topp,
    temperature,
    cfg_coef,
    base_duration,
    sliding_window_seconds,
    continue_file,
    cf_cutoff,
    sc_text,
    seed,
):
    # seed workaround
    global first_run
    if first_run:
        first_run = False
        d = generate(
            model,
            "A",
            None,
            1,
            topk,
            topp,
            temperature,
            2,
            base_duration,
            sliding_window_seconds,
            None,
            cf_cutoff,
            sc_text,
            seed,
        )
    #

    final_length_seconds = duration
    descriptions = text
    global MODEL
    topk = int(topk)
    int_seed = int(seed)
    set_seed(int_seed)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    if duration > 30:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=base_duration,
        )
    else:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )
    iterations_required = int(final_length_seconds / sliding_window_seconds)
    print(f"Iterations required: {iterations_required}")
    sr = MODEL.sample_rate
    print(f"Sample rate: {sr}")
    msr = None
    wav = None  # wav shape will be [1, 1, sr * seconds]
    melody_boolean = False
    if melody:
        msr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., : int(msr * MODEL.lm.cfg.dataset.segment_duration)]
        melody_boolean = True

    if duration > 30:
        for i in range(iterations_required):
            print(f"Generating {i + 1}/{iterations_required}")
            if i == 0:
                wav = initial_generate(
                    melody_boolean, MODEL, text, melody, msr, continue_file, base_duration, cf_cutoff, sc_text
                )
                wav = wav[:, :, : sr * sliding_window_seconds]
            else:
                new_chunk = None
                previous_chunk = wav[:, :, -sr * (base_duration - sliding_window_seconds) :]
                if continue_file:
                    if not sc_text:
                        new_chunk = MODEL.generate_continuation(
                            previous_chunk, prompt_sample_rate=sr, progress=False
                        )
                    else:
                        new_chunk = MODEL.generate_continuation(
                            previous_chunk, descriptions=[text], prompt_sample_rate=sr, progress=False
                        )
                else:
                    new_chunk = MODEL.generate_continuation(
                        previous_chunk, descriptions=[text], prompt_sample_rate=sr, progress=False
                    )
                wav = torch.cat((wav, new_chunk[:, :, -sr * sliding_window_seconds :]), dim=2)
    else:
        wav = initial_generate(
            melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff, sc_text
        )

    print(f"Final length: {wav.shape[2] / sr}s")
    output = wav.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(
            file.name,
            output,
            MODEL.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            add_suffix=False,
            loudness_compressor=True,
        )
    set_seed(-1)
    return file.name


try:
    theme = gr.themes.Base(
        primary_hue=colors.violet,
        secondary_hue=colors.indigo,
        neutral_hue=colors.slate,
        font=[gr.themes.GoogleFont("Fira Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
    ).set(slider_color_dark="*primary_500")
    css = Path(__file__).with_suffix(".css").read_text()
except Exception:
    css = None

demo = gr.Blocks(
    title="MusicGen",
    analytics_enabled=False,
    theme=theme,
    css=css,
)

with demo:
    gr.Markdown("""# MusicGen""")
    with gr.Row():
        with gr.Column():
            with gr.Row(variant="panel").style(equal_height=True):
                with gr.Column(scale=2):
                    text = gr.Textbox(label="Prompt", interactive=True)
                    sc_text = gr.Checkbox(
                        label="Use prompt for continuation",
                        info="Uncheck to continue from audio only",
                        value=True,
                    )
                    model = gr.Radio(
                        ["small", "medium", "large", "melody"],
                        label="Model",
                        value="melody",
                        interactive=True,
                    )

                with gr.Column(scale=1, min_width=240):
                    melody = gr.Audio(
                        source="upload",
                        type="numpy",
                        label="Melody Conditioning",
                        interactive=True,
                    )
                with gr.Column(scale=1, min_width=240):
                    continue_file = gr.Audio(
                        source="upload",
                        type="filepath",
                        label="Continue sound file",
                        interactive=True,
                    )

            with gr.Row(variant="panel").style(equal_height=True):
                with gr.Column(scale=1):
                    duration = gr.Slider(
                        label="Duration", info="seconds", value=30, minimum=1, maximum=600, step=0.5
                    )
                    base_duration = gr.Slider(
                        label="Base duration", info="seconds", value=30, minimum=1, maximum=60, step=0.5
                    )
                    sliding_window_seconds = gr.Slider(
                        label="Sliding window", info="seconds", value=15, minimum=1, maximum=30, step=0.5
                    )
                    cf_cutoff = gr.Slider(
                        label="Cont. cutoff", info="seconds", value=7.5, minimum=1, maximum=30, step=0.5
                    )
                with gr.Column(scale=1):
                    cfg_coef = gr.Slider(
                        label="CFG Scale",
                        info="3.0 recommended",
                        value=3.0,
                        interactive=True,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.01,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        info=">1.0 not recommended",
                        value=1.0,
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        interactive=True,
                    )
                    topk = gr.Slider(
                        label="Top K",
                        info="tokens",
                        value=250,
                        minimum=0,
                        maximum=500,
                        step=1,
                    )
                    topp = gr.Slider(
                        label="Top P",
                        value=0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        interactive=True,
                        info="0 = off, mixed results...",
                    )

                with gr.Column(scale=1):
                    seed = gr.Number(
                        label="seed",
                        value=-1,
                        precision=0,
                    )
                    submit = gr.Button(
                        "Submit",
                        variant="primary",
                    ).style(full_width=True)
                    output = gr.Audio(
                        label="Generated Output",
                        type="filepath",
                    )

    submit.click(
        generate,
        inputs=[
            model,
            text,
            melody,
            duration,
            topk,
            topp,
            temperature,
            cfg_coef,
            base_duration,
            sliding_window_seconds,
            continue_file,
            cf_cutoff,
            sc_text,
            seed,
        ],
        outputs=[output],
    )
    # gr.Examples(
    #     fn=generate,
    #     examples=[
    #         [
    #             "An 80s driving pop song with heavy drums and synth pads in the background",
    #             "./assets/bach.mp3",
    #             "melody",
    #         ],
    #         ["A cheerful country song with acoustic guitars", "./assets/bolero_ravel.mp3", "melody"],
    #         ["90s rock song with electric guitar and heavy drums", None, "medium"],
    #         [
    #             "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
    #             "./assets/bach.mp3",
    #             "melody",
    #         ],
    #         [
    #             "lofi slow bpm electro chill with organic samples",
    #             None,
    #             "medium",
    #         ],
    #     ],
    #     inputs=[text, melody, model],
    #     outputs=[output],
    # )
    gr.Markdown(
        """
        This is a webui for MusicGen with 30+ second generation support.

        Models
        1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
        2. Small -- a 300M transformer decoder conditioned on text only.
        3. Medium -- a 1.5B transformer decoder conditioned on text only.
        4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.) - recommended for continuing songs

        When the optional melody conditioning wav is provided, the model will extract
        a broad melody and try to follow it in the generated samples. Only the first chunk of the song will
        be generated with melody conditioning, the others will just continue on the first chunk.

        Base duration of 30 seconds is recommended.

        Sliding window of 10/15/20 seconds is recommended.

        When continuing songs, a continuing song cutoff of 5 seconds gives good results.

        Gradio analytics are disabled.
        """
    )

if __name__ == "__main__":
    while True:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7778,
            width="100%",
            show_error=True,
            enable_queue=True,
        )
