import os
import io
import time
import logging
import wave
from pathlib import Path

import numpy as np
import s3tokenizer
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml

def fade_in_out(fade_in_mel:torch.Tensor, fade_out_mel:torch.Tensor, window:torch.Tensor):
    """perform fade_in_out in tensor style
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class Token2wav():
    CHUNK_SIZE = 25
    WARMUP_TOKENS = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]

    def __init__(self, model_path, float16=False, prompt_wav:str="assets/default_female.wav", **kwargs):
        self.float16 = float16

        logging.info(f"init token2wav ...")
        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx")

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.audio_tokenizer.to(self.device).eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.to(self.device).eval()

        if torch.cuda.is_available() and kwargs.get("cuda_graph", False):
            logging.info(f"move token2wav flow to cuda and scatter_cuda_graph ...")
            self.flow.scatter_cuda_graph(True)

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        self.cache = {}
        self.stream_cache = None

        # stream conf
        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)   # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(self.device)

        # hifigan cache
        self.hift_cache_dict = {}

        self._verbose = kwargs.get("verbose", False)

        if kwargs.get("warmup_cn", 0) > 0:
            self.warmup(prompt_wav, kwargs.get("warmup_cn"))


    def _prepare_prompt(self, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device=self.device)

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]), mode='replicate')
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens, prompt_wav):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device)

        with torch.amp.autocast(self.device, dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)

        wav, _ = self.hift(speech_feat=mel)
        output = io.BytesIO()
        torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')

        return output.getvalue()

    def set_stream_cache(self, prompt_wav):
        logging.info(f"set_stream_cache {prompt_wav}")
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]
        self.stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1),
            prompt_mels, spk_emb, n_timesteps=10)
        # hift cache
        self.hift_cache_dict = dict(
            mel = torch.zeros(1, prompt_mels.shape[2], 0, device=self.device), 
            source = torch.zeros(1, 1, 0, device=self.device),
            speech = torch.zeros(1, 0, device=self.device),
        )

    def stream(self, generated_speech_tokens, prompt_wav, last_chunk=False):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device)

        if self.stream_cache is None:
            # raise ValueError("stream_cache is not set")
            logging.warning("stream_cache is not set, then set it")
            self.set_stream_cache(prompt_wav)

        if self._verbose is True:
            for k, v in self.stream_cache.items():
                print(f"{k}: {v.shape}")
        
        with torch.amp.autocast(self.device, dtype=torch.float16 if self.float16 else torch.float32):
            start = time.time()
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=10,
            )
            elapsed = time.time() - start
            if self._verbose is True:
                print(f"flow inference_chunk time: {elapsed:.3f}s")
        if self.stream_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache['estimator_att_cache'] = torch.cat([
                self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                self.stream_cache['estimator_att_cache'][:, :, :, :, -100:],
            ], dim=4)
        
        # vocoder cache
        hift_cache_mel = self.hift_cache_dict['mel']
        hift_cache_source = self.hift_cache_dict['source']
        hift_cache_speech = self.hift_cache_dict['speech']
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        start = time.time()
        speech, source = self.hift(mel, hift_cache_source)
        elapsed = time.time() - start
        if self._verbose is True:
            print(f"hift inference_chunk time: {elapsed:.3f}s")

        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # update vocoder cache
        self.hift_cache_dict = dict(
            mel = mel[..., -self.mel_cache_len:].clone().detach(),
            source = source[:, :, -self.source_cache_len:].clone().detach(),
            speech = speech[:, -self.source_cache_len:].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]

        wav_np = speech.cpu().numpy()
        # Clip to [-1, 1] to avoid overflow, then scale to int16
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes
    
    def warmup(self, prompt_wav, warmup_cn:int=2):
        if warmup_cn > 0:
            for i in range(warmup_cn):
                start = time.time()
                self.stream(self.WARMUP_TOKENS[:self.CHUNK_SIZE + self.flow.pre_lookahead_len], prompt_wav=prompt_wav) # Warm up
                logging.info(f"Token2wav warmup {i=} done in {time.time() - start:.3f}s")
            # NOTE: reset stream cache
            self.set_stream_cache(prompt_wav)


def test_stream(token2wav: Token2wav):
    prompt_wav = "assets/default_male.wav"
    #prompt_wav = "assets/default_female.wav"
    token2wav.set_stream_cache(prompt_wav)
    token2wav.warmup(prompt_wav, warmup_cn=2)

    buffer = []
    pcm = b""

    output_file = Path("output_chunks_stream_tts.wav")
    output_file.unlink(missing_ok=True)

    for audio_token_id in token2wav.WARMUP_TOKENS:
        buffer.append(audio_token_id)
        if len(buffer) >= token2wav.CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
            start = time.time()
            output = token2wav.stream(
                buffer[: token2wav.CHUNK_SIZE + token2wav.flow.pre_lookahead_len],
                prompt_wav=prompt_wav,
                last_chunk=False,
            )
            print(len(buffer), len(output), output[:50],time.time()-start)
            pcm += output
            buffer = buffer[token2wav.CHUNK_SIZE:]

    if len(buffer) > 0:
        start = time.time()
        output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
        print("last_chunk", len(buffer), len(output), output[:50], time.time()-start)
        pcm += output
    with wave.open(str(output_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)

def test_token2wav(token2wav:Token2wav):
    audio = token2wav(token2wav.WARMUP_TOKENS, 'assets/default_male.wav')
    with open('give_me_a_brief_introduction_to_the_great_wall.wav', 'wb') as f:
        f.write(audio)

"""
python -m token2wav
TEST_FUNC=test_stream python -m token2wav
"""
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # huggingface-cli download stepfun-ai/Step-Audio-2-mini --include token2wav/ --local-dir Step-Audio-2-mini
    token2wav = Token2wav('Step-Audio-2-mini/token2wav')

    test_func = os.getenv("TEST_FUNC","test_token2wav")
    globals()[test_func](token2wav)
