def think_test(model, token2wav):
    history = [{"role": "system", "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append(
            {"role": "human", "content": [{"type": "audio", "audio": inp_audio}]}
        )
        # get think content, stop when "</think>" appears
        history.append({"role": "assistant", "content": "\n<think>\n", "eot": False})
        _, think_content, _ = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True, stop_strings=['</think>'])
        print('<think>' + think_content + '>')
        # get audio response
        history[-1]["content"] += think_content + ">\n\n<tts_start>"
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(text)
        audio = [x for x in audio if x < 6561] # remove audio padding
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open(f'output-round-{round_idx}-think.wav', 'wb') as f:
            f.write(audio)
        # remove think content from history
        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content":[
                    {"type": "text", "text":"<tts_start>"},
                    {"type":"token", "token": tokens}
                ]
            }
        )

def think_test_vllm(model, token2wav):
    history = [{"role": "system", "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"}]
    for round_idx, inp_audio in enumerate([
        "assets/give_me_a_brief_introduction_to_the_great_wall.wav"
        #"assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        #"assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": "<think>", "eot": False})
        #_, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=[{"token": "</think>"}])
        _, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=["</think>"])
        print('<think>' + think_content + '</think>')
        history[-1]["content"] += think_content + "</think>" + "\n\n<tts_start>"
        response, text, audio = model(history, max_tokens=2048, temperature=0.7, repetition_penalty=1.05)
        print(text, audio)
        if audio:
            audio = [x for x in audio if x < 6561]
            audio = token2wav(audio, prompt_wav='assets/default_female.wav')
            with open(f'output-round-{round_idx}-think.wav', 'wb') as f:
                f.write(audio)
        history.pop(-1)
        history.append({"role": "assistant", "tts_content": response.get("tts_content", {})})

def think_test_vllm_stream(model, token2wav):
    history = [{"role": "system", "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"}]
    for round_idx, inp_audio in enumerate([
        "assets/give_me_a_brief_introduction_to_the_great_wall.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": "<think>", "eot": False})
        #_, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=[{"token": "</think>"}])
        _, think_content, _ = model(history, max_tokens=2048, temperature=0.7, stop=["</think>"])
        print('<think>' + think_content + '</think>')
        history[-1]["content"] += think_content + "</think>" + "\n\n<tts_start>"
        stream_iter = model.stream(history, max_tokens=2048, temperature=0.7, repetition_penalty=1.05)
        audio_tokens = []
        texts=""
        for response, text, audio in stream_iter:
            print(f"{response=} {text=} {audio=}")
            if audio:
                audio_tokens.extend(audio)
            if text:
                texts += text

        print(texts, audio_tokens)
        if audio_tokens:
            audio_tokens = [x for x in audio_tokens if x < 6561]
            audio = token2wav(audio_tokens, prompt_wav='assets/default_female.wav')
            with open(f'output-round-{round_idx}-think_stream.wav', 'wb') as f:
                f.write(audio)

def test_vllm():
    from stepaudio2vllm import StepAudio2
    from token2wav import Token2wav

    #model = StepAudio2("http://localhost:8000/v1/chat/completions", "step-audio-2-mini-think")
    model = StepAudio2("https://weege009--vllm-step-audio2-serve-dev.modal.run/v1/chat/completions", "step-audio-2-mini-think")
    token2wav = Token2wav('../../models/stepfun-ai/Step-Audio-2-mini/token2wav')
    think_test_vllm(model, token2wav)
    think_test_vllm_stream(model, token2wav)

def test_transformers():
    from stepaudio2 import StepAudio2
    from token2wav import Token2wav

    model = StepAudio2('Step-Audio-2-mini-Think')
    token2wav = Token2wav('Step-Audio-2-mini-Think/token2wav')
    think_test(model, token2wav)

"""
# run vllm serve
LLM_MODEL_NAME=step-audio-2-mini-think LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think IMAGE_GPU=L40s modal serve src/llm/vllm/step_audio2.py

python -m examples-think
"""
if __name__ == '__main__':
    test_vllm()

