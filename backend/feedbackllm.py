# pip install transformers peft sentence-transformers accelerate  # (accelerate not strictly needed on Mac)
import os, platform, importlib.util, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sentence_transformers import CrossEncoder
from huggingface_hub import login
import os

login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

ADAPTER_ID = "CodeSlayer/finetuned-feedback-model"


peft_cfg = PeftConfig.from_pretrained(ADAPTER_ID)
BASE_ID = peft_cfg.base_model_name_or_path or "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Base model: {BASE_ID}")


use_mps = torch.backends.mps.is_available()
device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
dtype  = torch.float16 if device in ["cuda", "mps"] else torch.float32


tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=dtype,
    device_map=None,          
    trust_remote_code=True,
)
base.to(device)


model = PeftModel.from_pretrained(base, ADAPTER_ID)

model = model.merge_and_unload()
model.to(device)

# ----- Aux models (unchanged) -----
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ======= your functions (minor safe tweaks) =======

def generate_feedback(question, answer, tone, emotion):
    prompt = f"""You are an expert interviewer. Based on the question and answer, tone, and emotion, analyze the answers delivered by the user and give constructive, personalized feedback indicating strengths, weaknesses, and areas of improvement, including commentary on tone and emotions.

Question: {question}
Answer: {answer}
Tone Analysis: {tone}
Emotion Analysis: {emotion}
Feedback:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = decoded.replace(prompt.strip(), "").strip()
    if "Overall" in cleaned:
        cleaned = cleaned.split("Overall")[0].strip()
    if not cleaned.startswith("Feedback:"):
        cleaned = "Feedback: " + cleaned
    return cleaned

def generate_ideal_answer(question):
    prompt = f"Provide a technically sound, concise and clear ideal answer for the following question:\n{question}\nIdeal Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output.sequences[0][input_len:]
    ideal_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return ideal_answer

def compute_model_confidence(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output.sequences[0][input_len:]
    scores = output.scores
    log_probs = []
    for i, token_id in enumerate(generated_ids):
        logits = scores[i][0]
        log_softmax = F.log_softmax(logits, dim=-1)
        log_probs.append(log_softmax[token_id].item())
    avg_log_prob = sum(log_probs) / max(1, len(log_probs))
    conf = round(torch.exp(torch.tensor(avg_log_prob)).item() * 10, 2)
    return max(0.0, min(conf, 10.0))

def compute_accuracy_score(answer, ideal_answer):
    raw_score = cross_encoder.predict([(answer, ideal_answer)])[0]
    normalized = max(0.0, min(float(raw_score), 1.0))
    return round(2 + normalized * 8, 2)

def heuristic_tone_emotion_score(tone, emotion):
    tone = (tone or "").lower()
    emotion = (emotion or "").lower()
    pos_tones = {"confident", "enthusiastic", "engaging"}
    pos_emotions = {"happy", "pleasant", "pleasant_surprised", "surprise"}
    neg_emotions = {"sad", "anger", "disgust", "fear"}
    score = 5.0
    if tone in pos_tones:
        score += 2.5
    if emotion in pos_emotions:
        score += 2.0
    elif emotion in neg_emotions:
        score -= 1.0
    return max(0.0, min(score, 10.0))

def calculate_final_score(accuracy_score, tone_emotion_score, model_confidence):
    final = 0.5*accuracy_score + 0.3*tone_emotion_score + 0.2*model_confidence
    return round(max(0.0, min(final, 10.0)), 2)

def run_feedback_pipeline(question, answer, tone, emotion):
    feedback_output = generate_feedback(question, answer, tone, emotion)
    ideal_answer = generate_ideal_answer(question)
    prompt_for_conf = f"""You are an expert interviewer...
Question: {question}
Answer: {answer}
Tone Analysis: {tone}
Emotion Analysis: {emotion}
Feedback:"""
    model_conf = compute_model_confidence(prompt_for_conf)
    accuracy_score = compute_accuracy_score(answer, ideal_answer)
    tone_emotion_score = heuristic_tone_emotion_score(tone, emotion)
    final_score = calculate_final_score(accuracy_score, tone_emotion_score, model_conf)

    print("Question:", question)
    print("Answer:", answer)
    print("Ideal Answer:", ideal_answer)
    for line in feedback_output.splitlines():
        if line.strip():
            print(line.strip())
    print("Model Confidence:", model_conf)
    print("Accuracy Score:", accuracy_score)
    print("Tone/Emotion Score:", tone_emotion_score)
    print("Final Score:", final_score)

if __name__ == "__main__":
    q = input("Enter the interview question: ")
    a = input("Enter your answer: ")
    t = input("Enter the detected tone: ")
    e = input("Enter the detected emotion: ")
    run_feedback_pipeline(q, a, t, e)



# # pip install transformers peft sentence-transformers accelerate  # peft only if you later add a Llama LoRA
# import os, torch, torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import CrossEncoder
# login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
# # ------------ Base model: Llama 3.2 3B Instruct ------------
# BASE_ID = "meta-llama/Llama-3.2-3B-Instruct"  # you need HF access to Meta Llama
# # If access is an issue, alternatives:
# # BASE_ID = "Qwen/Qwen2.5-3B-Instruct" or "microsoft/Phi-3.5-mini-instruct"

# device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
# dtype = torch.bfloat16 if supports_bf16 else (torch.float16 if device in ["cuda","mps"] else torch.float32)

# tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_ID,
#     torch_dtype=dtype,
#     device_map="auto",   # lets HF place weights on GPU/CPU
# )
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# # ----- Aux models (unchanged) -----
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # ========== Helpers for chat templating ==========
# def chat_inputs(system_prompt, user_prompt):
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": user_prompt})
#     toks = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     )
#     return toks.to(model.device)

# # ========== Your functions (adapted to chat) ==========
# SYSTEM_COACH = (
#     "You are an expert technical interviewer and a supportive coach. "
#     "Be concise, specific, and actionable."
# )

# def generate_feedback(question, answer, tone, emotion, max_new_tokens=300):
#     user = (
#         "Analyze the candidate's answer and provide constructive, personalized feedback. "
#         "Include strengths, weaknesses, and 2–3 specific improvements, referencing tone and emotions.\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n"
#         f"Tone Analysis: {tone}\n"
#         f"Emotion Analysis: {emotion}\n"
#         "Return only the feedback text."
#     )
#     inputs = chat_inputs(SYSTEM_COACH, user)
#     with torch.inference_mode():
#         out = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()
#     return decoded

# def generate_ideal_answer(question, max_new_tokens=150):
#     user = (
#         "Provide a concise, technically sound, clear ideal answer (5–8 lines) for the question below.\n\n"
#         f"{question}\n\nIdeal Answer:"
#     )
#     inputs = chat_inputs(SYSTEM_COACH, user)
#     in_len = inputs.shape[1]
#     with torch.inference_mode():
#         out = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             return_dict_in_generate=True,
#             output_scores=True,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     gen_ids = out.sequences[0][in_len:]
#     return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# def compute_model_confidence(question, answer, tone, emotion, max_new_tokens=200):
#     user = (
#         "You will generate feedback (do it normally)."
#         "\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n"
#         f"Tone Analysis: {tone}\n"
#         f"Emotion Analysis: {emotion}\n"
#         "Feedback:"
#     )
#     inputs = chat_inputs(SYSTEM_COACH, user)
#     in_len = inputs.shape[1]
#     with torch.inference_mode():
#         out = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,                 # we want token-level probs of a sampled path
#             return_dict_in_generate=True,
#             output_scores=True,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     gen_ids = out.sequences[0][in_len:]
#     scores = out.scores  # list of logits per generated step
#     logps = []
#     for i, tok_id in enumerate(gen_ids):
#         logits = scores[i][0]
#         logps.append(torch.log_softmax(logits, dim=-1)[tok_id].item())
#     avg_logp = sum(logps)/max(1,len(logps))
#     conf = float(torch.exp(torch.tensor(avg_logp)).item() * 10)
#     return round(max(0.0, min(conf, 10.0)), 2)

# def compute_accuracy_score(answer, ideal_answer):
#     raw = cross_encoder.predict([(answer, ideal_answer)])[0]
#     normalized = max(0.0, min(float(raw), 1.0))
#     return round(2 + normalized * 8, 2)

# def heuristic_tone_emotion_score(tone, emotion):
#     tone = (tone or "").lower()
#     emotion = (emotion or "").lower()
#     pos_tones = { "Happy", "neutral", "sad", "pleasant suprise"}
#     neg_tones={"Angry","Disgust","Fear","sad"}
#     pos_emotions = {"Happy",  "Surprise",  "Neutral"}
#     neg_emotions = {"Sad", "Anger",  "Disgust", "Fear"}
#     score = 5.0
#     if tone in pos_tones:
#         score += 2.5
#     if emotion in pos_emotions:
#         score += 2.0
#     elif emotion in neg_emotions:
#         score -= 1.0
#     return max(0.0, min(score, 10.0))

# def calculate_final_score(accuracy_score, tone_emotion_score, model_confidence):
#     final = 0.5*accuracy_score + 0.3*tone_emotion_score + 0.2*model_confidence
#     return round(max(0.0, min(final, 10.0)), 2)

# def run_feedback_pipeline(question, answer, tone, emotion):
#     feedback_output = generate_feedback(question, answer, tone, emotion)
#     ideal_answer = generate_ideal_answer(question)
#     model_conf = compute_model_confidence(question, answer, tone, emotion)
#     accuracy_score = compute_accuracy_score(answer, ideal_answer)
#     tone_emotion_score = heuristic_tone_emotion_score(tone, emotion)
#     final_score = calculate_final_score(accuracy_score, tone_emotion_score, model_conf)

#     print("Question:", question)
#     print("Answer:", answer)
#     print("Ideal Answer:", ideal_answer)
#     print("Feedback:", feedback_output)
#     print("Model Confidence:", model_conf)
#     print("Accuracy Score:", accuracy_score)
#     print("Tone/Emotion Score:", tone_emotion_score)
#     print("Final Score:", final_score)

# if __name__ == "__main__":
#     q = input("Enter the interview question: ")
#     a = input("Enter your answer: ")
#     t = input("Enter the detected tone: ")
#     e = input("Enter the detected emotion: ")
#     run_feedback_pipeline(q, a, t, e)
