# pip install transformers peft sentence-transformers accelerate  # (accelerate not strictly needed on Mac)
import os, platform, importlib.util, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sentence_transformers import CrossEncoder


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
