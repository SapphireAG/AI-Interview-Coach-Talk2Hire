# install sentence-transformers bitsandbytes before running

from huggingface_hub import login
login(token="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")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model_name = "CodeSlayer/finetuned-feedback-model"
quant_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",
    quantization_config=quant_config
)



def generate_feedback(question, answer, tone, emotion):
    prompt = f"""You are an expert interviewer. Based on the question and answer, tone, and emotion, analyze the answers delivered by the user and give constructive, personalized feedback indicating strengths, weaknesses, and areas of improvement, including commentary on tone and emotions.

Question: {question}
Answer: {answer}
Tone Analysis: {tone}
Emotion Analysis: {emotion}
Feedback:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_output = decoded.replace(prompt.strip(), "").strip()

    if "Overall" in cleaned_output:
        cleaned_output = cleaned_output.split("Overall")[0].strip()

    if not cleaned_output.startswith("Feedback:"):
        cleaned_output = "Feedback: " + cleaned_output

    return cleaned_output

def generate_ideal_answer(question):
    prompt = f"Provide a technically sound, concise and clear ideal answer for the following question:\n{question}\nIdeal Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

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
        token_log_prob = log_softmax[token_id].item()
        log_probs.append(token_log_prob)

    avg_log_prob = sum(log_probs) / len(log_probs)
    return round(torch.exp(torch.tensor(avg_log_prob)).item() * 10, 2)

def compute_accuracy_score(answer, ideal_answer):
    raw_score = cross_encoder.predict([(answer, ideal_answer)])[0]
    normalized = max(0.0, min(raw_score, 1.0))
    adjusted_score = 2 + (normalized * 8)
    return round(adjusted_score, 2)


def heuristic_tone_emotion_score(tone, emotion):
    positive_tones = ["confident", "enthusiastic", "engaging",]
    positive_emotions = ["Happy", "Surprise", "Sad","Anger","Disgust","Fear","Neutral","Pleasant"]
    score = 5.0
    if tone.lower() in positive_tones:
        score += 2.5
    if emotion.lower() in positive_emotions:
        score += 2.5
    return min(score, 10.0)

def calculate_final_score(accuracy_score, tone_emotion_score, model_confidence):
    final_score = (
        0.5 * accuracy_score +
        0.3 * tone_emotion_score +
        0.2 * model_confidence
    )
    return round(final_score, 2)

def run_feedback_pipeline(question, answer, tone, emotion):
    feedback_output = generate_feedback(question, answer, tone, emotion)
    ideal_answer = generate_ideal_answer(question)

    prompt_for_conf = f"""You are an expert interviewer. Based on the question and answer, tone, and emotion, analyze the answers delivered by the user and give constructive, personalized feedback indicating strengths, weaknesses, and areas of improvement, including commentary on tone and emotions.

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

    lines = feedback_output.splitlines()
    for line in lines:
        if line.strip():
            print(line.strip())

    print("Model Confidence:", model_conf)
    print("Accuracy Score:", accuracy_score)
    print("Tone/Emotion Score:", tone_emotion_score)
    print("Final Score:", final_score)

question = input("Enter the interview question: ")
answer = input("Enter your answer: ")
tone = input("Enter the detected tone: ")
emotion = input("Enter the detected emotion: ")

run_feedback_pipeline(question, answer, tone, emotion)
