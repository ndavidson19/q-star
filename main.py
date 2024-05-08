import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_scheduler,
    pipeline
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import numpy as np
from tqdm import tqdm
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and Dataset Setup
teacher_model_id = "gpt2-xl"
assistant_model_id = "gpt2-medium"
student_model_id = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(teacher_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)  # Converting pad_token to its ID


teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_id)
assistant_model = GPT2LMHeadModel.from_pretrained(assistant_model_id)
student_model = GPT2LMHeadModel.from_pretrained(student_model_id)

dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Evaluation Metrics
def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Chain-of-Thought Reasoning Generation
def generate_chain_of_thought(model, prompt, max_steps=3, max_length=50, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    reasoning_chain = prompt
    for step in range(max_steps):
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.shape[1] + max_length, 
                pad_token_id=pad_token_id  # Convert pad token to ID
            )
        step_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        reasoning_chain += f" Step {step + 1}: {step_output.strip()}"
        input_ids = tokenizer.encode(reasoning_chain, return_tensors="pt").to(device)
    
    logger.debug(f"Chain of Thought: {reasoning_chain}")
    return reasoning_chain


# Scoring Function (Placeholder: Customize based on task)
sentiment_analysis = pipeline("sentiment-analysis")
def score_answer(answer, label):
    result = sentiment_analysis(answer)
    sentiment_score = 1 if result[0]['label'] == 'POSITIVE' else 0

    # Assuming label is 1 for positive and 0 for negative
    return 1 if sentiment_score == label else 0


# Self-Play Training Function
def train_self_play(student_model, teacher_model, assistant_model, dataloader, criterion, optimizer, lr_scheduler, device):
    student_model.train()
    teacher_model.eval()
    assistant_model.eval()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Generate reasoning prompt using Teacher's Assistant (Chain-of-Thought)
        prompt = "Answer the following question step by step: " + tokenizer.decode(input_ids[0], skip_special_tokens=True)
        chain_of_thought = generate_chain_of_thought(assistant_model, prompt, device=device)
        prompt_input_ids = tokenizer.encode(chain_of_thought, return_tensors="pt").to(device)

        logger.debug(f"Prompt Input IDs: {prompt_input_ids}")
        logger.debug(f"Chain of Thought: {chain_of_thought}")

        # Teacher's Answer
        with torch.no_grad():
            teacher_outputs = teacher_model.generate(
                prompt_input_ids, max_length=prompt_input_ids.shape[1] + 50, pad_token_id=pad_token_id
            )
            teacher_logits = teacher_model(prompt_input_ids).logits
            teacher_probs = nn.functional.softmax(teacher_logits, dim=-1)
        teacher_answer = tokenizer.decode(teacher_outputs[0][prompt_input_ids.shape[1]:], skip_special_tokens=True)
        teacher_score = score_answer(teacher_answer, labels[0].cpu().item())

        # Student's Answer
        student_outputs = student_model(prompt_input_ids, attention_mask=None)
        student_logits = student_outputs.logits[:, prompt_input_ids.shape[1]:, :]
        student_answer = tokenizer.decode(torch.argmax(student_logits, dim=-1)[0], skip_special_tokens=True)
        student_score = score_answer(student_answer, labels[0].cpu().item())

        # Aligning the teacher_probs with student_logits
        if teacher_probs.size(1) > student_logits.size(1):
            teacher_probs = teacher_probs[:, :student_logits.size(1), :]

        print(f"Teacher Answer: {teacher_answer}, Score: {teacher_score}")
        print(f"Student Answer: {student_answer}, Score: {student_score}")

        # Update student model only if its performance is worse than the teacher's
        if student_score < teacher_score:
            teacher_probs = nn.functional.softmax(teacher_outputs[:, prompt_input_ids.shape[1]:].float(), dim=-1)
            loss = criterion(nn.functional.log_softmax(student_logits, dim=-1).view(-1, student_logits.size(-1)), teacher_probs.view(-1, teacher_probs.size(-1)))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1) if batch_idx + 1 > 0 else 0
        progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logger.info(f"Average Training Loss: {avg_loss:.4f}")


def train_self_play(student_model, teacher_model, assistant_model, dataloader, criterion, optimizer, lr_scheduler, device):
    student_model.train()
    teacher_model.eval()
    assistant_model.eval()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        #prompt = "Answer the following question step by step: " + tokenizer.decode(input_ids[0], skip_special_tokens=True) For math and coding
        prompt = "Respond to the following passage with what sentiment you feel. Reason through this step by step: " + tokenizer.decode(input_ids[0], skip_special_tokens=True)
        chain_of_thought = generate_chain_of_thought(assistant_model, prompt, device=device)
        prompt_input_ids = tokenizer.encode(chain_of_thought, return_tensors="pt").to(device)

        # Generate teacher's output
        with torch.no_grad():
            teacher_outputs = teacher_model.generate(prompt_input_ids, max_length=prompt_input_ids.shape[1] + 50, pad_token_id=pad_token_id)
            teacher_logits = teacher_model(prompt_input_ids).logits
            teacher_probs = nn.functional.softmax(teacher_logits, dim=-1)

        # Teacher's Answer
        teacher_answer = tokenizer.decode(teacher_outputs[0][prompt_input_ids.shape[1]:], skip_special_tokens=True)
        teacher_score = score_answer(teacher_answer, labels[0].cpu().item())

        # Generate student's output
        student_outputs = student_model(prompt_input_ids)
        student_logits = student_outputs.logits

        # Aligning the teacher_probs with student_logits
        if teacher_probs.size(1) > student_logits.size(1):
            teacher_probs = teacher_probs[:, :student_logits.size(1), :]

        # Loss calculation
        loss = criterion(nn.functional.log_softmax(student_logits, dim=-1), teacher_probs)
        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        logger.info(f"Teacher Answer: {teacher_answer}, Score: {teacher_score}")
        logger.info(f"Student Answer: {tokenizer.decode(student_logits.argmax(dim=-1)[0], skip_special_tokens=True)}, Score: {score_answer(teacher_answer, labels[0].cpu().item())}")

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logger.info(f"Average Training Loss: {avg_loss:.4f}")


# Testing Function
def evaluate_student(student_model, dataloader, device):
    student_model.eval()
    all_predictions = []
    all_labels = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            prompt = "Answer the following question step by step: " + tokenizer.decode(input_ids[0], skip_special_tokens=True)
            prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = student_model.generate(
                prompt_input_ids, max_length=prompt_input_ids.shape[1] + 50, pad_token_id=pad_token_id
            )
            predictions = tokenizer.decode(outputs[0][prompt_input_ids.shape[1]:], skip_special_tokens=True)

        all_predictions.append(1 if "positive" in predictions else 0 if "negative" in predictions else -1)
        all_labels.append(labels[0].cpu().item())

    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

def interactive_session(student_model, teacher_model, dataloader, device, optimizer):
    student_model.train()
    conversation_log = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            teacher_output = teacher_model.generate(input_ids, max_length=50, pad_token_id=pad_token_id)
        
        student_output = student_model.generate(input_ids, max_length=50, pad_token_id=pad_token_id)
        
        teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
        student_text = tokenizer.decode(student_output[0], skip_special_tokens=True)

        conversation_log.append((teacher_text, student_text))

        feedback_input_ids = tokenizer.encode(teacher_text, return_tensors='pt').input_ids.to(device)
        student_outputs = student_model(feedback_input_ids, labels=feedback_input_ids)
        loss = student_outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return conversation_log


# Curriculum-Based Training with Interactive Self-Play
def curriculum_training_self_play(student_model, teacher_model, assistant_model, train_dataloader, test_dataloader, device):
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.KLDivLoss(reduction="batchmean")
    num_training_steps = len(train_dataloader) * 3
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}: Self-Play Training")
        train_self_play(student_model, teacher_model, assistant_model, train_dataloader, criterion, optimizer, lr_scheduler, device)

        logger.info(f"Epoch {epoch + 1}: Interactive Session")
        interactive_log = interactive_session(student_model, teacher_model, train_dataloader, device, optimizer)
        logger.info("Interactive Session Log: {}".format(interactive_log))

        # Intermediate Evaluation (Mid-Term Exam)
        logger.info(f"Epoch {epoch + 1}: Mid-Term Evaluation")
        evaluate_student(student_model, test_dataloader, device)

    # Final Exam
    logger.info("Final Exam Evaluation")
    evaluate_student(student_model, test_dataloader, device)

    # Check if the student model has surpassed the teacher model
    student_metrics = evaluate_student(student_model, test_dataloader, device)
    teacher_metrics = evaluate_student(teacher_model, test_dataloader, device)

    # If the student surpasses the teacher, make the student the new teacher
    if student_metrics["accuracy"] > teacher_metrics["accuracy"]:
        logger.info("Student has surpassed the teacher! Promoting student to new teacher.")
        for param_student, param_teacher in zip(student_model.parameters(), teacher_model.parameters()):
            param_teacher.data.copy_(param_student.data)


# Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
teacher_model.to(device)
assistant_model.to(device)
student_model.to(device)

curriculum_training_self_play(student_model, teacher_model, assistant_model, train_dataloader, test_dataloader, device)
