# Q* 

This project trains a "student" language model (LM) using responses from a "teacher" LM to enhance its understanding and generation capabilities in a self-play setup. The goal is to iteratively improve the student model by comparing its outputs with those of the more advanced teacher model and adjusting based on the differences. 

## Overview

The script employs Transformers models from Hugging Face to perform a chain-of-thought reasoning generation where a student model learns to improve its answers based on a teacher model's output. The process involves generating an answer, evaluating it against the teacher's answer, and adjusting the student model using KL-divergence loss.


