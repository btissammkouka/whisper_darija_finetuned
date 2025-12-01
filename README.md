# **Whisper Fine-Tuning for Moroccan Darija  🇲🇦 (Latin Script) **

This repository contains a Jupyter Notebook for fine-tuning OpenAI's **Whisper Small** model to transcribe Moroccan Darija audio directly into **Latin script (Arabizi/Chat Darija)**.

Standard Whisper models typically transcribe Darija into Arabic script. This project uses **LoRA (Low-Rank Adaptation)** to efficiently retrain the model to switch writing systems and recognize dialect-specific phonetics without requiring massive computational resources.

## **🚀 Key Features**

* **Model:** openai/whisper-small  
* **Technique:** PEFT (Parameter-Efficient Fine-Tuning) using **LoRA**.  
* **Objective:** Force the model to output Latin script (e.g., "kifach") instead of Arabic script (e.g., "كيفاش").  
* **Evaluation:** Comprehensive evaluation using both **WER** (Word Error Rate) and **CER** (Character Error Rate).

## **📂 Dataset**

The project utilizes two primary datasets from Hugging Face:

1. **Training:** [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)  
   * Filtered to ensure valid audio and non-empty Latin transcriptions (darija\_Latn).  
2. **Testing/Evaluation:** [Snousnou/Moroccan-Darija-ASR](https://www.google.com/search?q=https://huggingface.co/datasets/Snousnou/Moroccan-Darija-ASR)  & [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)

## **🛠️ Installation & Dependencies**

To replicate this notebook, you will need the following libraries:

pip install transformers datasets librosa jiwer accelerate peft bitsandbytes

## **🧠 Methodology**

### **1\. Data Preparation**

The notebook processes audio into **log-mel input features** (sampling rate: 16kHz) and tokenizes labels using the standard Whisper tokenizer.

### **2\. LoRA Configuration**

We use LoRA to train only specific attention layers, significantly reducing memory usage:

* **Rank (r):** 8  
* **Target Modules:** k\_proj, v\_proj  
* **Task:** SEQ\_2\_SEQ\_LM

### **3\. Training**

* **Batch Size:** 2 (with gradient accumulation steps \= 4\)  
* **Max Steps:** 500  
* **FP16:** Enabled for training efficiency.

## **📊 Results & Performance**

The evaluation highlights a significant improvement in **script switching**. While WER remains high due to segmentation issues (e.g., writing "ghan mchi" instead of "ghanmchi"), the **CER (Character Error Rate)** proves the model successfully learned the Latin orthography.

### **Quantitative Metrics**

The following metrics represent the average scores over the test set:

| Metric | Base Model (Whisper Small) | LoRA Fine-Tuned Model | Improvement |
| :---- | :---- | :---- | :---- |
| **WER** (Word Error Rate) | 1.190 | 1.103 | Slight |
| **CER** (Char Error Rate) | 0.928 | 0.331 | **Major** |

### **Analysis**

* **Why is WER high (\~1.10)?** The model often splits words incorrectly (e.g., ho ma instead of homa). Since WER penalizes every split heavily, the score looks poor despite the phonetic transcription being accurate.  
* **Why is CER low (\~0.33)?** The CER of **0.331** confirms that the model is getting roughly 67% of the characters correct. In contrast, the Base Model has a CER of **0.928**, confirming that it is outputting a completely different script (Arabic) that shares almost no characters with the Latin reference.

### **Qualitative Comparison**

| Context | Reference | Base Output (Arabic) | LoRA Output (Latin) | CER |
| :---- | :---- | :---- | :---- | :---- |
| **Ex 1** | homa mkhbbyin chi haja | هم مخبنشي حاجة | ho ma mkhbin chi 3aja | 0.278 |
| **Ex 2** | bayna homa tay7awlo | بينهما كحولوا | baina homa ki7awlo | 0.265 |
| **Ex 3** | loTilat mabaynach | لو طيلات ما بيناش | lotaylat mbeynach | 0.237 |
| **Ex 10** | bghit n3ref ch7al | أريد أن نعرف | bghit n3ra fch | 0.310 |

## **💻 Usage**

### **Inference with the Fine-Tuned Model**

from transformers import WhisperProcessor, WhisperForConditionalGeneration  
from peft import PeftModel  
import torch

\# 1\. Load Processor and Base Model  
processor \= WhisperProcessor.from\_pretrained("./whisper-darja-lora")  
base\_model \= WhisperForConditionalGeneration.from\_pretrained("openai/whisper-small")

\# 2\. Load LoRA Adapters  
model \= PeftModel.from\_pretrained(base\_model, "./whisper-darja-lora")  
model.eval()

\# 3\. Transcribe  
generated\_ids \= model.base\_model.generate(  
    input\_features,  
    max\_length=128,  
    forced\_decoder\_ids=processor.get\_decoder\_prompt\_ids(language="ar", task="transcribe")  
)  
transcription \= processor.batch\_decode(generated\_ids, skip\_special\_tokens=True)\[0\]  
print(transcription)

## **🚧 Future Improvements**

* **Text Normalization:** Implement a scoring metric that handles variable Arabizi spelling (e.g., ignoring case, merging 3 and a, fixing splits) to reduce WER.  
* **Increase Training:** 500 steps was sufficient for script switching, but more training is needed to fix word segmentation ambiguities.
* **Use another metric for synonym problem using BERT:** The model sometimes substitutes the specific dialect word used in the reference with a common synonym (Taychof # Kaychof)
