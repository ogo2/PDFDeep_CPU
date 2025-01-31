import logging
import nltk
import os
import fitz  # PyMuPDF
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import json
from PIL import Image
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import sacrebleu
from dataclasses import dataclass
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from typing import List, Dict, Tuple, Any
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import time

nltk.download('punkt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# [CPU VERSION CHANGE] Убираем использование os.environ, связанное с CUDA.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

@dataclass
class VisualElement:
    type: str  # 'image', 'table', 'diagram', 'line', 'other'
    bbox: Tuple[float, float, float, float]
    page_num: int
    data: Any  # Содержимое элемента
    properties: Dict  # Дополнительные свойства

@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-ru"
    max_length: int = 512
    num_beams: int = 4
    train_ratio: float = 0.7
    eval_ratio: float = 0.15
    test_ratio: float = 0.15
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    num_epochs: int = 3
    preserve_visual_elements: bool = True
    visual_elements_dir: str = "./visual_elements"
    
    def __post_init__(self):
        total_ratio = self.train_ratio + self.eval_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Sum of ratios must be 1.0, got {total_ratio}")
        
        for ratio_name, ratio_value in [
            ("train_ratio", self.train_ratio),
            ("eval_ratio", self.eval_ratio),
            ("test_ratio", self.test_ratio)
        ]:
            if not (0 <= ratio_value <= 1):
                raise ValueError(f"{ratio_name} must be between 0 and 1")

def clean_text(text: str) -> str:
    """Clean text from PDF artifacts."""
    # Remove hyphenation
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove PDF artifacts (you may need to add more patterns)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()

class ParallelPDFDataset:
    def __init__(self, source_pdf_dir: str, target_pdf_dir: str, config: TranslationConfig):
        self.source_pdf_dir = source_pdf_dir
        self.target_pdf_dir = target_pdf_dir
        self.pairs = []
        self.aligned_texts = []
        self.config = config
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        text_blocks = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text += span['text'] + " "
                        
                        clean = clean_text(text)
                        if clean:
                            sentences = sent_tokenize(clean)
                            for sent in sentences:
                                if sent.strip():
                                    text_blocks.append({
                                        'text': sent.strip(),
                                        'bbox': block['bbox'],
                                        'page_num': page_num
                                    })
            if not text_blocks:
                logger.warning(f"No text extracted from {pdf_path}")
            return text_blocks
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return []

    def align_texts(self, source_blocks: List[Dict], target_blocks: List[Dict]) -> List[Tuple[str, str]]:
        """Improved text alignment using multiple criteria."""
        aligned_pairs = []
        
        def normalize_text(text: str) -> str:
            text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text.lower())
            return text.strip()
        
        def is_number_or_code(text: str) -> bool:
            return bool(re.match(r'^[0-9/-]*$', text.strip()))
        
        source_by_page = {}
        target_by_page = {}
        
        for block in source_blocks:
            page = block['page_num']
            if page not in source_by_page:
                source_by_page[page] = []
            if not is_number_or_code(block['text']):
                source_by_page[page].append(block)
                
        for block in target_blocks:
            page = block['page_num']
            if page not in target_by_page:
                target_by_page[page] = []
            if not is_number_or_code(block['text']):
                target_by_page[page].append(block)
        
        for page in source_by_page.keys():
            if page in target_by_page:
                source_page = sorted(source_by_page[page], key=lambda x: (x['bbox'][1], x['bbox'][0]))
                target_page = sorted(target_by_page[page], key=lambda x: (x['bbox'][1], x['bbox'][0]))
                
                s_idx = t_idx = 0
                while s_idx < len(source_page) and t_idx < len(target_page):
                    source_block = source_page[s_idx]
                    target_block = target_page[t_idx]
                    
                    source_text = source_block['text'].strip()
                    target_text = target_block['text'].strip()
                    
                    if not source_text or not target_text or \
                       is_number_or_code(source_text) or is_number_or_code(target_text):
                        s_idx += 1
                        t_idx += 1
                        continue
                    
                    vertical_distance = abs(source_block['bbox'][1] - target_block['bbox'][1])
                    len_ratio = len(source_text) / len(target_text) if len(target_text) > 0 else float('inf')
                    
                    if vertical_distance < 50 and 0.5 < len_ratio < 2.0:
                        if re.search('[a-zA-Zа-яА-Я]', source_text) and re.search('[a-zA-Zа-яА-Я]', target_text):
                            aligned_pairs.append((source_text, target_text))
                        s_idx += 1
                        t_idx += 1
                    elif source_block['bbox'][1] < target_block['bbox'][1]:
                        s_idx += 1
                    else:
                        t_idx += 1

        # Сохраняем промежуточные результаты для отладки
        with open('parallel_texts_debug.json', 'w', encoding='utf-8') as f:
            debug_data = {
                'source_blocks': [
                    {'text': b['text'], 'bbox': b['bbox'], 'page': b['page_num']}
                    for p in self.pairs for b in self.extract_text_from_pdf(p[0])
                ],
                'target_blocks': [
                    {'text': b['text'], 'bbox': b['bbox'], 'page': b['page_num']}
                    for p in self.pairs for b in self.extract_text_from_pdf(p[1])
                ],
                'aligned_pairs': self.aligned_texts
            }
            json.dump(debug_data, f, ensure_ascii=False, indent=2)
        
        filtered_pairs = []
        for source, target in aligned_pairs:
            if (len(source) > 5 and len(target) > 5 and 
                re.search('[a-zA-Zа-яА-Я]', source) and 
                re.search('[a-zA-Zа-яА-Я]', target)):
                filtered_pairs.append((source, target))
        
        logger.info(f"Found {len(filtered_pairs)} valid text pairs after filtering")
        return filtered_pairs
   
    def load_data(self):
        source_texts = []
        target_texts = []
        
        for pdf in os.listdir(self.source_pdf_dir):
            if pdf.endswith('.pdf'):
                source_blocks = self.extract_text_from_pdf(os.path.join(self.source_pdf_dir, pdf))
                if source_blocks:
                    source_texts.extend(source_blocks)
                    
        for pdf in os.listdir(self.target_pdf_dir):
            if pdf.endswith('.pdf'):
                target_blocks = self.extract_text_from_pdf(os.path.join(self.target_pdf_dir, pdf))
                if target_blocks:
                    target_texts.extend(target_blocks)

        if not source_texts or not target_texts:
            logger.error("No valid text found in PDFs")
            return [], [], []

        aligned_texts = self.align_texts(source_texts, target_texts)
        return self.split_data(aligned_texts)

    def split_data(self, aligned_texts):
        if not aligned_texts:
            logger.warning("No aligned texts found!")
            return [], [], []
        
        if len(aligned_texts) < 3:
            logger.warning("Too few samples to split into train/eval/test sets!")
            return aligned_texts, [], []
        
        train_data, temp_data = train_test_split(
            aligned_texts, 
            train_size=self.config.train_ratio,
            random_state=42
        )
        
        remaining_ratio = 1.0 - self.config.train_ratio
        if remaining_ratio > 0:
            eval_ratio_adjusted = self.config.eval_ratio / remaining_ratio
            
            eval_data, test_data = train_test_split(
                temp_data,
                train_size=eval_ratio_adjusted,
                random_state=42
            )
        else:
            eval_data = []
            test_data = []
        
        logger.info(f"Dataset split: train={len(train_data)}, eval={len(eval_data)}, test={len(test_data)}")
        return train_data, eval_data, test_data

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        
        source_encoded = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoded = self.tokenizer(
            text=target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoded['input_ids'].squeeze(0),
            'attention_mask': source_encoded['attention_mask'].squeeze(0),
            'labels': target_encoded['input_ids'].squeeze(0)
        }

class TranslationMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute_metrics(self, pred):
        try:
            predictions = pred.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            labels = pred.label_ids
            
            predicted_texts = self.tokenizer.batch_decode(
                predictions, 
                skip_special_tokens=True
            )
            
            label_texts = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )
            
            bleu = sacrebleu.corpus_bleu(
                predicted_texts, 
                [[text] for text in label_texts]
            )
            
            return {
                "bleu": bleu.score,
                "num_samples": len(predicted_texts)
            }
        except Exception as e:
            logger.error(f"Metrics computation error: {e}")
            return {"bleu": 0.0, "error": str(e)}

class PDFVisualExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.visual_elements = []

    def extract_images(self, page_num: int) -> List[VisualElement]:
        page = self.doc[page_num]
        images = []
        
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                
                image_data = base_image["image"]
                
                try:
                    pil_image = Image.open(io.BytesIO(image_data))
                    if pil_image.mode in ['CMYK', 'P']:
                        pil_image = pil_image.convert('RGB')
                    
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image_data = img_byte_arr.getvalue()
                    
                except Exception as e:
                    logger.warning(f"Image conversion failed: {e}, using original")
                
                image_rect = page.get_image_bbox(img)
                
                images.append(VisualElement(
                    type='image',
                    bbox=image_rect,
                    data=image_data,
                    page_num=page_num,
                    properties={
                        'format': 'png',
                        'colorspace': 'rgb'
                    }
                ))
            except Exception as e:
                logger.error(f"Error extracting image: {e}")
                continue
                
        return images

    def extract_tables(self, page_num: int) -> List[VisualElement]:
        page = self.doc[page_num]
        tables = []
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 1:
                tables.append(VisualElement(
                    type='table',
                    bbox=tuple(block["bbox"]),
                    page_num=page_num,
                    data=block,
                    properties={'cells': block.get("lines", [])}
                ))
        return tables

    def extract_lines(self, page_num: int) -> List[VisualElement]:
        page = self.doc[page_num]
        paths = page.get_drawings()
        lines = []
        
        for path in paths:
            if path["type"] == "l":
                lines.append(VisualElement(
                    type='line',
                    bbox=tuple(path["rect"]),
                    page_num=page_num,
                    data=path,
                    properties={
                        'color': path.get("color", (0, 0, 0)),
                        'width': path.get("width", 1),
                        'style': path.get("style", "solid")
                    }
                ))
        return lines

    def extract_all_elements(self):
        for page_num in range(len(self.doc)):
            self.visual_elements.extend(self.extract_images(page_num))
            self.visual_elements.extend(self.extract_tables(page_num))
            self.visual_elements.extend(self.extract_lines(page_num))
        
        return self.visual_elements

    def save_elements_data(self, output_path: str):
        try:
            base_dir = os.path.dirname(output_path)
            os.makedirs(base_dir, exist_ok=True)
            
            images_dir = f"{output_path}_images"
            os.makedirs(images_dir, exist_ok=True)
            
            elements_data = []
            for elem in self.visual_elements:
                elem_dict = {
                    'type': elem.type,
                    'bbox': tuple(elem.bbox) if hasattr(elem.bbox, '__iter__') else elem.bbox,
                    'page_num': elem.page_num,
                    'properties': {
                        k: tuple(v) if isinstance(v, (fitz.Rect, tuple, list)) else v
                        for k, v in elem.properties.items()
                    }
                }
                
                if elem.type == 'image':
                    img_filename = f"{len(elements_data)}.{elem_dict['properties'].get('format', 'png')}"
                    img_path = os.path.join(images_dir, img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    
                    with open(img_path, 'wb') as f:
                        f.write(elem.data)
                    elem_dict['data'] = img_path
                else:
                    elem_dict['data'] = str(elem.data)
                
                elements_data.append(elem_dict)
            
            with open(f"{output_path}.json", 'w', encoding='utf-8') as f:
                json.dump(elements_data, f, ensure_ascii=False, indent=4, default=str)
                
            logger.info(f"Saved visual elements to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving visual elements: {str(e)}")
            raise

class PDFReconstructor:
    def __init__(self, elements_data: List[VisualElement], translated_text: Dict):
        self.elements = elements_data
        self.translated_text = translated_text

    def create_pdf(self, output_path: str):
        c = canvas.Canvas(output_path, pagesize=letter)
        
        current_page = 0
        page_elements = [e for e in self.elements if e.page_num == current_page]
        
        for elem in page_elements:
            if elem.type == 'image':
                self._add_image(c, elem)
            elif elem.type == 'line':
                self._add_line(c, elem)
            elif elem.type == 'table':
                self._add_table(c, elem)
        
        if current_page in self.translated_text:
            for text_block in self.translated_text[current_page]:
                c.drawString(text_block['x'], text_block['y'], text_block['text'])
        
        c.save()

    def _add_image(self, canvas, element):
        if isinstance(element.data, bytes):
            img = Image.open(io.BytesIO(element.data))
            canvas.drawImage(img, *element.bbox)

    def _add_line(self, canvas, element):
        canvas.setStrokeColor(element.properties['color'])
        canvas.setLineWidth(element.properties['width'])
        canvas.line(*element.bbox)

    def _add_table(self, canvas, element):
        pass

from sacrebleu.metrics import BLEU

class PDFTranslationModel:
    def __init__(self, config: TranslationConfig):
        self.config = config
        
        try:
            self.model = MarianMTModel.from_pretrained(config.model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(config.model_name)
            
            # [CPU VERSION CHANGE] Убираем перенос модели на GPU.
            # if torch.cuda.is_available():
            #     self.model = self.model.cuda()
            #     logger.info("Model moved to GPU")
            
            for param in self.model.parameters():
                param.requires_grad = True
            
            # [CPU VERSION CHANGE] Не используем GPU, поэтому fp16 не нужен.
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters: {trainable_params}")
            
            self.bleu = BLEU()
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str):
        if self.config.preserve_visual_elements:
            self.visual_extractor = PDFVisualExtractor(pdf_path)
            elements = self.visual_extractor.extract_all_elements()
            self.visual_extractor.save_elements_data(
                os.path.join(self.config.visual_elements_dir, 
                             os.path.basename(pdf_path))
            )
    
    def compute_metrics(self, eval_pred):
        try:
            predictions, labels = eval_pred

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            # Превращаем каждую строку в список (для sacrebleu), если нужно
            # но здесь используем BLEU из sacrebleu.metrics
            bleu_score = self.bleu.corpus_score(decoded_preds, [decoded_labels])

            return {
                "bleu": bleu_score.score,
                "num_samples": len(decoded_preds)
            }

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0, "error": str(e)}

    def train(self, train_dataset, eval_dataset, test_dataset, save_dir="./final_model"):
        try:
            callback = ModelCallback(save_dir)
            
            # [CPU VERSION CHANGE]
            # 1) no_cuda=True, чтобы использовать только CPU
            # 2) fp16=False, чтобы отключить mixed precision
            training_args = TrainingArguments(
                output_dir=save_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                eval_accumulation_steps=2,
                learning_rate=self.config.learning_rate,
                # [CPU VERSION CHANGE] Убираем fp16 (mixed precision)
                fp16=False,
                # [CPU VERSION CHANGE] Явно указываем на отсутствие CUDA
                no_cuda=True,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                logging_dir='./logs',
                logging_steps=2,
                eval_steps=2,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=100,
                logging_first_step=True,
                load_best_model_at_end=True,
                metric_for_best_model="bleu"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[callback],
                compute_metrics=self.compute_metrics,
                # [CPU VERSION CHANGE] Здесь не нужно указывать GPU девайс
                # data_collator и т.д. можно использовать по умолчанию
            )

            if len(eval_dataset) == 0:
                logger.warning("⚠️ eval_dataset is empty! Evaluation will not be performed!")
            
            trainer.train()
            trainer.save_model(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            np.save(f"{save_dir}/training_history.npy", callback.history)
            
            test_results = trainer.evaluate(test_dataset)
            logger.info(f"Test results: {test_results}")

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

class ModelCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'bleu_score': [],
            'learning_rate': [],
            'epoch': []
        }
        self.start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Trainer logs: {logs}")
            
            if 'loss' in logs:
                self.history['train_loss'].append(logs['loss'])
                logger.info(f"Train loss: {logs['loss']:.4f}")
                    
            if 'eval_loss' in logs:
                self.history['eval_loss'].append(logs['eval_loss'])
                logger.info(f"Eval loss: {logs['eval_loss']:.4f}")
                    
            if 'eval_bleu' in logs:
                self.history['bleu_score'].append(logs['eval_bleu'])
                logger.info(f"BLEU score: {logs['eval_bleu']:.4f}")
                    
            if 'learning_rate' in logs:
                self.history['learning_rate'].append(logs['learning_rate'])
                logger.info(f"Learning Rate: {logs['learning_rate']:.6f}")

            if 'epoch' in logs:
                self.history['epoch'].append(logs['epoch'])
                logger.info(f"Epoch: {logs['epoch']:.2f}")
            
            logger.info(f"Current history state: {self.history}")

            if any(len(v) > 0 for v in self.history.values()):
                np.save(f"{self.output_dir}/training_history.npy", self.history)
                self.plot_progress()

    def plot_progress(self):
        if not any(self.history.values()):
            logger.warning("No training history to plot.")
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        if self.history['train_loss']:
            plt.plot(self.history['train_loss'], label='Train Loss', color='blue')
        
        if self.history['eval_loss']:
            plt.plot(self.history['eval_loss'], label='Eval Loss', color='red')

        if self.history['bleu_score']:
            plt.plot(self.history['bleu_score'], label='BLEU Score', color='green')

        plt.title('Training and Evaluation Loss / BLEU')
        plt.xlabel('Steps')
        plt.ylabel('Loss / BLEU')
        plt.legend()

        plt.subplot(2, 2, 2)
        if self.history['bleu_score']:
            plt.plot(self.history['bleu_score'], label='BLEU Score', color='green')
            plt.title('BLEU Score Over Time')
            plt.xlabel('Eval Steps')
            plt.ylabel('BLEU')
            plt.legend()

        plt.subplot(2, 2, 3)
        if self.history['learning_rate']:
            plt.plot(self.history['learning_rate'], label='Learning Rate', color='orange')
            plt.title('Learning Rate Progression')
            plt.xlabel('Steps')
            plt.ylabel('LR')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_progress.png")
        plt.close()

def main():
    # [CPU VERSION CHANGE] Убираем блоки с torch.cuda.empty_cache() и логику GPU.
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    try:
        config = TranslationConfig()
        
        os.makedirs(config.visual_elements_dir, exist_ok=True)
        
        dataset = ParallelPDFDataset("source_pdfs", "target_pdfs", config)
        
        logger.info("Extracting visual elements from PDFs...")
        for pdf_file in os.listdir("source_pdfs"):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join("source_pdfs", pdf_file)
                extractor = PDFVisualExtractor(pdf_path)
                elements = extractor.extract_all_elements()
                
                output_path = os.path.join(config.visual_elements_dir, f"{os.path.splitext(pdf_file)[0]}_elements")
                extractor.save_elements_data(output_path)
                logger.info(f"Saved visual elements for {pdf_file}")
        
        train_data, eval_data, test_data = dataset.load_data()
        
        with open("parallel_texts.json", "w", encoding="utf-8") as f:
            json.dump({
                "train": train_data,
                "eval": eval_data,
                "test": test_data
            }, f, ensure_ascii=False, indent=4)
        
        translation_model = PDFTranslationModel(config=config)
        train_dataset = CustomDataset(train_data, translation_model.tokenizer, config.max_length)
        eval_dataset = CustomDataset(eval_data, translation_model.tokenizer, config.max_length)
        test_dataset = CustomDataset(test_data, translation_model.tokenizer, config.max_length)
        
        translation_model.train(train_dataset, eval_dataset, test_dataset)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # [CPU VERSION CHANGE] Убираем логику очистки кэша
        pass

if __name__ == "__main__":
    main()
