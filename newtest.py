import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from transformers import MarianTokenizer, MarianMTModel
import torch
import io
from reportlab.lib.utils import ImageReader
from PIL import Image
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualElement:
    type: str  # 'image', 'table', 'line', 'shape', 'fill'
    bbox: Tuple[float, float, float, float]
    data: Any
    page_num: int
    properties: Dict

class CustomPDFTranslator:
    def __init__(self, model_path: str, input_pdf_path: str, output_pdf_path: str):
        self.input_pdf_path = input_pdf_path
        self.output_pdf_path = self._get_safe_output_path(output_pdf_path)
        self.visual_elements = []

        logger.info("Загрузка пользовательской модели перевода (CPU)...")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
            self.model = MarianMTModel.from_pretrained(model_path)
            self.model.eval()
            logger.info("Модель перевода будет работать на CPU (no GPU usage).")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

        # Инициализация шрифтов
        self._initialize_fonts()

    def _get_safe_output_path(self, desired_path: str) -> str:
        """Получение безопасного пути для сохранения PDF"""
        try:
            directory = os.path.dirname(desired_path) or '.'
            filename = os.path.basename(desired_path)
            os.makedirs(directory, exist_ok=True)
            
            output_path = desired_path
            counter = 0
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                counter += 1
                output_path = os.path.join(directory, f"{name}_{counter}{ext}")
            
            return output_path
            
        except (PermissionError, OSError):
            temp_dir = os.path.join(os.path.expanduser("~"), "Documents")
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(temp_dir, f"translated_pdf_{timestamp}.pdf")

    def _initialize_fonts(self):
        """Инициализация шрифтов с поддержкой кириллицы"""
        try:
            fonts_to_try = [
                ('Arial-Unicode', 'arial-unicode-ms.ttf'),
                ('DejaVuSerif', 'DejaVuSerif.ttf'),
                ('DejaVuSans', 'DejaVuSans.ttf'),
                ('FreeSans', 'FreeSans.ttf')
            ]
            
            self.available_fonts = []
            for font_name, font_file in fonts_to_try:
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_file))
                    self.available_fonts.append(font_name)
                    logger.info(f"Зарегистрирован шрифт: {font_name}")
                except:
                    # Если нет файла шрифта - просто переходим к следующему
                    pass
            
            if not self.available_fonts:
                raise Exception("Не найдено шрифтов с поддержкой кириллицы")
            
            self.default_font = self.available_fonts[0]
        except Exception as e:
            logger.error(f"Ошибка инициализации шрифтов: {str(e)}")
            raise

    def translate_text(self, text: str) -> str:
        if not text.strip():
            return text

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                # Генерация на CPU
                outputs = self.model.generate(**inputs.to('cpu'))
            
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text

        except Exception as e:
            logger.error(f"Ошибка перевода: {str(e)}")
            return text

    def extract_text_with_info(self) -> List[Dict]:
        """Извлечение текстовых блоков из PDF c их координатами и стилями."""
        text_blocks = []
        # Используем контекст-менеджер with, чтобы doc точно закрылся после выхода.
        with fitz.open(self.input_pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_blocks.append({
                                    'text': span['text'].strip(),
                                    'bbox': span['bbox'],
                                    'font_size': span['size'],
                                    'font_name': span['font'],
                                    'color': span['color'],
                                    'page_num': page_num,
                                    # Ширина исходного блока (чтобы ориентироваться при переносе строк)
                                    'origin_width': span['bbox'][2] - span['bbox'][0]
                                })
        # Здесь doc уже закрыт, но все нужные данные (text_blocks) сохранены в память.
        return text_blocks

    def get_text_width(self, text: str, font_name: str, font_size: float) -> float:
        """Примерная оценка ширины строки в пунктах."""
        try:
            face = pdfmetrics.getFont(font_name).face
            return face.stringWidth(text, font_size)
        except:
            return len(text) * font_size * 0.6

    def _wrap_text_to_width(self, text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = self.get_text_width(word, font_name, font_size)
            space_width = self.get_text_width(" ", font_name, font_size) if current_line else 0
            
            if current_width + word_width + space_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                # Добавляем предыдущую линию
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def extract_visual_elements(self):
        """Извлекает изображения, линии, заливки и другие графические объекты."""
        self.visual_elements = []  # Сбрасываем, если вызываем повторно
        
        with fitz.open(self.input_pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Изображения
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if "image" not in base_image:
                        continue
                    image_bytes = base_image["image"]
                    image_rect = page.get_image_bbox(img)

                    self.visual_elements.append(VisualElement(
                        type='image',
                        bbox=image_rect,
                        data=image_bytes,
                        page_num=page_num,
                        properties={'format': base_image.get("ext", "png")}
                    ))
                    logger.info(f"Добавлено изображение на страницу {page_num}")

                # Анализируем текстовые блоки (вдруг там «пустые» блоки = линии)
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    bbox = block.get("bbox", None)
                    lines = block.get("lines", [])
                    # Если в блоке нет текста, но есть bbox
                    if bbox and not lines:
                        x0, y0, x1, y1 = bbox
                        height = abs(y1 - y0)
                        # Иногда это может быть «линия» (очень тонкий прямоугольник)
                        if height < 3:
                            self.visual_elements.append(VisualElement(
                                type='line',
                                bbox=bbox,
                                data=None,
                                page_num=page_num,
                                properties={'color': (0, 0, 0), 'width': height}
                            ))
                            logger.info(f"Пустой блок => Линия, стр. {page_num}, bbox={bbox}")

                # Графические примитивы: линии, заливки
                for drawing in page.get_drawings():
                    d_type = drawing["type"]
                    rect = drawing["rect"]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

                    if d_type in ("l", "s"):  # Линия или штрих
                        color = drawing.get("color", (0, 0, 0))
                        width = drawing.get("width", 1)
                        self.visual_elements.append(VisualElement(
                            type='line',
                            bbox=(x0, y0, x1, y1),
                            data=None,
                            page_num=page_num,
                            properties={'color': color, 'width': width}
                        ))
                        logger.info(f"Линия (get_drawings), стр. {page_num}, bbox={(x0, y0, x1, y1)}")

                    elif d_type == "f":  # Заливка
                        fill_color = drawing.get("fill", (0, 0, 0))
                        self.visual_elements.append(VisualElement(
                            type='fill',
                            bbox=(x0, y0, x1, y1),
                            data=None,
                            page_num=page_num,
                            properties={'fill_color': fill_color}
                        ))
                        logger.info(f"Заливка (get_drawings), стр. {page_num}, bbox={(x0, y0, x1, y1)}")

                # Аннотации (например, «линейные»)
                annots = page.annots()
                if annots:
                    for annot in annots:
                        # type[0] == 8 => Line Annotation
                        if annot.type[0] == 8:
                            bbox = annot.rect
                            color = annot.colors.get("stroke", (0, 0, 0))
                            width = annot.border[0]
                            self.visual_elements.append(VisualElement(
                                type='line',
                                bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                                data=None,
                                page_num=page_num,
                                properties={'color': color, 'width': width}
                            ))
                            logger.info(f"Аннотация - линия, стр. {page_num}, bbox={bbox}")
        # Здесь doc тоже закрыт, но все данные visual_elements хранятся в self.visual_elements.

    def create_translated_pdf(self, text_blocks: List[Dict]):
        """Создаёт новый PDF-файл с переведённым текстом и наложенными элементами."""
        try:
            c = canvas.Canvas(self.output_pdf_path, pagesize=A4)

            current_page = -1
            for block in text_blocks:
                page_num = block['page_num']
                if page_num != current_page:
                    # Если страница уже была, завершаем её
                    if current_page >= 0:
                        c.showPage()
                    
                    # Переходим на новую страницу
                    current_page = page_num
                    # Добавляем все визуальные объекты для этой страницы
                    for elem in self.visual_elements:
                        if elem.page_num == current_page:
                            self._add_visual_element(c, elem)

                # Если блок пустой, пропускаем
                if not block['text'].strip():
                    continue

                # Перевод текста
                translated_text = self.translate_text(block['text'])
                original_font_size = block['font_size']
                adjusted_font_size = original_font_size * 0.7
                c.setFont(self.default_font, adjusted_font_size)

                # Цвет шрифта
                color = block['color']
                if isinstance(color, int):
                    # RGB из int
                    color = (
                        ((color >> 16) & 255) / 255,
                        ((color >> 8) & 255) / 255,
                        (color & 255) / 255
                    )
                if not (isinstance(color, (tuple, list)) and len(color) == 3):
                    color = (0, 0, 0)  # На всякий случай
                c.setFillColorRGB(*color)

                # Перенос строк
                available_width = block['origin_width']
                wrapped_lines = self._wrap_text_to_width(
                    translated_text,
                    available_width,
                    self.default_font,
                    adjusted_font_size
                )
                
                x, y = block['bbox'][0], A4[1] - block['bbox'][1]
                line_height = adjusted_font_size * 1.2
                total_height = line_height * (len(wrapped_lines) - 1)
                y_offset = total_height / 2

                for i, line in enumerate(wrapped_lines):
                    y_pos = y - y_offset + (i * line_height)
                    c.drawString(x, y_pos, line)

            c.save()
            logger.info(f"Переведённый PDF сохранён: {self.output_pdf_path}")
        except Exception as e:
            logger.error(f"Ошибка создания PDF: {str(e)}")
            raise

    def _add_visual_element(self, c, elem: VisualElement):
        """Добавляет на текущую страницу визуальный элемент (изображение, линия, заливка)."""
        try:
            x0, y0, x1, y1 = elem.bbox
            if elem.type == 'image':
                image_bytes = elem.data
                if not isinstance(image_bytes, bytes):
                    logger.warning(f"Некорректные данные изображения на стр. {elem.page_num}")
                    return
                img = Image.open(io.BytesIO(image_bytes))
                width = x1 - x0
                height = y1 - y0
                y_pos = A4[1] - y1
                c.drawImage(
                    ImageReader(img),
                    x0,
                    y_pos,
                    width=width,
                    height=height,
                    mask='auto'
                )

            elif elem.type == 'line':
                y0_pdf = A4[1] - y0
                y1_pdf = A4[1] - y1
                color = elem.properties.get('color', (0, 0, 0))
                width = elem.properties.get('width', 1)

                if not (isinstance(color, (tuple, list)) and len(color) == 3):
                    color = (0, 0, 0)
                c.setStrokeColorRGB(*color)

                if width <= 0:
                    width = 1
                c.setLineWidth(width)

                c.line(x0, y0_pdf, x1, y1_pdf)

            elif elem.type == 'fill':
                fill_color = elem.properties.get('fill_color', (0, 0, 0))
                if not (isinstance(fill_color, (tuple, list)) and len(fill_color) == 3):
                    fill_color = (0, 0, 0)
                rect_x = x0
                rect_y = A4[1] - y1
                rect_width = x1 - x0
                rect_height = y1 - y0

                c.setFillColorRGB(*fill_color)
                c.rect(rect_x, rect_y, rect_width, rect_height, stroke=0, fill=1)

            elif elem.type == 'table':
                # Если нужно реализовать отрисовку таблицы
                pass

        except Exception as e:
            logger.error(f"Ошибка при добавлении визуального элемента (стр. {elem.page_num}): {str(e)}")

def main():
    try:
        model_path = "final_model"
        input_pdf = "pdf_original/test3.pdf"
        output_pdf = "pdf_translate/output_ru.pdf"
        
        translator = CustomPDFTranslator(model_path, input_pdf, output_pdf)
        
        logger.info("Извлечение текста из PDF...")
        text_blocks = translator.extract_text_with_info()
        
        logger.info("Извлечение визуальных элементов из PDF...")
        translator.extract_visual_elements()
        
        logger.info("Создание переведённого PDF...")
        translator.create_translated_pdf(text_blocks)
        
        logger.info("Перевод PDF успешно завершён!")
        
    except Exception as e:
        logger.error(f"Ошибка в процессе перевода: {str(e)}")

if __name__ == "__main__":
    main()
