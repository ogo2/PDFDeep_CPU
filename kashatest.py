import os
import shutil

# Create directories if they don't exist
os.makedirs('source_pdfs', exist_ok=True)
os.makedirs('target_pdfs', exist_ok=True)

# Get list of PDF files in 'kasha' directory
kasha_dir = 'kasha'
for filename in os.listdir(kasha_dir):
    if filename.endswith('.pdf'):
        source_path = os.path.join(kasha_dir, filename)
        
        # Process -EN.pdf files
        if filename.endswith('-EN.pdf'):
            new_filename = filename.replace('-EN.pdf', '.pdf')
            target_path = os.path.join('source_pdfs', new_filename)
            shutil.move(source_path, target_path)
            
        # Process -RU.pdf files
        elif filename.endswith('-RU.pdf'):
            new_filename = filename.replace('-RU.pdf', '_translated.pdf')
            target_path = os.path.join('target_pdfs', new_filename)
            shutil.move(source_path, target_path)

print("Files processed successfully!")