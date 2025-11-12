import os

base_path = r"ImageClassification/assets/inputs/images"
output_path = os.path.join(base_path, "tags.tsv")

with open(output_path, "w", encoding="utf-8") as f:
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    f.write(f"{folder}/{img}\t{folder}\n")

print(f"✅ tags.tsv berhasil dibuat di: {output_path}")
