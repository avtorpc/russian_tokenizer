"""Скачать и проверить фиксированную версию русско-английского OPUS Books."""
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "Helsinki-NLP/opus_books"
REVISION = "1f9f6191d0e91a3c539c2595e2fe48fc1420de9b"
FILENAME = "en-ru/train-00000-of-00001.parquet"
SHA256 = "df1756a630ad7e70c85146eb051792ab999c09fae2ff7fb28718c210499238a3"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "opus_books"


def validate(path):
    """Хеш проверяет версию и целостность; Parquet — читаемость и схему."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("SHA-256 не совпадает: файл повреждён, имеет другую версию или является LFS-указателем")
    parquet = pq.ParquetFile(path)
    translation = parquet.schema_arrow.field("translation").type
    if not pa.types.is_struct(translation):
        raise ValueError("Поле translation должно быть структурой")
    if not pa.types.is_string(translation.field("ru").type):
        raise ValueError("Поле translation.ru должно содержать строки")
    rows = 0
    for batch in parquet.iter_batches(columns=["translation"]):
        rows += batch.num_rows
    if rows == 0:
        raise ValueError("Корпус пуст")
    return rows


def main():
    target = DATA_DIR / FILENAME
    if target.exists():
        try:
            rows = validate(target)
        except (ValueError, OSError, KeyError) as error:
            print(f"Локальный файл не прошёл проверку: {error}. Скачиваем заново.")
        else:
            print(f"Корпус уже готов, скачивание пропущено. Записей: {rows:,}. Файл: {target}")
            return
    print(f"Скачиваем {REPO_ID}, ревизия {REVISION}, пара en-ru...")
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        filename=FILENAME,
        local_dir=DATA_DIR,
        force_download=True,
    )
    rows = validate(target)
    print(f"Корпус готов. Записей: {rows:,}. Файл: {target}")


if __name__ == "__main__":
    main()
