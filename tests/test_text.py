from unittest.mock import MagicMock

from src.features.text import chunk_text, prepare_finetune_text


def test_chunk_text_with_mock_tokenizer():
    mock_tok = MagicMock()
    # Giả lập text dài 400 tokens
    mock_tok.encode.return_value = list(range(400))
    mock_tok.decode.side_effect = lambda tokens, **kw: f"chunk_{len(tokens)}"

    chunks = chunk_text("dummy text", mock_tok, chunk_size=200, overlap=50)
    # Stride = 200 - 50 = 150
    # i=0 (0..200), i=150 (150..350), i=300 (300..400) -> 3 chunks
    assert len(chunks) == 3
    assert mock_tok.encode.called



def test_prepare_finetune_text():
    raw_cv = "John Doe\nAddress: 123 Street\nPhone: 123456\nSummary: 5 years experience in Python."
    cleaned = prepare_finetune_text(raw_cv)
    assert cleaned.startswith("Summary:")
    assert "John Doe" not in cleaned

