from src.features.skills import is_exact_match


def test_exact_match_symbols_and_synonyms():
    assert is_exact_match("C++", "Senior C++ developer")
    assert is_exact_match("C#", "5 years of C# and .NET")
    assert is_exact_match("AWS", "Built pipelines on AWS EC2")        # dạng gốc vẫn khớp dù có từ đồng nghĩa
    assert is_exact_match("node", "Backend in Node and Express")
    assert is_exact_match("ms excel", "Advanced Excel, pivot tables")  # dạng chuẩn hoá khớp


def test_exact_match_respects_word_boundaries():
    assert not is_exact_match("java", "JavaScript developer")
    assert not is_exact_match("golang", "Ready to go live with the team")  # "go" thường không phải Go
    assert not is_exact_match("sql", "nosql databases")
