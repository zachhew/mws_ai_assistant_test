from app.services.mws_parser import MWSParser


def test_parse_models_page_extracts_expected_models() -> None:
    parser = MWSParser()

    html = """
    <html>
      <body>
        <table>
          <tr>
            <td>deepseek-r1-distill-qwen-32b</td>
            <td>Text</td>
            <td>Text</td>
            <td>128</td>
            <td>32</td>
          </tr>
          <tr>
            <td>gemma-3-27b-it</td>
            <td>Text, Image</td>
            <td>Text</td>
            <td>128</td>
            <td>27</td>
          </tr>
          <tr>
            <td>bge-m3</td>
            <td>Text</td>
            <td>Embedding</td>
            <td>8</td>
            <td>0.6</td>
          </tr>
        </table>
      </body>
    </html>
    """

    models = parser.parse_models_page(html, "https://example.com/models")

    names = {model.name for model in models}

    assert "deepseek-r1-distill-qwen-32b" in names
    assert "gemma-3-27b-it" in names
    assert "bge-m3" in names

    gemma = next(model for model in models if model.name == "gemma-3-27b-it")
    assert gemma.supports_image_input is True
    assert gemma.context_window_tokens == 128000

    bge = next(model for model in models if model.name == "bge-m3")
    assert bge.is_embedding_model is True
    assert bge.output_modalities == ["embedding"]


def test_parse_pricing_page_extracts_regular_prices() -> None:
    parser = MWSParser()

    html = """
    <html>
      <body>
        <table>
          <tr>
            <td>deepseek-r1-distill-qwen-32b</td>
            <td>0,054 ₽</td>
            <td>0,219 ₽</td>
            <td>1,098 ₽</td>
            <td>1,098 ₽</td>
            <td>100</td>
          </tr>
          <tr>
            <td>bge-m3</td>
            <td>0,0006 ₽</td>
            <td>–</td>
            <td>0,0122 ₽</td>
            <td>–</td>
            <td>1000</td>
          </tr>
        </table>
      </body>
    </html>
    """

    prices = parser.parse_pricing_page(html, "https://example.com/pricing")

    deepseek = next(item for item in prices if item.model_name == "deepseek-r1-distill-qwen-32b")
    assert deepseek.input_price_per_1k_tokens_rub == 1.098
    assert deepseek.output_price_per_1k_tokens_rub == 1.098
    assert deepseek.billing_unit_tokens == 100

    bge = next(item for item in prices if item.model_name == "bge-m3")
    assert bge.input_price_per_1k_tokens_rub == 0.0122
    assert bge.output_price_per_1k_tokens_rub is None
    assert bge.billing_unit_tokens == 1000