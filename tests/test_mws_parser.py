from app.services.mws_parser import MWSParser


def test_parse_models_page_extracts_models_without_hardcoded_name_list() -> None:
    parser = MWSParser()

    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>Model</th>
            <th>Input</th>
            <th>Output</th>
            <th>Context</th>
            <th>Size</th>
          </tr>
        </table>
        <table>
          <tr>
            <td>new-model-alpha-12b</td>
            <td>Text</td>
            <td>Text</td>
            <td>128</td>
            <td>12</td>
          </tr>
          <tr>
            <td>vision-model-pro</td>
            <td>Text, Image</td>
            <td>Text</td>
            <td>128</td>
            <td>27</td>
          </tr>
          <tr>
            <td>embed-fast-v1</td>
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

    assert "new-model-alpha-12b" in names
    assert "vision-model-pro" in names
    assert "embed-fast-v1" in names

    vision = next(model for model in models if model.name == "vision-model-pro")
    assert vision.supports_image_input is True
    assert vision.context_window_tokens == 128000

    embedding = next(model for model in models if model.name == "embed-fast-v1")
    assert embedding.is_embedding_model is True
    assert embedding.output_modalities == ["embedding"]


def test_parse_pricing_page_extracts_prices_without_hardcoded_name_list() -> None:
    parser = MWSParser()

    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>Model</th>
            <th>Cached input</th>
            <th>Cached output</th>
            <th>Input</th>
            <th>Output</th>
            <th>Billing</th>
          </tr>
          <tr>
            <td>new-model-alpha-12b</td>
            <td>0,054 ₽</td>
            <td>0,219 ₽</td>
            <td>1,098 ₽</td>
            <td>1,098 ₽</td>
            <td>100</td>
          </tr>
          <tr>
            <td>embed-fast-v1</td>
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

    regular = next(item for item in prices if item.model_name == "new-model-alpha-12b")
    assert regular.input_price_per_1k_tokens_rub == 1.098
    assert regular.output_price_per_1k_tokens_rub == 1.098
    assert regular.billing_unit_tokens == 100

    embedding = next(item for item in prices if item.model_name == "embed-fast-v1")
    assert embedding.input_price_per_1k_tokens_rub == 0.0122
    assert embedding.output_price_per_1k_tokens_rub is None
    assert embedding.billing_unit_tokens == 1000
