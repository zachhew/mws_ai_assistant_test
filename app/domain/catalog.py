from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)

    context_window_tokens: int | None = Field(default=None, ge=0)
    model_size_label: str | None = None
    family: str | None = None

    supports_text_input: bool = False
    supports_image_input: bool = False
    supports_text_output: bool = False
    is_embedding_model: bool = False

    source_url: str


class PricingSpec(BaseModel):
    model_name: str
    input_price_per_1k_tokens_rub: float | None = Field(default=None, ge=0)
    output_price_per_1k_tokens_rub: float | None = Field(default=None, ge=0)
    billing_unit_tokens: int | None = Field(default=None, ge=0)

    currency: str = "RUB"
    source_url: str


class CatalogEntry(BaseModel):
    model: ModelSpec
    pricing: PricingSpec | None = None
