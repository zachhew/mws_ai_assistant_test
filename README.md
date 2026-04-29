# AI Model Selection Assistant

Ассистент для подбора моделей MWS GPT Model Hub под пользовательский сценарий и бюджет.

Сервис предоставляет OpenAI-совместимый endpoint `/v1/chat/completions`, динамически загружает данные о моделях и тарифах с сайта MWS, оценивает стоимость использования и формирует структурированный ответ с рекомендациями.

## Что делает сервис

Ассистент принимает пользовательский сценарий в свободной форме, например:
- для customer support чата;
- для multimodal-сценария с текстом и изображениями;
- для embeddings и поиска по документам.

После этого сервис:
- извлекает параметры сценария;
- загружает актуальные модели и тарифы MWS;
- рассчитывает ориентировочную месячную стоимость;
- подбирает подходящие варианты;
- возвращает структурированный ответ;
- поддерживает follow-up уточнения в рамках одной сессии через `session_id`.

## Архитектура

Решение построено как гибрид из агентного orchestration-слоя и детерминированного расчетного ядра.

### Agent layer
Используется **Google ADK** как orchestration framework.

ADK-слой отвечает за:
- обработку пользовательского запроса;
- вызов workflow для запуска основного пайплайна;
- поддержку session-aware сценария;
- генерацию финального ответа через LLM backend;
- поддержку `stream=true`.

### LLM backend
В качестве runtime LLM backend используется **OpenRouter**.

LLM используется для:
- извлечения структурированного профиля пользовательского сценария;
- ранжирования уже подготовленных технически подходящих вариантов;
- генерации финального ответа на русском языке;
- follow-up поведения поверх сохраненного session state.

### Deterministic core
Критичная фактологическая и расчетная логика не делегируется LLM.

Детерминированно реализованы:
- загрузка данных с MWS;
- парсинг характеристик моделей и тарифов;
- расчет стоимости;
- фильтрация моделей по hard constraints;
- построение structured ranking context для ranking-agent;
- построение структурированного отчета.

Такой подход позволяет совместить:
- гибкость LLM;
- воспроизводимость расчетов;
- прозрачность выбора моделей;
- устойчивость к ошибкам генерации.

## Выбранный агентный паттерн

Использован **tool-based coordinator pattern** с вложенным multi-agent workflow.

Поток обработки выглядит так:

1. Клиент отправляет запрос в `/v1/chat/completions`
2. Извлекается `session_id`
3. Запрос передается в ADK runtime
4. ADK-агент вызывает tool `run_model_selection`
5. Tool запускает workflow из нескольких агентов:
   - `ProfileAgent` извлекает `UserCaseProfile`
   - deterministic preparation step загружает каталог, фильтрует модели и считает стоимость
   - `RankingAgent` выбирает top candidates без rule-based weight scoring
   - deterministic finalization step собирает `RecommendationReport`
6. Верхнеуровневый агент формирует финальный текстовый ответ
7. API возвращает OpenAI-compatible response

## Почему цены и расчеты не делаются через LLM

Прайсы и характеристики моделей не хардкодятся и не извлекаются через prompt engineering.

Вместо этого:
- данные по моделям и стоимости динамически загружаются с сайта MWS;
- стоимость рассчитывается детерминированно;
- LLM не используется как источник фактов для тарифов.

Это сделано для того, чтобы сохранить:
- воспроизводимость;
- тестируемость;
- устойчивость к hallucination risk.

## Поддержка диалога в рамках сессии

Сервис поддерживает уточнение вводных по `session_id`.

Это позволяет:
- менять только бюджет;
- менять только количество запросов;
- уточнять modality;
- задавать follow-up вопросы без повторного ввода всего сценария.

Например:
- сначала пользователь задает базовый сценарий;
- затем отправляет: `А если бюджет увеличить до 60000 рублей?`
- система сохраняет остальные параметры сценария и пересчитывает только изменившуюся часть.

## API

### Endpoint

`POST /v1/chat/completions`

Поддерживаются:
- обычный JSON-ответ;
- `stream=true` в SSE-формате.

## Пример запроса

```json
{
  "model": "mws-model-selector",
  "messages": [
    {
      "role": "user",
      "content": "Мне нужна модель для customer support чата, около 50000 запросов в месяц, вход 700 токенов, выход 250 токенов, бюджет 12000 рублей, нужен хороший баланс цены и качества."
    }
  ],
  "metadata": {
    "session_id": "demo-session-1"
  }
}
```
## Пример ответа
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1776860000,
  "model": "mws-model-selector",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "1. Входные данные: ... 2. Подходящие или ближайшие технические варианты: ... 3. Расчеты: ... 4. Пояснения/ограничения: ... 5. Итог: ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

## Stream mode
Если передать `"stream": true`, endpoint возвращает OpenAI-подобный SSE stream.
```json
{
  "model": "mws-model-selector",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "Нужна модель для построения эмбеддингов для поиска по документам. Около 200000 запросов в месяц, примерно 350 токенов на вход, бюджет 10000 рублей."
    }
  ]
}
```

## Основные компоненты проекта
* `app/api` — API слой и response mapping
* `app/agent` — ADK runtime, coordinator, session flow
* `app/services` — MWS ingestion, parser, estimator, profile extraction, recommender
* `app/domain` — доменные сущности
* `app/core` — конфигурация, логирование, ошибки
* `tests` — unit tests для parser, estimator и recommender

## Локальный запуск
### 1. Создать виртуальное окружение
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
### 2. Установить зависимости
```bash
pip install -e ".[dev]"
```

### 3. Создать .env
Пример переменных окружения:
```
APP_NAME=mws-ai-assistant-test
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=dev
LOG_LEVEL=INFO

MWS_MODELS_URL=https://mws.ru/docs/cloud-platform/gpt/general/gpt-models.html
MWS_PRICING_URL=https://mws.ru/docs/cloud-platform/gpt/general/pricing.html
CATALOG_CACHE_TTL_SECONDS=900

DEFAULT_SESSION_TTL_MINUTES=60

ADK_LITELLM_MODEL=openrouter/openai/gpt-4o-mini
OR_API_KEY=your_openrouter_api_key
OR_SITE_URL=http://localhost:8000
OR_APP_NAME=mws-ai-assistant-test

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_HTTP_REFERER=http://localhost:8000
OPENROUTER_APP_TITLE=mws-ai-assistant-test
```

### 4. Запустить сервис
```bash
uvicorn app.main:app --reload
```

### 5. Проверить healthcheck
```bash
curl http://127.0.0.1:8000/health
```

---

## Тесты
Запуск тестов:
```bash
pytest
```
Покрыты:
* parser;
* estimator;
* recommender;
* profile finalization.

## Что дополнительно реализовано

Помимо базового требования, в проекте есть:

* `stream=true` режим;
* unit tests;
* логирование основных этапов;
* валидация входных данных через `Pydantic`;
* поддержка follow-up сценариев в рамках одной сессии.
