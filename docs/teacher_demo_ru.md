# Демонстрация первого вертикального среза BikeFlow

## Что решает проект

BikeFlow прогнозирует суммарное число аренд велосипедов в Сеуле на конкретный час
при заданной погоде. Такой прогноз помогает заранее оценить нагрузку. Погода сейчас
не загружается автоматически: пользователь API передаёт ожидаемые значения сам,
поэтому ответ читается как «спрос при этих погодных условиях».

MLOps здесь нужен, чтобы одинаково и проверяемо пройти путь от исходных данных до
работающего сервиса: зафиксировать источник и split, воспроизвести обучение,
упаковать preprocessing вместе с моделью, проверить контракт тестами и запустить
тот же прогноз в Docker.

## Роли и provisional-решения

- Участник A, `EgorMa1tsev`: данные, признаки, обучение, сравнение моделей и
  ML-артефакт.
- Участница B, `daya-alexandra`: GitHub-репозиторий, API, интеграция модели, тесты,
  CI и Docker.

Для первого среза HGB временно зафиксирована как serving-модель: у неё лучший MAE
на validation. MAE — основная метрика выбора, WAPE — дополнительная. MLP сохранена
как отдельный эксперимент. Решения нужно обсудить с A.

## Данные и разбиение

Источник — [Seoul Bike Sharing Demand, UCI](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand),
8760 почасовых строк за один год. Загрузчик принимает только CSV с закреплённым
SHA256. Датасет и модели не коммитятся.

| Split | Период | Все часы | Рабочие часы |
| --- | --- | ---: | ---: |
| train | 2017-12-01…2018-07-31 | 5832 | 5784 |
| validation | 2018-08-01…2018-09-30 | 1464 | 1368 |
| test | 2018-10-01…2018-11-30 | 1464 | 1313 |

Хронологическое разбиение имитирует реальность: модель учится на прошлом и
проверяется на будущем. Случайное перемешивание позволило бы информации из будущих
месяцев попасть в train. Test не выбирает модель; он открывается только для
финальной оценки уже выбранной HGB.

Фактические метрики текущего запуска:

| Split | MAE | WAPE |
| --- | ---: | ---: |
| validation | 161.085 | 0.166250 |
| test | 277.908 | 0.326851 |

## Путь данных до прогноза

1. FastAPI проверяет JSON по Pydantic-схеме.
2. `prediction_time` без timezone получает `422`; корректное время переводится в
   `Asia/Seoul`.
3. Из времени выводятся час, день недели и сезон.
4. Имена API переводятся в canonical-имена. Типы, единицы и диапазоны API и ML
   берут из одного `FEATURE_CONTRACT`.
5. `InferencePipeline` из `.joblib` повторно валидирует строку, строит признаки и
   вызывает настоящий `HistGradientBoostingRegressor`.
6. API возвращает неотрицательный прогноз и версию артефакта.

Metadata содержит Git SHA, seed, параметры HGB, хеш исходных данных, хеш
конфигурации и версии ключевых библиотек. Фиксация версий и seed повышает
воспроизводимость, но bit-for-bit совпадение между любыми ОС не обещается.

## Команды с чистого проекта

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install --constraint requirements/runtime-py311.lock -e ".[dev,ml]"
python -m bikeflow.ml download
python -m bikeflow.ml preprocess
python -m bikeflow.ml split
python -m bikeflow.ml train --no-figures
```

Артефакт: `models/model.joblib`. Точная команда его создания — последняя команда
выше. Для локального API:

```bash
BIKEFLOW_MODEL_PATH=models/model.joblib uvicorn bikeflow.api.main:app --host 127.0.0.1 --port 8000
```

Для Docker:

```bash
docker compose up --build
```

Compose монтирует `./models/model.joblib` как `/models/model.joblib:ro`. Serving-
образ использует Python 3.11 и не содержит PyTorch.

## Настоящий запрос и ответ

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "prediction_time":"2026-07-15T08:00:00+09:00",
  "temperature_c":24.5,
  "humidity_pct":61,
  "wind_speed_m_s":1.8,
  "visibility_10m":1800,
  "dew_point_c":16.4,
  "solar_radiation_mj_m2":1.2,
  "rainfall_mm":0,
  "snowfall_cm":0,
  "holiday":false,
  "functioning_day":true
}'
```

```json
{
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "predicted_rentals": 2336.968455081055,
  "model_version": "hgb-373339b7-c3867ac1-98fd0d01"
}
```

## Тесты и CI

Локальная проверка:

```bash
ruff check .
ruff format --check .
pytest
docker build --tag bikeflow:local .
```

Тесты включают малое обучение HGB, `train → save → load → predict`, настоящий
adapter, timezone, общие границы API/ML и ответы `422`. CI запускает `lint`,
`tests`, `ml-tests`, `docker-build`, `docker-runtime-smoke`. Runtime smoke обучает
небольшую настоящую HGB, монтирует bundle read-only и обращается к `/predict` по
HTTP. Stub остаётся только injected test double.

## Сценарий показа на 3–5 минут

1. Открыть новый integration PR и показать, что он основан на двух неизменённых
   коммитах PR №6, а интеграционные исправления идут отдельными коммитами.
2. Открыть `README.md`, затем `reports/model_selection.md`: объяснить временной
   split, MAE/WAPE и то, что test не выбирал модель.
3. Открыть `src/bikeflow/ml/features.py`, `src/bikeflow/ml/pipeline.py` и
   `src/bikeflow/api/dependencies.py`: показать общий контракт, единый bundle и
   настоящий predictor вместо stub.
4. Выполнить `pytest`, затем `docker compose up --build`.
5. Открыть <http://127.0.0.1:8000/docs>, отправить пример запроса и указать реальный
   прогноз и `model_version` в ответе.
6. Открыть вкладку Checks PR: показать lint, тесты, build и runtime smoke.

## Ограничения и следующие этапы

Сейчас реализован только первый вертикальный срез. Ещё нет DVC, MLflow, Airflow,
PostgreSQL, мониторинга, drift detection, автоматического переобучения,
автоматического weather API, Kubernetes, Argo CD и полноценного UI. Эти компоненты
могут появиться позже, но документация не выдаёт их за готовые.

Главные текущие риски: один год данных, только один город, test целиком осенний,
ошибки на пиках/дожде, зависимость прогноза от качества введённой пользователем
погоды. В репозитории пока нет LICENSE — его нельзя выбирать без согласования обоих
авторов.
