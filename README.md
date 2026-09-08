# BikeFlow

BikeFlow — учебный MLOps-проект для прогноза почасового спроса на городской
велопрокат в Сеуле. Первый вертикальный срез уже проходит весь путь: официальный
датасет UCI → проверка и хронологический split → обучение HGB → единый артефакт
preprocessing + model → FastAPI `/predict` → Docker.

Участник A ([EgorMa1tsev](https://github.com/EgorMa1tsev)) отвечает за данные,
признаки, обучение и оценку моделей. Участница B
([daya-alexandra](https://github.com/daya-alexandra)) отвечает за репозиторий,
API, интеграцию, тесты, CI и Docker. Решения текущего среза provisional: команда
может пересмотреть их после обсуждения.

## Текущий путь запроса

`JSON request` → единая схема признаков → перевод времени в `Asia/Seoul` →
календарные признаки → pipeline из `models/model.joblib` → HGB →
`predicted_rentals` + `model_version`.

Погодные значения передаёт вызывающая сторона. Это условный прогноз «каким будет
спрос при заданной погоде», а не автоматически полученная метеосводка.

## Установка и обучение

Требуется Python 3.11. Прямые версии закреплены в `pyproject.toml`, полный runtime
lock — в `requirements/runtime-py311.lock`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --constraint requirements/runtime-py311.lock -e ".[dev,ml]"
python -m bikeflow.ml download
python -m bikeflow.ml preprocess
python -m bikeflow.ml split
python -m bikeflow.ml train --no-figures
```

Загрузка принимает только исходный CSV с SHA256
`373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60`.
Данные и `.joblib`-артефакты игнорируются Git.

По умолчанию обучаются baseline и production-кандидат HGB. Дополнительный MLP-
эксперимент запускается отдельно после установки `.[mlp]`:

```bash
python -m pip install -e ".[mlp]"
python -m bikeflow.ml train --include-mlp --no-figures
```

Выбор делается по MAE на validation, WAPE — дополнительная метрика. Test не
участвует в выборе и вычисляется только после фиксации HGB.

| Split | MAE | WAPE |
| --- | ---: | ---: |
| validation | 161.085 | 0.166250 |
| test (final evaluation) | 277.908 | 0.326851 |

## API локально

Артефакт создаётся в `models/model.joblib`. Путь можно изменить переменной
`BIKEFLOW_MODEL_PATH`.

```bash
BIKEFLOW_MODEL_PATH=models/model.joblib uvicorn bikeflow.api.main:app --host 127.0.0.1 --port 8000
```

PowerShell:

```powershell
$env:BIKEFLOW_MODEL_PATH = "models/model.joblib"
uvicorn bikeflow.api.main:app --host 127.0.0.1 --port 8000
```

Пример настоящего запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "temperature_c": 24.5,
  "humidity_pct": 61,
  "wind_speed_m_s": 1.8,
  "visibility_10m": 1800,
  "dew_point_c": 16.4,
  "solar_radiation_mj_m2": 1.2,
  "rainfall_mm": 0,
  "snowfall_cm": 0,
  "holiday": false,
  "functioning_day": true
}'
```

Ответ текущего проверенного артефакта:

```json
{
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "predicted_rentals": 2336.968455081055,
  "model_version": "hgb-373339b7-c3867ac1-98fd0d01"
}
```

`prediction_time` без timezone и значения вне общего диапазона возвращают `422`.
Swagger UI доступен на <http://127.0.0.1:8000/docs>.

## Docker

Compose передаёт локально обученный артефакт в контейнер read-only:

```bash
docker compose up --build
```

Serving-образ работает на Python 3.11 и не устанавливает PyTorch. Перед запуском
нужно создать `models/model.joblib` командой обучения выше.

Для контрольного обучения именно внутри Linux/Python 3.11 доступен отдельный stage:

```bash
docker build --target training --tag bikeflow:training-py311 .
docker run --rm -e BIKEFLOW_GIT_SHA=$(git rev-parse HEAD) \
  -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" bikeflow:training-py311
```

## Проверки

```bash
ruff check .
ruff format --check .
pytest
docker build --tag bikeflow:local .
```

CI выполняет `lint`, `tests`, `ml-tests`, `docker-build` и
`docker-runtime-smoke`; последний обучает небольшой настоящий HGB, монтирует
артефакт в контейнер и вызывает `/predict` по HTTP. Stub используется только в
изолированном API-тесте через dependency override.

## Ограничения и следующие этапы

Сейчас нет автоматического получения погоды, DVC, MLflow, Airflow, PostgreSQL,
drift monitoring, автоматического переобучения, Kubernetes, Argo CD и UI. Данные
охватывают один город и один год; test состоит только из осени, а пики спроса
часто недооцениваются. Подробности и сценарий показа преподавателю находятся в
[`docs/teacher_demo_ru.md`](docs/teacher_demo_ru.md).
