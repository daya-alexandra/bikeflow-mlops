# Итог интеграции контракта

Исходная ревизия участника A в PR №6 подтвердила признаки, единицы, временной split
и необходимость `day_of_week`, но оставила открытыми диапазоны, timezone и способ
доставки артефакта. В интеграционной ветке приняты provisional-решения:

- диапазоны API и ML читаются из одного `FEATURE_CONTRACT`;
- timezone обязательна, время нормализуется в `Asia/Seoul`;
- сезон выводится в canonical-регистре (`Winter`, `Spring`, `Summer`, `Autumn`);
- HGB выбрана по validation MAE, test используется только для финальной оценки;
- estimator и preprocessing сохранены как один `InferencePipeline`;
- `BIKEFLOW_MODEL_PATH` задаёт артефакт, Compose монтирует его read-only;
- production dependency использует настоящую модель, stub остался только для DI-
  теста;
- PyTorch отсутствует в serving-образе.

Эти пункты нужно подтвердить участнику A перед объявлением контракта окончательным.
