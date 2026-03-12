ROADMAP_STEP_META_BY_TYPE = {
    "cleanser": {
        "points": 120,
        "why": "Базовый шаг для стабильной рутины.",
        "improves": "Очищение и подготовка кожи.",
        "benefit": "Первые изменения обычно заметны в течение недели.",
    },
    "toner": {
        "points": 90,
        "why": "Помогает выровнять баланс после очищения.",
        "improves": "Комфорт и текстура кожи.",
        "benefit": "Кожа выглядит более ровной и спокойной.",
    },
    "serum": {
        "points": 140,
        "why": "Целевой шаг под вашу текущую задачу.",
        "improves": "Выраженность ключевой проблемы.",
        "benefit": "Результат обычно проявляется через 2-4 недели.",
    },
    "moisturizer": {
        "points": 130,
        "why": "Закрепляет эффект предыдущих шагов.",
        "improves": "Защитный барьер и эластичность.",
        "benefit": "Меньше сухости и дискомфорта.",
    },
    "spf": {
        "points": 190,
        "why": "Ключевой этап дневной защиты кожи.",
        "improves": "Профилактику пигментации и фотостарения.",
        "benefit": "Долгосрочная защита результата ухода.",
    },
    "shampoo": {
        "points": 100,
        "why": "Основа регулярного ухода за волосами.",
        "improves": "Состояние кожи головы.",
        "benefit": "Чистота и комфорт между мытьем.",
    },
    "conditioner": {
        "points": 110,
        "why": "Нужен для защиты длины после очищения.",
        "improves": "Мягкость и управляемость волос.",
        "benefit": "Меньше спутывания и ломкости.",
    },
    "hair_mask": {
        "points": 150,
        "why": "Усиливает базовый уход раз в неделю.",
        "improves": "Плотность и восстановление длины.",
        "benefit": "Волосы выглядят более гладкими.",
    },
    "hair_oil": {
        "points": 130,
        "why": "Защищает длину от пересушивания.",
        "improves": "Гладкость и блеск.",
        "benefit": "Меньше пушения и сухости кончиков.",
    },
    "scalp_serum": {
        "points": 145,
        "why": "Целевой уход за кожей головы.",
        "improves": "Баланс и комфорт кожи головы.",
        "benefit": "Повышает эффективность всей рутины.",
    },
}


DEFAULT_ROADMAP_STEP_META = {
    "points": 100,
    "why": "Персональный шаг подобран на основе ваших данных.",
    "improves": "Результат вашей рутины.",
    "benefit": "Улучшения обычно заметны при регулярном использовании.",
}


def get_roadmap_step_meta(product_type: str | None) -> dict[str, str | int]:
    normalized = str(product_type or "").strip()
    if normalized in ROADMAP_STEP_META_BY_TYPE:
        return ROADMAP_STEP_META_BY_TYPE[normalized]
    return DEFAULT_ROADMAP_STEP_META


def build_roadmap_step_presentation(product_type: str | None, title: str, description: str) -> dict[str, str | int]:
    meta = get_roadmap_step_meta(product_type)
    return {
        "title": title,
        "description": description,
        "points": int(meta["points"]),
        "why": str(meta["why"]),
        "improves": str(meta["improves"]),
        "benefit": str(meta["benefit"]),
    }
