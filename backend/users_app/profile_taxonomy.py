from copy import deepcopy

from .models import CustomerProfile


PROFILE_WIZARD_STEPS = [
    {"id": 1, "key": "skin_type", "title": "Тип кожи", "description": "Выберите ваш тип кожи.", "optional": False},
    {"id": 2, "key": "goals", "title": "Цели", "description": "Что вы хотите улучшить в первую очередь?", "optional": False},
    {"id": 3, "key": "avoid_flags", "title": "Избегать", "description": "Ингредиенты и категории, которых вы хотите избегать.", "optional": True},
    {"id": 4, "key": "budget", "title": "Бюджет", "description": "Комфортный ценовой диапазон для рекомендаций.", "optional": True},
    {"id": 5, "key": "hair_profile", "title": "Волосы", "description": "Базовые предпочтения по уходу за волосами.", "optional": True},
    {"id": 6, "key": "makeup_profile", "title": "Макияж", "description": "Предпочтения по покрытию и макияжу.", "optional": True},
    {"id": 7, "key": "fragrance_profile", "title": "Парфюм", "description": "Любимые ноты и интенсивность аромата.", "optional": True},
]

SKIN_TYPE_OPTIONS = [
    {"value": CustomerProfile.SkinType.OILY, "label": "Жирная", "aliases": ["oily"]},
    {"value": CustomerProfile.SkinType.COMBINATION, "label": "Комбинированная", "aliases": ["combination"]},
    {"value": CustomerProfile.SkinType.DRY, "label": "Сухая", "aliases": ["dry"]},
    {"value": CustomerProfile.SkinType.NORMAL, "label": "Нормальная", "aliases": ["normal"]},
    {"value": CustomerProfile.SkinType.SENSITIVE, "label": "Чувствительная", "aliases": ["sensitive"]},
]

GOAL_OPTIONS = [
    {"value": "hydration", "label": "Увлажнение", "aliases": ["moisturizing"]},
    {"value": "anti_aging", "label": "Антивозрастной уход", "aliases": ["anti_age", "aging"]},
    {"value": "acne", "label": "Против акне", "aliases": ["blemishes", "cleansing"]},
    {"value": "brightening", "label": "Сияние и тон", "aliases": ["glow", "even_tone", "pigmentation"]},
    {"value": "spf", "label": "Защита SPF", "aliases": ["sun_protection"]},
    {"value": "soothing", "label": "Успокоение", "aliases": ["sensitivity", "calming"]},
]

AVOID_FLAG_OPTIONS = [
    {"value": "fragrance", "label": "Отдушки", "aliases": ["perfume"]},
    {"value": "alcohol", "label": "Спирт", "aliases": []},
    {"value": "essential_oils", "label": "Эфирные масла", "aliases": ["essential oils"]},
    {"value": "parabens", "label": "Парабены", "aliases": []},
    {"value": "silicones", "label": "Силиконы", "aliases": []},
    {"value": "gluten", "label": "Глютен", "aliases": []},
]

BUDGET_OPTIONS = [
    {"value": CustomerProfile.Budget.LOW, "label": "До 2 500 ₸", "min": 500, "max": 2500, "currency": "KZT"},
    {"value": CustomerProfile.Budget.MEDIUM, "label": "2 500 – 7 500 ₸", "min": 2500, "max": 7500, "currency": "KZT"},
    {"value": CustomerProfile.Budget.HIGH, "label": "От 7 500 ₸", "min": 7500, "max": None, "currency": "KZT"},
]

HAIR_TYPE_OPTIONS = [
    {"value": "straight", "label": "Прямые", "aliases": []},
    {"value": "wavy", "label": "Волнистые", "aliases": []},
    {"value": "curly", "label": "Кудрявые", "aliases": []},
    {"value": "coily", "label": "Афро", "aliases": ["coils"]},
]

HAIR_CONCERN_OPTIONS = [
    {"value": "hair_loss", "label": "Выпадение", "aliases": []},
    {"value": "repair", "label": "Секущиеся концы", "aliases": ["damage", "split_ends"]},
    {"value": "dryness", "label": "Сухость", "aliases": []},
    {"value": "oiliness", "label": "Жирность", "aliases": []},
    {"value": "flakes", "label": "Перхоть", "aliases": ["dandruff"]},
    {"value": "volume", "label": "Объем", "aliases": []},
]

COVERAGE_OPTIONS = [
    {"value": "light", "label": "Легкое", "aliases": []},
    {"value": "medium", "label": "Среднее", "aliases": []},
    {"value": "full", "label": "Плотное", "aliases": ["full_coverage"]},
]

FRAGRANCE_NOTE_OPTIONS = [
    {"value": "citrus", "label": "Цитрус", "aliases": ["citrusy"]},
    {"value": "floral", "label": "Цветочные", "aliases": ["flowers"]},
    {"value": "woody", "label": "Древесные", "aliases": ["wood"]},
    {"value": "oriental", "label": "Восточные", "aliases": ["amber"]},
    {"value": "fresh", "label": "Свежие", "aliases": ["clean"]},
    {"value": "spicy", "label": "Пряные", "aliases": []},
]

INTENSITY_OPTIONS = [
    {"value": "light", "label": "Легкий", "aliases": []},
    {"value": "medium", "label": "Средний", "aliases": []},
    {"value": "strong", "label": "Интенсивный", "aliases": ["intense"]},
]


PROFILE_TAXONOMY_PAYLOAD = {
    "steps": PROFILE_WIZARD_STEPS,
    "skin_types": SKIN_TYPE_OPTIONS,
    "goals": GOAL_OPTIONS,
    "avoid_flags": AVOID_FLAG_OPTIONS,
    "budget_options": BUDGET_OPTIONS,
    "hair_types": HAIR_TYPE_OPTIONS,
    "hair_concerns": HAIR_CONCERN_OPTIONS,
    "coverage_options": COVERAGE_OPTIONS,
    "fragrance_notes": FRAGRANCE_NOTE_OPTIONS,
    "intensity_options": INTENSITY_OPTIONS,
}


def get_profile_taxonomy_payload() -> dict[str, object]:
    return deepcopy(PROFILE_TAXONOMY_PAYLOAD)
