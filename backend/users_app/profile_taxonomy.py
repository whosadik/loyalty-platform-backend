from copy import deepcopy

from backend.request_language import AppLanguage, normalize_language

from .models import CustomerProfile


PROFILE_TAXONOMY_PAYLOADS: dict[AppLanguage, dict[str, object]] = {
    "ru": {
        "steps": [
            {
                "id": 1,
                "key": "skin_type",
                "title": "Тип кожи",
                "description": "Выберите ваш тип кожи.",
                "optional": False,
            },
            {
                "id": 2,
                "key": "goals",
                "title": "Цели ухода",
                "description": "Что вы хотите улучшить в первую очередь?",
                "optional": False,
            },
            {
                "id": 3,
                "key": "avoid_flags",
                "title": "Избегать",
                "description": "Ингредиенты и категории, которых вы хотите избегать.",
                "optional": True,
            },
            {
                "id": 4,
                "key": "budget",
                "title": "Бюджет",
                "description": "Комфортный ценовой диапазон для рекомендаций.",
                "optional": True,
            },
            {
                "id": 5,
                "key": "hair_profile",
                "title": "Волосы",
                "description": "Базовые предпочтения по уходу за волосами.",
                "optional": True,
            },
            {
                "id": 6,
                "key": "makeup_profile",
                "title": "Макияж",
                "description": "Предпочтения по покрытию и макияжу.",
                "optional": True,
            },
            {
                "id": 7,
                "key": "fragrance_profile",
                "title": "Ароматы",
                "description": "Любимые ноты и интенсивность аромата.",
                "optional": True,
            },
        ],
        "skin_types": [
            {"value": CustomerProfile.SkinType.OILY, "label": "Жирная", "aliases": ["oily", "майлы"]},
            {
                "value": CustomerProfile.SkinType.COMBINATION,
                "label": "Комбинированная",
                "aliases": ["combination", "аралас"],
            },
            {"value": CustomerProfile.SkinType.DRY, "label": "Сухая", "aliases": ["dry", "құрғақ"]},
            {
                "value": CustomerProfile.SkinType.NORMAL,
                "label": "Нормальная",
                "aliases": ["normal", "қалыпты"],
            },
            {
                "value": CustomerProfile.SkinType.SENSITIVE,
                "label": "Чувствительная",
                "aliases": ["sensitive", "сезімтал"],
            },
        ],
        "goals": [
            {
                "value": "hydration",
                "label": "Увлажнение",
                "aliases": ["moisturizing", "hydration", "ылғалдандыру"],
            },
            {
                "value": "anti_aging",
                "label": "Антивозрастной уход",
                "aliases": ["anti_age", "aging", "қартаюға қарсы"],
            },
            {
                "value": "acne",
                "label": "Против акне",
                "aliases": ["blemishes", "acne", "безеуге қарсы"],
            },
            {
                "value": "brightening",
                "label": "Сияние и тон",
                "aliases": ["glow", "brightening", "жарқырау"],
            },
            {
                "value": "spf",
                "label": "Защита SPF",
                "aliases": ["sun_protection", "spf", "spf қорғаныс"],
            },
            {
                "value": "soothing",
                "label": "Успокоение",
                "aliases": ["sensitivity", "calming", "тыныштандыру"],
            },
        ],
        "avoid_flags": [
            {
                "value": "fragrance",
                "label": "Отдушки",
                "aliases": ["perfume", "fragrance", "хош иіс"],
            },
            {"value": "alcohol", "label": "Спирт", "aliases": ["alcohol", "спирт"]},
            {
                "value": "essential_oils",
                "label": "Эфирные масла",
                "aliases": ["essential oils", "эфир майлары"],
            },
            {"value": "parabens", "label": "Парабены", "aliases": ["parabens"]},
            {"value": "silicones", "label": "Силиконы", "aliases": ["silicones"]},
            {"value": "gluten", "label": "Глютен", "aliases": ["gluten"]},
        ],
        "budget_options": [
            {"value": CustomerProfile.Budget.LOW, "label": "До 2 500 ₸", "min": 500, "max": 2500, "currency": "KZT"},
            {
                "value": CustomerProfile.Budget.MEDIUM,
                "label": "2 500 - 7 500 ₸",
                "min": 2500,
                "max": 7500,
                "currency": "KZT",
            },
            {"value": CustomerProfile.Budget.HIGH, "label": "От 7 500 ₸", "min": 7500, "max": None, "currency": "KZT"},
        ],
        "hair_types": [
            {"value": "straight", "label": "Прямые", "aliases": ["straight", "тік"]},
            {"value": "wavy", "label": "Волнистые", "aliases": ["wavy", "толқынды"]},
            {"value": "curly", "label": "Кудрявые", "aliases": ["curly", "бұйра"]},
            {"value": "coily", "label": "Афро", "aliases": ["coily", "coils"]},
        ],
        "hair_concerns": [
            {"value": "hair_loss", "label": "Выпадение", "aliases": ["hair loss", "түсу"]},
            {
                "value": "repair",
                "label": "Восстановление",
                "aliases": ["damage", "split_ends", "қалпына келтіру"],
            },
            {"value": "dryness", "label": "Сухость", "aliases": ["dryness", "құрғақтық"]},
            {"value": "oiliness", "label": "Жирность", "aliases": ["oiliness", "майлылық"]},
            {"value": "flakes", "label": "Перхоть", "aliases": ["dandruff", "қайызғақ"]},
            {"value": "volume", "label": "Объем", "aliases": ["volume", "көлем"]},
        ],
        "coverage_options": [
            {"value": "light", "label": "Легкое", "aliases": ["light", "жеңіл"]},
            {"value": "medium", "label": "Среднее", "aliases": ["medium", "орташа"]},
            {"value": "full", "label": "Плотное", "aliases": ["full", "full_coverage", "тығыз"]},
        ],
        "fragrance_notes": [
            {"value": "citrus", "label": "Цитрус", "aliases": ["citrus", "цитрус"]},
            {"value": "floral", "label": "Цветочные", "aliases": ["floral", "гүлді"]},
            {"value": "woody", "label": "Древесные", "aliases": ["woody", "ағашты"]},
            {"value": "oriental", "label": "Восточные", "aliases": ["oriental", "amber", "шығыстық"]},
            {"value": "fresh", "label": "Свежие", "aliases": ["fresh", "балғын"]},
            {"value": "spicy", "label": "Пряные", "aliases": ["spicy", "дәмдеуішті"]},
        ],
        "intensity_options": [
            {"value": "light", "label": "Легкий", "aliases": ["light", "жеңіл"]},
            {"value": "medium", "label": "Средний", "aliases": ["medium", "орташа"]},
            {"value": "strong", "label": "Интенсивный", "aliases": ["strong", "intense", "қанық"]},
        ],
    },
    "kk": {
        "steps": [
            {
                "id": 1,
                "key": "skin_type",
                "title": "Тері түрі",
                "description": "Теріңіздің түрін таңдаңыз.",
                "optional": False,
            },
            {
                "id": 2,
                "key": "goals",
                "title": "Күтім мақсаттары",
                "description": "Ең алдымен нені жақсартқыңыз келеді?",
                "optional": False,
            },
            {
                "id": 3,
                "key": "avoid_flags",
                "title": "Болдырмау",
                "description": "Құрамында болмағанын қалайтын ингредиенттер мен санаттар.",
                "optional": True,
            },
            {
                "id": 4,
                "key": "budget",
                "title": "Бюджет",
                "description": "Ұсыныстарға ыңғайлы баға диапазоны.",
                "optional": True,
            },
            {
                "id": 5,
                "key": "hair_profile",
                "title": "Шаш",
                "description": "Шаш күтіміне қатысты негізгі қалаулар.",
                "optional": True,
            },
            {
                "id": 6,
                "key": "makeup_profile",
                "title": "Макияж",
                "description": "Жабын мен макияж бойынша қалаулар.",
                "optional": True,
            },
            {
                "id": 7,
                "key": "fragrance_profile",
                "title": "Хош иіс",
                "description": "Ұнайтын ноталар мен хош иіс қарқындылығы.",
                "optional": True,
            },
        ],
        "skin_types": [
            {"value": CustomerProfile.SkinType.OILY, "label": "Майлы", "aliases": ["oily", "жирная"]},
            {
                "value": CustomerProfile.SkinType.COMBINATION,
                "label": "Аралас",
                "aliases": ["combination", "комбинированная"],
            },
            {"value": CustomerProfile.SkinType.DRY, "label": "Құрғақ", "aliases": ["dry", "сухая"]},
            {
                "value": CustomerProfile.SkinType.NORMAL,
                "label": "Қалыпты",
                "aliases": ["normal", "нормальная"],
            },
            {
                "value": CustomerProfile.SkinType.SENSITIVE,
                "label": "Сезімтал",
                "aliases": ["sensitive", "чувствительная"],
            },
        ],
        "goals": [
            {
                "value": "hydration",
                "label": "Ылғалдандыру",
                "aliases": ["moisturizing", "hydration", "увлажнение"],
            },
            {
                "value": "anti_aging",
                "label": "Қартаюға қарсы күтім",
                "aliases": ["anti_age", "aging", "антивозрастной уход"],
            },
            {
                "value": "acne",
                "label": "Безеуге қарсы",
                "aliases": ["blemishes", "acne", "против акне"],
            },
            {
                "value": "brightening",
                "label": "Жарқырау мен реңк",
                "aliases": ["glow", "brightening", "сияние"],
            },
            {
                "value": "spf",
                "label": "SPF қорғаныс",
                "aliases": ["sun_protection", "spf", "защита spf"],
            },
            {
                "value": "soothing",
                "label": "Тыныштандыру",
                "aliases": ["sensitivity", "calming", "успокоение"],
            },
        ],
        "avoid_flags": [
            {
                "value": "fragrance",
                "label": "Хош иіс",
                "aliases": ["perfume", "fragrance", "отдушки"],
            },
            {"value": "alcohol", "label": "Спирт", "aliases": ["alcohol", "спирт"]},
            {
                "value": "essential_oils",
                "label": "Эфир майлары",
                "aliases": ["essential oils", "эфирные масла"],
            },
            {"value": "parabens", "label": "Парабендер", "aliases": ["parabens", "парабены"]},
            {"value": "silicones", "label": "Силикондар", "aliases": ["silicones", "силиконы"]},
            {"value": "gluten", "label": "Глютен", "aliases": ["gluten"]},
        ],
        "budget_options": [
            {"value": CustomerProfile.Budget.LOW, "label": "2 500 ₸ дейін", "min": 500, "max": 2500, "currency": "KZT"},
            {
                "value": CustomerProfile.Budget.MEDIUM,
                "label": "2 500 - 7 500 ₸",
                "min": 2500,
                "max": 7500,
                "currency": "KZT",
            },
            {"value": CustomerProfile.Budget.HIGH, "label": "7 500 ₸ бастап", "min": 7500, "max": None, "currency": "KZT"},
        ],
        "hair_types": [
            {"value": "straight", "label": "Тік", "aliases": ["straight", "прямые"]},
            {"value": "wavy", "label": "Толқынды", "aliases": ["wavy", "волнистые"]},
            {"value": "curly", "label": "Бұйра", "aliases": ["curly", "кудрявые"]},
            {"value": "coily", "label": "Афро", "aliases": ["coily", "afro"]},
        ],
        "hair_concerns": [
            {"value": "hair_loss", "label": "Түсу", "aliases": ["hair loss", "выпадение"]},
            {
                "value": "repair",
                "label": "Қалпына келтіру",
                "aliases": ["damage", "split_ends", "восстановление"],
            },
            {"value": "dryness", "label": "Құрғақтық", "aliases": ["dryness", "сухость"]},
            {"value": "oiliness", "label": "Майлылық", "aliases": ["oiliness", "жирность"]},
            {"value": "flakes", "label": "Қайызғақ", "aliases": ["dandruff", "перхоть"]},
            {"value": "volume", "label": "Көлем", "aliases": ["volume", "объем"]},
        ],
        "coverage_options": [
            {"value": "light", "label": "Жеңіл", "aliases": ["light", "легкое"]},
            {"value": "medium", "label": "Орташа", "aliases": ["medium", "среднее"]},
            {"value": "full", "label": "Тығыз", "aliases": ["full", "full_coverage", "плотное"]},
        ],
        "fragrance_notes": [
            {"value": "citrus", "label": "Цитрус", "aliases": ["citrus"]},
            {"value": "floral", "label": "Гүлді", "aliases": ["floral", "цветочные"]},
            {"value": "woody", "label": "Ағашты", "aliases": ["woody", "древесные"]},
            {"value": "oriental", "label": "Шығыстық", "aliases": ["oriental", "amber", "восточные"]},
            {"value": "fresh", "label": "Балғын", "aliases": ["fresh", "свежие"]},
            {"value": "spicy", "label": "Дәмдеуішті", "aliases": ["spicy", "пряные"]},
        ],
        "intensity_options": [
            {"value": "light", "label": "Жеңіл", "aliases": ["light", "легкий"]},
            {"value": "medium", "label": "Орташа", "aliases": ["medium", "средний"]},
            {"value": "strong", "label": "Қанық", "aliases": ["strong", "intense", "интенсивный"]},
        ],
    },
    "en": {
        "steps": [
            {
                "id": 1,
                "key": "skin_type",
                "title": "Skin type",
                "description": "Choose your skin type.",
                "optional": False,
            },
            {
                "id": 2,
                "key": "goals",
                "title": "Care goals",
                "description": "What would you like to improve first?",
                "optional": False,
            },
            {
                "id": 3,
                "key": "avoid_flags",
                "title": "Avoid",
                "description": "Ingredients and categories you prefer to avoid.",
                "optional": True,
            },
            {
                "id": 4,
                "key": "budget",
                "title": "Budget",
                "description": "A comfortable price range for recommendations.",
                "optional": True,
            },
            {
                "id": 5,
                "key": "hair_profile",
                "title": "Hair",
                "description": "Basic hair care preferences.",
                "optional": True,
            },
            {
                "id": 6,
                "key": "makeup_profile",
                "title": "Makeup",
                "description": "Coverage and makeup preferences.",
                "optional": True,
            },
            {
                "id": 7,
                "key": "fragrance_profile",
                "title": "Fragrance",
                "description": "Favorite notes and fragrance intensity.",
                "optional": True,
            },
        ],
        "skin_types": [
            {"value": CustomerProfile.SkinType.OILY, "label": "Oily", "aliases": ["жирная", "майлы"]},
            {
                "value": CustomerProfile.SkinType.COMBINATION,
                "label": "Combination",
                "aliases": ["комбинированная", "аралас"],
            },
            {"value": CustomerProfile.SkinType.DRY, "label": "Dry", "aliases": ["сухая", "құрғақ"]},
            {"value": CustomerProfile.SkinType.NORMAL, "label": "Normal", "aliases": ["нормальная", "қалыпты"]},
            {
                "value": CustomerProfile.SkinType.SENSITIVE,
                "label": "Sensitive",
                "aliases": ["чувствительная", "сезімтал"],
            },
        ],
        "goals": [
            {"value": "hydration", "label": "Hydration", "aliases": ["увлажнение", "ылғалдандыру"]},
            {
                "value": "anti_aging",
                "label": "Anti-aging care",
                "aliases": ["anti_age", "антивозрастной уход", "қартаюға қарсы"],
            },
            {"value": "acne", "label": "Acne care", "aliases": ["против акне", "безеуге қарсы"]},
            {"value": "brightening", "label": "Glow and tone", "aliases": ["сияние", "жарқырау"]},
            {"value": "spf", "label": "SPF protection", "aliases": ["защита spf", "spf қорғаныс"]},
            {"value": "soothing", "label": "Soothing", "aliases": ["успокоение", "тыныштандыру"]},
        ],
        "avoid_flags": [
            {"value": "fragrance", "label": "Fragrance", "aliases": ["отдушки", "хош иіс"]},
            {"value": "alcohol", "label": "Alcohol", "aliases": ["спирт"]},
            {"value": "essential_oils", "label": "Essential oils", "aliases": ["эфирные масла", "эфир майлары"]},
            {"value": "parabens", "label": "Parabens", "aliases": ["парабены", "парабендер"]},
            {"value": "silicones", "label": "Silicones", "aliases": ["силиконы", "силикондар"]},
            {"value": "gluten", "label": "Gluten", "aliases": ["глютен"]},
        ],
        "budget_options": [
            {"value": CustomerProfile.Budget.LOW, "label": "Up to 2,500 ₸", "min": 500, "max": 2500, "currency": "KZT"},
            {
                "value": CustomerProfile.Budget.MEDIUM,
                "label": "2,500 - 7,500 ₸",
                "min": 2500,
                "max": 7500,
                "currency": "KZT",
            },
            {"value": CustomerProfile.Budget.HIGH, "label": "From 7,500 ₸", "min": 7500, "max": None, "currency": "KZT"},
        ],
        "hair_types": [
            {"value": "straight", "label": "Straight", "aliases": ["прямые", "тік"]},
            {"value": "wavy", "label": "Wavy", "aliases": ["волнистые", "толқынды"]},
            {"value": "curly", "label": "Curly", "aliases": ["кудрявые", "бұйра"]},
            {"value": "coily", "label": "Coily", "aliases": ["афро"]},
        ],
        "hair_concerns": [
            {"value": "hair_loss", "label": "Hair loss", "aliases": ["выпадение", "түсу"]},
            {"value": "repair", "label": "Repair", "aliases": ["восстановление", "қалпына келтіру"]},
            {"value": "dryness", "label": "Dryness", "aliases": ["сухость", "құрғақтық"]},
            {"value": "oiliness", "label": "Oiliness", "aliases": ["жирность", "майлылық"]},
            {"value": "flakes", "label": "Dandruff", "aliases": ["перхоть", "қайызғақ"]},
            {"value": "volume", "label": "Volume", "aliases": ["объем", "көлем"]},
        ],
        "coverage_options": [
            {"value": "light", "label": "Light", "aliases": ["легкое", "жеңіл"]},
            {"value": "medium", "label": "Medium", "aliases": ["среднее", "орташа"]},
            {"value": "full", "label": "Full", "aliases": ["плотное", "тығыз", "full_coverage"]},
        ],
        "fragrance_notes": [
            {"value": "citrus", "label": "Citrus", "aliases": ["цитрус"]},
            {"value": "floral", "label": "Floral", "aliases": ["цветочные", "гүлді"]},
            {"value": "woody", "label": "Woody", "aliases": ["древесные", "ағашты"]},
            {"value": "oriental", "label": "Oriental", "aliases": ["восточные", "шығыстық", "amber"]},
            {"value": "fresh", "label": "Fresh", "aliases": ["свежие", "балғын"]},
            {"value": "spicy", "label": "Spicy", "aliases": ["пряные", "дәмдеуішті"]},
        ],
        "intensity_options": [
            {"value": "light", "label": "Light", "aliases": ["легкий", "жеңіл"]},
            {"value": "medium", "label": "Medium", "aliases": ["средний", "орташа"]},
            {"value": "strong", "label": "Strong", "aliases": ["интенсивный", "қанық", "intense"]},
        ],
    },
}


def get_profile_taxonomy_payload(language: AppLanguage = "ru") -> dict[str, object]:
    return deepcopy(PROFILE_TAXONOMY_PAYLOADS[normalize_language(language)])
