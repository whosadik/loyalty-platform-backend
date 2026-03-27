from __future__ import annotations

from backend.request_language import AppLanguage

ROADMAP_STEP_TEXTS_BY_TYPE: dict[str, dict[AppLanguage, dict[str, str]]] = {
    "cleanser": {
        "ru": {
            "title": "Очищение",
            "description": "Начните с мягкого очищающего средства для вашего типа кожи.",
        },
        "kk": {
            "title": "Тазарту",
            "description": "Тері түріңізге сай жұмсақ тазартқыштан бастаңыз.",
        },
        "en": {
            "title": "Cleanse",
            "description": "Start with a gentle cleanser for your skin type.",
        },
    },
    "toner": {
        "ru": {
            "title": "Тонизирование",
            "description": "Восстановите баланс кожи с помощью подходящего тоника.",
        },
        "kk": {
            "title": "Тонерлеу",
            "description": "Терінің тепе-теңдігін лайықты тонермен қалпына келтіріңіз.",
        },
        "en": {
            "title": "Tone",
            "description": "Restore skin balance with a suitable toner.",
        },
    },
    "serum": {
        "ru": {
            "title": "Сыворотка",
            "description": "Добавьте активный этап для решения текущей задачи кожи.",
        },
        "kk": {
            "title": "Сарысу",
            "description": "Ағымдағы тері мақсатыңызға арналған белсенді қадам қосыңыз.",
        },
        "en": {
            "title": "Serum",
            "description": "Add an active step to address your current skin goal.",
        },
    },
    "moisturizer": {
        "ru": {
            "title": "Увлажнение",
            "description": "Закрепите уход увлажняющим средством для поддержки кожного барьера.",
        },
        "kk": {
            "title": "Ылғалдандыру",
            "description": "Тері тосқауылын қолдау үшін күтімді ылғалдандырғышпен бекітіңіз.",
        },
        "en": {
            "title": "Moisturize",
            "description": "Seal in the routine with a moisturizer that supports the skin barrier.",
        },
    },
    "spf": {
        "ru": {
            "title": "SPF-защита",
            "description": "Завершите дневной уход средством с солнцезащитой.",
        },
        "kk": {
            "title": "SPF қорғаныс",
            "description": "Күндізгі күтімді күннен қорғайтын құралмен аяқтаңыз.",
        },
        "en": {
            "title": "SPF protection",
            "description": "Finish your daytime routine with sun protection.",
        },
    },
    "shampoo": {
        "ru": {
            "title": "Очищение кожи головы",
            "description": "Подберите шампунь по типу кожи головы и частоте мытья.",
        },
        "kk": {
            "title": "Бас терісін тазарту",
            "description": "Бас терісінің түрі мен жуу жиілігіне сай сусабын таңдаңыз.",
        },
        "en": {
            "title": "Scalp cleanse",
            "description": "Choose a shampoo based on your scalp type and wash frequency.",
        },
    },
    "conditioner": {
        "ru": {
            "title": "Кондиционирование",
            "description": "Используйте кондиционер для защиты длины и гладкости волос.",
        },
        "kk": {
            "title": "Кондиционерлеу",
            "description": "Шаш ұзындығын қорғап, тегістік беру үшін кондиционер қолданыңыз.",
        },
        "en": {
            "title": "Condition",
            "description": "Use conditioner to protect your lengths and add smoothness.",
        },
    },
    "hair_mask": {
        "ru": {
            "title": "Маска для волос",
            "description": "Добавьте еженедельный восстанавливающий этап ухода.",
        },
        "kk": {
            "title": "Шашқа арналған маска",
            "description": "Апталық қалпына келтіретін күтім қадамын қосыңыз.",
        },
        "en": {
            "title": "Hair mask",
            "description": "Add a weekly restorative care step.",
        },
    },
    "hair_oil": {
        "ru": {
            "title": "Масло для волос",
            "description": "Используйте масло для защиты длины, гладкости и блеска.",
        },
        "kk": {
            "title": "Шаш майы",
            "description": "Ұзындықты қорғап, тегістік пен жылтыр беру үшін май қолданыңыз.",
        },
        "en": {
            "title": "Hair oil",
            "description": "Use oil to protect the lengths, smooth frizz, and add shine.",
        },
    },
    "scalp_serum": {
        "ru": {
            "title": "Сыворотка для кожи головы",
            "description": "Добавьте целевой уход для кожи головы и корней волос.",
        },
        "kk": {
            "title": "Бас терісіне арналған сарысу",
            "description": "Бас терісі мен түпке бағытталған күтім қадамын қосыңыз.",
        },
        "en": {
            "title": "Scalp serum",
            "description": "Add targeted care for the scalp and roots.",
        },
    },
    "foundation": {
        "ru": {
            "title": "Тон",
            "description": "Подберите основу, подходящую по тону и типу кожи.",
        },
        "kk": {
            "title": "Тон",
            "description": "Теріңіздің реңкі мен түріне сай негіз таңдаңыз.",
        },
        "en": {
            "title": "Foundation",
            "description": "Choose a base that matches your tone and skin type.",
        },
    },
    "eyeshadow": {
        "ru": {
            "title": "Акцент для глаз",
            "description": "Добавьте продукт для акцента и завершения макияжа глаз.",
        },
        "kk": {
            "title": "Көзге акцент",
            "description": "Көз макияжын толықтыратын акценттік өнім қосыңыз.",
        },
        "en": {
            "title": "Eye accent",
            "description": "Add a product that defines and finishes your eye makeup.",
        },
    },
    "lipstick": {
        "ru": {
            "title": "Акцент для губ",
            "description": "Завершите образ подходящим оттенком для губ.",
        },
        "kk": {
            "title": "Ерінге акцент",
            "description": "Бейнеңізді лайықты ерін реңкімен аяқтаңыз.",
        },
        "en": {
            "title": "Lip accent",
            "description": "Finish the look with a lip shade that suits you.",
        },
    },
    "perfume": {
        "ru": {
            "title": "Парфюмерная база",
            "description": "Подберите аромат, который соответствует вашим предпочтениям.",
        },
        "kk": {
            "title": "Хош иіс негізі",
            "description": "Қалауыңызға сай келетін хош иісті таңдаңыз.",
        },
        "en": {
            "title": "Fragrance base",
            "description": "Choose a fragrance that matches your preferences.",
        },
    },
}

ROADMAP_STEP_META_BY_TYPE: dict[str, dict[AppLanguage, dict[str, str | int]]] = {
    "cleanser": {
        "ru": {
            "points": 120,
            "why": "Базовый шаг для стабильной рутины.",
            "improves": "Очищение и подготовку кожи.",
            "benefit": "Первые изменения обычно заметны в течение недели.",
        },
        "kk": {
            "points": 120,
            "why": "Тұрақты рутинаға арналған негізгі қадам.",
            "improves": "Теріні тазарту мен дайындауды.",
            "benefit": "Алғашқы өзгерістер әдетте бір апта ішінде байқалады.",
        },
        "en": {
            "points": 120,
            "why": "A core step for a stable routine.",
            "improves": "Cleansing and skin prep.",
            "benefit": "First changes are usually noticeable within a week.",
        },
    },
    "toner": {
        "ru": {
            "points": 90,
            "why": "Помогает выровнять баланс после очищения.",
            "improves": "Комфорт и текстуру кожи.",
            "benefit": "Кожа выглядит более ровной и спокойной.",
        },
        "kk": {
            "points": 90,
            "why": "Тазартудан кейін тепе-теңдікті қалпына келтіруге көмектеседі.",
            "improves": "Терінің жайлылығы мен құрылымын.",
            "benefit": "Тері біркелкі әрі тыныш көрінеді.",
        },
        "en": {
            "points": 90,
            "why": "Helps rebalance the skin after cleansing.",
            "improves": "Comfort and skin texture.",
            "benefit": "Skin looks calmer and more even.",
        },
    },
    "serum": {
        "ru": {
            "points": 140,
            "why": "Целевой шаг под вашу текущую задачу.",
            "improves": "Выраженность ключевой проблемы.",
            "benefit": "Результат обычно проявляется через 2-4 недели.",
        },
        "kk": {
            "points": 140,
            "why": "Ағымдағы мақсатыңызға бағытталған қадам.",
            "improves": "Негізгі мәселенің айқындылығын.",
            "benefit": "Нәтиже әдетте 2-4 аптада көріне бастайды.",
        },
        "en": {
            "points": 140,
            "why": "A targeted step for your current concern.",
            "improves": "The visibility of the main concern.",
            "benefit": "Results usually show in 2-4 weeks.",
        },
    },
    "moisturizer": {
        "ru": {
            "points": 130,
            "why": "Закрепляет эффект предыдущих шагов.",
            "improves": "Защитный барьер и эластичность.",
            "benefit": "Меньше сухости и дискомфорта.",
        },
        "kk": {
            "points": 130,
            "why": "Алдыңғы қадамдардың әсерін бекітеді.",
            "improves": "Қорғаныс тосқауылы мен серпімділікті.",
            "benefit": "Құрғақтық пен жайсыздық азаяды.",
        },
        "en": {
            "points": 130,
            "why": "Locks in the effect of previous steps.",
            "improves": "Barrier support and elasticity.",
            "benefit": "Less dryness and discomfort.",
        },
    },
    "spf": {
        "ru": {
            "points": 190,
            "why": "Ключевой этап дневной защиты кожи.",
            "improves": "Профилактику пигментации и фотостарения.",
            "benefit": "Защищает результат ухода в долгую.",
        },
        "kk": {
            "points": 190,
            "why": "Күндізгі қорғаныстың негізгі кезеңі.",
            "improves": "Пигментация мен фотокартаюдың алдын алуды.",
            "benefit": "Күтім нәтижесін ұзақ мерзімге қорғайды.",
        },
        "en": {
            "points": 190,
            "why": "A key step in daytime protection.",
            "improves": "Prevention of pigmentation and photoaging.",
            "benefit": "Protects your routine results long term.",
        },
    },
    "shampoo": {
        "ru": {
            "points": 100,
            "why": "Основа регулярного ухода за волосами.",
            "improves": "Состояние кожи головы.",
            "benefit": "Больше чистоты и комфорта между мытьем.",
        },
        "kk": {
            "points": 100,
            "why": "Шаш күтімінің тұрақты негізі.",
            "improves": "Бас терісінің жағдайын.",
            "benefit": "Жуу аралығындағы тазалық пен жайлылықты арттырады.",
        },
        "en": {
            "points": 100,
            "why": "The foundation of regular hair care.",
            "improves": "Scalp condition.",
            "benefit": "More cleanliness and comfort between washes.",
        },
    },
    "conditioner": {
        "ru": {
            "points": 110,
            "why": "Нужен для защиты длины после очищения.",
            "improves": "Мягкость и управляемость волос.",
            "benefit": "Меньше спутывания и ломкости.",
        },
        "kk": {
            "points": 110,
            "why": "Тазартудан кейін ұзындықты қорғау үшін қажет.",
            "improves": "Шаштың жұмсақтығы мен басқарылуын.",
            "benefit": "Шатасу мен сыну азаяды.",
        },
        "en": {
            "points": 110,
            "why": "Needed to protect the lengths after cleansing.",
            "improves": "Softness and manageability.",
            "benefit": "Less tangling and breakage.",
        },
    },
    "hair_mask": {
        "ru": {
            "points": 150,
            "why": "Усиливает базовый уход раз в неделю.",
            "improves": "Плотность и восстановление длины.",
            "benefit": "Волосы выглядят более гладкими.",
        },
        "kk": {
            "points": 150,
            "why": "Апталық күтімді күшейтеді.",
            "improves": "Ұзындықтың тығыздығы мен қалпына келуін.",
            "benefit": "Шаш тегіс көрінеді.",
        },
        "en": {
            "points": 150,
            "why": "Boosts your base care once a week.",
            "improves": "Hair density and repair.",
            "benefit": "Hair looks smoother.",
        },
    },
    "hair_oil": {
        "ru": {
            "points": 130,
            "why": "Защищает длину от пересушивания.",
            "improves": "Гладкость и блеск.",
            "benefit": "Меньше пушения и сухости на концах.",
        },
        "kk": {
            "points": 130,
            "why": "Ұзындықты құрғаудан қорғайды.",
            "improves": "Тегістік пен жылтырды.",
            "benefit": "Пушение мен ұштың құрғауы азаяды.",
        },
        "en": {
            "points": 130,
            "why": "Protects hair lengths from overdrying.",
            "improves": "Smoothness and shine.",
            "benefit": "Less frizz and dry ends.",
        },
    },
    "scalp_serum": {
        "ru": {
            "points": 145,
            "why": "Целевой уход за кожей головы.",
            "improves": "Баланс и комфорт кожи головы.",
            "benefit": "Повышает эффективность всей рутины.",
        },
        "kk": {
            "points": 145,
            "why": "Бас терісіне арналған мақсатты күтім.",
            "improves": "Бас терісінің тепе-теңдігі мен жайлылығын.",
            "benefit": "Бүкіл рутинаның тиімділігін арттырады.",
        },
        "en": {
            "points": 145,
            "why": "A targeted scalp care step.",
            "improves": "Scalp balance and comfort.",
            "benefit": "Improves the efficiency of the whole routine.",
        },
    },
}

DEFAULT_ROADMAP_STEP_META: dict[AppLanguage, dict[str, str | int]] = {
    "ru": {
        "points": 100,
        "why": "Персональный шаг подобран на основе ваших данных.",
        "improves": "Результат вашей рутины.",
        "benefit": "Улучшения обычно заметны при регулярном использовании.",
    },
    "kk": {
        "points": 100,
        "why": "Жеке қадам деректеріңізге сүйеніп таңдалды.",
        "improves": "Рутинаңыздың нәтижесін.",
        "benefit": "Тұрақты қолданғанда жақсару байқалады.",
    },
    "en": {
        "points": 100,
        "why": "A personal step selected from your data.",
        "improves": "Your routine results.",
        "benefit": "Improvements are usually visible with regular use.",
    },
}

DEFAULT_ROADMAP_STEP_TEXTS: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        "title": "Шаг ухода",
        "description": "Персональный шаг, добавленный в ваш roadmap.",
    },
    "kk": {
        "title": "Күтім қадамы",
        "description": "Roadmap-қа қосылған жеке қадам.",
    },
    "en": {
        "title": "Care step",
        "description": "A personal step added to your roadmap.",
    },
}


def _format_fallback_title(product_type: str | None, language: AppLanguage) -> str:
    prepared = str(product_type or "").replace("_", " ").strip()
    if not prepared:
        return str(DEFAULT_ROADMAP_STEP_TEXTS[language]["title"])

    if language == "en":
        return prepared.title()

    return prepared[:1].upper() + prepared[1:]


def get_roadmap_step_presentation(product_type: str | None, language: AppLanguage = "ru") -> dict[str, str]:
    normalized = str(product_type or "").strip()
    if normalized in ROADMAP_STEP_TEXTS_BY_TYPE:
        return ROADMAP_STEP_TEXTS_BY_TYPE[normalized][language]

    default_texts = DEFAULT_ROADMAP_STEP_TEXTS[language]
    return {
        "title": _format_fallback_title(normalized, language),
        "description": str(default_texts["description"]),
    }


def get_roadmap_step_meta(product_type: str | None, language: AppLanguage = "ru") -> dict[str, str | int]:
    normalized = str(product_type or "").strip()
    if normalized in ROADMAP_STEP_META_BY_TYPE:
        return ROADMAP_STEP_META_BY_TYPE[normalized][language]
    return DEFAULT_ROADMAP_STEP_META[language]


def build_roadmap_step_presentation(
    product_type: str | None,
    *,
    title: str | None = None,
    description: str | None = None,
    language: AppLanguage = "ru",
) -> dict[str, str | int]:
    step_texts = get_roadmap_step_presentation(product_type, language)
    meta = get_roadmap_step_meta(product_type, language)
    return {
        "title": str(title or step_texts["title"]),
        "description": str(description or step_texts["description"]),
        "points": int(meta["points"]),
        "why": str(meta["why"]),
        "improves": str(meta["improves"]),
        "benefit": str(meta["benefit"]),
    }
